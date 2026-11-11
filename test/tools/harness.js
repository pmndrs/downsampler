// Tool-fixture harness: builds every fixture per render size, checks each
// against its CPU oracle, then times them.
//
// Timing (SPEC §8, adapted): each sample is one compute pass holding
// `repeats` back-to-back copies of a fixture's dispatches, bracketed by the
// pass's own timestampWrites (begin/end), divided by `repeats`. The passes are
// never empty, which is what made compute markers invalid on Metal. Samples
// are taken in `rounds` rounds, rotating the fixture order each round, after
// a warm-up; the result is the median. 'floor' (one single-invocation
// dispatch) shows the per-dispatch cost floor on the device.
//
/**
 * @typedef {object} Fixture
 * @property {string} name          'family/pass', e.g. 'fsr/luma-pyramid'
 * @property {'exposure'|'shading'} pass
 * @property {number} dispatches
 * @property {( pass: GPUComputePassEncoder ) => void} encodePass  records the fixture's dispatches
 * @property {() => void} [reset]   restores state the fixture reads and writes (FrameInfo)
 * @property {() => Promise<{ ok: boolean, checks: object[] }>} [check]  compares the last run with the oracle
 * @property {() => void} destroy
 */
import { createScene, defaultSizes } from './scene.js';
import { compile, createTexture, median, outputFormats, requestToolsDevice } from './gpu.js';
import { createFsrLumaPyramid, createLumaInputs, createLumaTool, lumaOracle } from './fsr-luma-pyramid.js';
import { createFsrShadingChange, createShadingInputs, createShadingTool } from './fsr-shading-change.js';
import { createUpscalerExposure, createUpscalerExposureResources } from './upscaler-exposure.js';
import { createUpscalerShadingChange, createUpscalerShadingResources } from './upscaler-shading-change.js';
import { generatorPasses } from './generator.js';

export const ratioPairs = [
	[ 'fsr/luma-pyramid', 'upscaler/exposure' ],
	[ 'fsr/luma-pyramid-correct', 'upscaler/exposure' ],
	[ 'fsr/luma-pyramid-correct', 'fsr/luma-pyramid' ],
	[ 'fsr/shading-change-total', 'upscaler/shading-change' ],
	[ 'fsr/shading-change-pyramid', 'upscaler/shading-change' ],
	[ 'fsr/shading-change-pyramid-amd', 'fsr/shading-change-pyramid' ],
	...generatorPasses.flatMap( entry => entry.compareWith.map( other => [ entry.name, other ] ) )
];

async function floorFixture( device ) {

	const module = await compile( device, 'floor', '@group(0) @binding(0) var<storage, read_write> sink: array<u32>;\n@compute @workgroup_size(1) fn main() { sink[0] += 1u; }' );
	const pipeline = await device.createComputePipelineAsync( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
	const buffer = device.createBuffer( { size: 16, usage: GPUBufferUsage.STORAGE } );
	const group = device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: { buffer } } ] } );
	return { name: 'floor', pass: 'floor', dispatches: 1, encodePass( pass ) { pass.setPipeline( pipeline ); pass.setBindGroup( 0, group ); pass.dispatchWorkgroups( 1 ); }, destroy() { buffer.destroy(); } };

}

function createTimer( device ) {

	const querySet = device.createQuerySet( { type: 'timestamp', count: 2 } );
	const resolve = device.createBuffer( { size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC } );
	const readback = device.createBuffer( { size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST } );
	return {
		async time( fixture, repeats ) {

			const encoder = device.createCommandEncoder();
			const pass = encoder.beginComputePass( { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } );
			for ( let i = 0; i < repeats; i ++ ) fixture.encodePass( pass );
			pass.end();
			encoder.resolveQuerySet( querySet, 0, 2, resolve, 0 );
			encoder.copyBufferToBuffer( resolve, 0, readback, 0, 16 );
			device.queue.submit( [ encoder.finish() ] );
			await readback.mapAsync( GPUMapMode.READ );
			const [ start, end ] = new BigUint64Array( readback.getMappedRange().slice( 0 ) );
			readback.unmap();
			if ( start === 0n || end <= start ) throw new Error( `invalid GPU timestamps for ${ fixture.name }: ${ start }, ${ end }` );
			return Number( end - start ) / 1e6 / repeats;

		},
		destroy() { querySet.destroy(); resolve.destroy(); readback.destroy(); }
	};

}

function run( device, fixture, times = 1 ) {

	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	for ( let i = 0; i < times; i ++ ) fixture.encodePass( pass );
	pass.end();
	device.queue.submit( [ encoder.finish() ] );

}

/** All fixtures for one scene, in dependency order (the FSR consumer reads the FSR pyramid's mips). */
async function createFixtures( device, scene, formats, owned, only ) {

	const colour = createTexture( device, { label: 'scene colour', width: scene.width, height: scene.height, format: 'rgba16float', data: scene.colour } );
	const lumaInputs = createLumaInputs( device, scene );
	const exposure = lumaOracle( scene, 'amd' ).frameInfo[ 0 ];
	const shadingInputs = createShadingInputs( device, scene, exposure );
	owned.push( colour, lumaInputs.currentLuma, lumaInputs.farthestDepth, shadingInputs );

	const fixtures = [
		await createUpscalerExposure( device, scene, colour ),
		await createFsrLumaPyramid( device, scene, lumaInputs, formats, { variant: 'amd' } ),
		await createFsrLumaPyramid( device, scene, lumaInputs, formats, { variant: 'correct' } ),
		await createUpscalerShadingChange( device, scene, colour ),
		...await createFsrShadingChange( device, scene, shadingInputs, formats )
	];

	const context = { scene, formats, colour, tools: {
		luma: options => createLumaTool( device, scene, lumaInputs, formats, options ),
		shading: options => createShadingTool( device, scene, shadingInputs, formats, options ),
		upscalerExposure: () => createUpscalerExposureResources( device, scene, colour ),
		upscalerShading: () => createUpscalerShadingResources( device, scene, colour )
	} };
	const pending = [];
	for ( const entry of generatorPasses ) {

		if ( entry.experiment && ! only?.some( name => entry.name.startsWith( name ) ) ) continue;
		const fixture = entry.create ? await entry.create( device, context ) : null;
		if ( fixture ) fixtures.push( fixture ); else pending.push( `${ entry.name } (${ entry.milestone })` );

	}

	fixtures.push( await floorFixture( device ) );
	return { fixtures, pending };

}

/**
 * Fail-before proof for the oracles: after the GPU inputs are uploaded, the
 * oracles' copy of the scene is perturbed (colour, current luma and depth ×(1 + 2^-7),
 * jitter +0.25 px), so every check must fail.
 */
function mutate( scene ) {

	for ( const key of [ 'colour', 'currentLuma', 'hostLuma', 'farthestDepth' ] ) scene[ key ] = scene[ key ].map( value => value * ( 1 + 2 ** - 7 ) );
	scene.jitter = [ scene.jitter[ 0 ] + 0.25, scene.jitter[ 1 ] ];

}

export async function runTools( { sizes = defaultSizes, rounds = 11, repeats = 16, bench = true, only = null, mutation = false, onProgress = () => {} } = {} ) {

	const { device, adapter, userAgent } = await requestToolsDevice();
	const errors = [];
	device.addEventListener( 'uncapturederror', event => errors.push( event.error.message ) );
	device.lost.then( info => { if ( info.reason !== 'destroyed' ) errors.push( `device lost: ${ info.message }` ); } );
	const formats = outputFormats( device );
	const timestamps = device.features.has( 'timestamp-query' );
	const timer = bench && timestamps ? createTimer( device ) : null;
	const results = [];

	try {

		for ( const [ width, height ] of sizes ) {

			let started = performance.now();
			const scene = createScene( width, height );
			const sceneMs = performance.now() - started;
			const owned = [];
			const { fixtures: all, pending } = await createFixtures( device, scene, formats, owned, only );
			const fixtures = only ? all.filter( fixture => fixture.name === 'floor' || only.some( name => fixture.name.startsWith( name ) ) ) : all;
			const row = { width, height, scene: { ms: Math.round( sceneMs ), ...scene.stats }, pending, fixtures: {} };
			if ( mutation ) mutate( scene );

			try {

				// Correctness: one run each, in order, then the oracle.
				started = performance.now();
				device.pushErrorScope( 'validation' );
				for ( const fixture of fixtures ) {

					fixture.reset?.();
					run( device, fixture );
					await device.queue.onSubmittedWorkDone();
					const check = fixture.check ? await fixture.check() : null;
					row.fixtures[ fixture.name ] = { pass: fixture.pass, dispatches: fixture.dispatches, check };
					onProgress( `${ width }x${ height } ${ fixture.name }: ${ check ? ( check.ok ? 'ok' : 'FAILED' ) : '-' }` );

				}

				const validation = await device.popErrorScope();
				if ( validation ) throw new Error( `GPU validation error: ${ validation.message }` );
				row.checkMs = Math.round( performance.now() - started );

				if ( timer ) {

					for ( const fixture of fixtures ) run( device, fixture, 4 );
					await device.queue.onSubmittedWorkDone();
					const samples = Object.fromEntries( fixtures.map( fixture => [ fixture.name, [] ] ) );
					for ( let round = 0; round < rounds; round ++ ) {

						const order = [ ...fixtures.slice( round % fixtures.length ), ...fixtures.slice( 0, round % fixtures.length ) ];
						for ( const fixture of order ) samples[ fixture.name ].push( await timer.time( fixture, repeats ) );

					}

					for ( const fixture of fixtures ) Object.assign( row.fixtures[ fixture.name ], { gpuMs: median( samples[ fixture.name ] ), samples: samples[ fixture.name ] } );
					row.ratios = {};
					for ( const [ a, b ] of ratioPairs ) {

						const x = row.fixtures[ a ]?.gpuMs, y = row.fixtures[ b ]?.gpuMs;
						if ( x && y ) row.ratios[ `${ a } ÷ ${ b }` ] = x / y;

					}

					onProgress( `${ width }x${ height } timed` );

				}

			} finally {

				for ( const fixture of all ) fixture.destroy();
				for ( const resource of owned ) resource.destroy();

			}

			results.push( row );

		}

		await device.queue.onSubmittedWorkDone();
		if ( errors.length ) throw new Error( errors.join( '\n' ) );
		const checked = results.flatMap( row => Object.entries( row.fixtures ).filter( ( [ , fixture ] ) => fixture.check ).map( ( [ name, fixture ] ) => ( { name: `${ row.width }x${ row.height } ${ name }`, ok: fixture.check.ok } ) ) );
		const ok = mutation ? checked.every( entry => ! entry.ok ) : checked.every( entry => entry.ok );
		return { date: new Date().toISOString(), adapter, userAgent, formats, timestamps, rounds, repeats, mutation, results, ok, ...( mutation ? { missed: checked.filter( entry => entry.ok ).map( entry => entry.name ) } : {} ) };

	} finally {

		timer?.destroy();
		device.destroy();

	}

}
