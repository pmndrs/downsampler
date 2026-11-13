// `compare: 'storage-usage'` (SPEC §8, M3): does adding STORAGE_BINDING to a texture make
// sampling it slower (Kangz, gpuweb#5253)? It matters for SPEC §10, where three.js would add
// the usage to every texture whose mips we generate.
//
// Two textures per case, identical except for STORAGE_BINDING (both TEXTURE_BINDING,
// RENDER_ATTACHMENT, COPY_SRC, COPY_DST; full mip chain). Their content is written the same
// way, then each is read by the same pass into the same rgba8unorm target:
// - shaders: 'fullscreen' (one tap per pixel at level 0, as a post pass), 'material' (nine
//   trilinear taps with implicit derivatives at ~1.6× minification, so levels 0–1, as a
//   material does), 'write' (the content pass itself, rendering into level 0: the cost of the
//   usage on a render target).
// - content: 'rendered' (drawn by a render pass: flat bands and gradients, the compressible
//   kind; lossless framebuffer compression, which a storage usage may disable, only helps
//   content like this) and 'noise' (uploaded random texels: incompressible, the control).
//   'write' runs only with 'rendered' (it rewrites the same content).
// Levels 1+ are filled by render passes (2×2 textureLoad averages) for both textures.
//
// Timing: `repeats` passes into the target, the first and last carrying timestampWrites
// (never empty passes), rotated order per round, medians, and the storage ÷ plain ratio paired
// per round. Without timestamp-query (or with `throughput`), wall-clock throughput per pass too.
import { createTexture, makeData, median, upload } from '../browser/gpu.js';
import { measureUnits } from './throughput.js';

export const storageUsageSizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1707, 960 ], [ 1920, 1080 ], [ 2560, 1440 ], [ 1024, 1024 ], [ 4096, 4096 ] ];
export const storageUsageFormats = [ 'rgba8unorm', 'rgba16float', 'r32float' ];
export const storageUsageShaders = [ 'fullscreen', 'material', 'write' ];

const vertex = `
struct Varyings { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) index: u32) -> Varyings {
	let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3))[index];
	return Varyings(vec4f(p, 0, 1), p * vec2f(0.5, -0.5) + 0.5);
}
`;

// Flat bands (top half) and smooth gradients with rings (bottom half).
const fillCode = vertex + `
@fragment fn fs(in: Varyings) -> @location(0) vec4f {
	let uv = in.uv;
	if (uv.y < 0.5) { return vec4f(floor(uv * 8.0) / 8.0, 0.25, 1.0); }
	let r = length(uv - vec2f(0.5, 0.75));
	return vec4f(uv.x, uv.y, 0.5 + 0.5 * sin(r * 40.0), 1.0);
}
`;

// One level from the level above: a 2×2 textureLoad average (any float-sampleable format).
const mipCode = vertex + `
@group(0) @binding(0) var above: texture_2d<f32>;
@fragment fn fs(in: Varyings) -> @location(0) vec4f {
	let last = vec2i(textureDimensions(above)) - 1;
	let p = vec2i(in.position.xy) * 2;
	return (textureLoad(above, min(p, last), 0) + textureLoad(above, min(p + vec2i(1, 0), last), 0)
		+ textureLoad(above, min(p + vec2i(0, 1), last), 0) + textureLoad(above, min(p + vec2i(1, 1), last), 0)) * 0.25;
}
`;

const sampleCode = {
	fullscreen: vertex + `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;
@fragment fn fs(in: Varyings) -> @location(0) vec4f {
	let v = textureSampleLevel(image, imageSampler, in.uv, 0.0);
	return vec4f(v.xyz * 0.5 + 0.25, 1.0);
}
`,
	material: vertex + `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;
@fragment fn fs(in: Varyings) -> @location(0) vec4f {
	// A rotated, 1.6× minified mapping (trilinear between levels 0 and 1), nine taps.
	let uv = mat2x2f(0.96, -0.28, 0.28, 0.96) * in.uv * 1.6 + vec2f(0.13, 0.07);
	let texel = 1.5 / vec2f(textureDimensions(image));
	var sum = vec4f(0.0);
	for (var y = -1; y <= 1; y++) {
		for (var x = -1; x <= 1; x++) { sum += textureSample(image, imageSampler, uv + vec2f(f32(x), f32(y)) * texel); }
	}
	return vec4f(sum.xyz / 9.0, 1.0);
}
`
};

const filterable = format => format !== 'r32float';

export async function storageUsage( device, { sizes = storageUsageSizes, formats = storageUsageFormats, rounds = 11, repeats = 16, shaders = storageUsageShaders, throughput = null, onProgress = () => {} } = {} ) {

	const unknown = shaders.filter( name => ! storageUsageShaders.includes( name ) );
	if ( unknown.length ) throw new Error( `storage-usage: unknown shaders ${ unknown.join( ', ' ) }; expected ${ storageUsageShaders.join( ', ' ) }` );
	const timestamps = device.features.has( 'timestamp-query' );
	const modules = { fill: device.createShaderModule( { label: 'storage-usage fill', code: fillCode } ), mip: device.createShaderModule( { label: 'storage-usage mip', code: mipCode } ) };
	for ( const [ name, code ] of Object.entries( sampleCode ) ) modules[ name ] = device.createShaderModule( { label: `storage-usage ${ name }`, code } );
	const pipelines = new Map();
	const pipeline = ( key, make ) => pipelines.get( key ) ?? pipelines.set( key, make() ).get( key );
	const renderPipeline = ( module, format, layout = 'auto' ) => device.createRenderPipeline( { layout, vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [ { format } ] } } );
	const samplers = {
		filtering: device.createSampler( { minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' } ),
		nearest: device.createSampler( { addressModeU: 'repeat', addressModeV: 'repeat' } )
	};
	const sampleLayout = format => pipeline( `layout:${ format }`, () => device.createBindGroupLayout( { entries: [
		{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: filterable( format ) ? 'float' : 'unfilterable-float' } },
		{ binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: filterable( format ) ? 'filtering' : 'non-filtering' } }
	] } ) );
	const mipLayout = pipeline( 'layout:mip', () => device.createBindGroupLayout( { entries: [ { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } } ] } ) );
	const pass = ( encoder, view, pipe, bindGroup, timestampWrites ) => {

		const p = encoder.beginRenderPass( { colorAttachments: [ { view, loadOp: 'clear', storeOp: 'store' } ], timestampWrites } );
		p.setPipeline( pipe );
		if ( bindGroup ) p.setBindGroup( 0, bindGroup );
		p.draw( 3 );
		p.end();

	};

	// Level 0 drawn (or uploaded), levels 1+ rendered from the level above, both textures alike.
	const fill = ( texture, content, data ) => {

		const encoder = device.createCommandEncoder();
		const level = mip => texture.createView( { baseMipLevel: mip, mipLevelCount: 1 } );
		if ( content === 'rendered' ) pass( encoder, level( 0 ), pipeline( `fill:${ texture.format }`, () => renderPipeline( modules.fill, texture.format ) ) );
		else device.queue.writeTexture( { texture }, data, { bytesPerRow: texture.width * { rgba8unorm: 4, rgba16float: 8, r32float: 4 }[ texture.format ], rowsPerImage: texture.height }, [ texture.width, texture.height ] );
		const mip = pipeline( `mip:${ texture.format }`, () => renderPipeline( modules.mip, texture.format, device.createPipelineLayout( { bindGroupLayouts: [ mipLayout ] } ) ) );
		for ( let m = 1; m < texture.mipLevelCount; m ++ ) pass( encoder, level( m ), mip, device.createBindGroup( { layout: mipLayout, entries: [ { binding: 0, resource: level( m - 1 ) } ] } ) );
		device.queue.submit( [ encoder.finish() ] );

	};

	/** `repeats` passes of `record( encoder, timestampWrites )`; GPU ms per pass. */
	const time = async record => {

		const querySet = device.createQuerySet( { type: 'timestamp', count: 2 } );
		const resolve = device.createBuffer( { size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC } );
		const readback = device.createBuffer( { size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST } );
		const encoder = device.createCommandEncoder();
		for ( let i = 0; i < repeats; i ++ ) record( encoder, i === 0 || i === repeats - 1 ? { querySet, ...( i === 0 ? { beginningOfPassWriteIndex: 0 } : {} ), ...( i === repeats - 1 ? { endOfPassWriteIndex: 1 } : {} ) } : undefined );
		encoder.resolveQuerySet( querySet, 0, 2, resolve, 0 );
		encoder.copyBufferToBuffer( resolve, 0, readback, 0, 16 );
		device.queue.submit( [ encoder.finish() ] );
		await readback.mapAsync( GPUMapMode.READ );
		const [ start, end ] = new BigUint64Array( readback.getMappedRange() );
		readback.unmap(); readback.destroy(); resolve.destroy(); querySet.destroy();
		if ( start === 0n || end <= start ) throw new Error( `invalid GPU timestamps: ${ start }, ${ end }` );
		return Number( end - start ) / 1e6 / repeats;

	};

	const results = [];
	device.pushErrorScope( 'validation' );

	try {

		for ( const format of formats ) for ( const [ width, height ] of sizes ) {

			const textures = {
				plain: createTexture( device, { width, height, format, storage: false, render: true } ),
				storage: createTexture( device, { width, height, format, storage: true, render: true } )
			};
			const target = device.createTexture( { size: [ width, height ], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT } );
			const targetView = target.createView();

			try {

				for ( const content of [ 'rendered', 'noise' ] ) {

					const data = content === 'noise' ? makeData( width, height, 1, format, 'random', 1337 ) : null;
					for ( const texture of Object.values( textures ) ) fill( texture, content, data );
					for ( const shader of shaders ) {

						if ( shader === 'write' && content !== 'rendered' ) continue;
						const record = {};
						for ( const [ name, texture ] of Object.entries( textures ) ) {

							if ( shader === 'write' ) {

								const view = texture.createView( { baseMipLevel: 0, mipLevelCount: 1 } ), pipe = pipeline( `fill:${ format }`, () => renderPipeline( modules.fill, format ) );
								record[ name ] = ( encoder, timestampWrites ) => pass( encoder, view, pipe, null, timestampWrites );

							} else {

								const layout = sampleLayout( format );
								const pipe = pipeline( `${ shader }:${ format }`, () => renderPipeline( modules[ shader ], 'rgba8unorm', device.createPipelineLayout( { bindGroupLayouts: [ layout ] } ) ) );
								const bindGroup = device.createBindGroup( { layout, entries: [ { binding: 0, resource: texture.createView() }, { binding: 1, resource: filterable( format ) ? samplers.filtering : samplers.nearest } ] } );
								record[ name ] = ( encoder, timestampWrites ) => pass( encoder, targetView, pipe, bindGroup, timestampWrites );

							}

						}

						const names = Object.keys( record );
						// Warm up both (pipelines, residency, clocks).
						for ( let i = 0; i < 4; i ++ ) for ( const name of names ) {

							const encoder = device.createCommandEncoder();
							for ( let r = 0; r < Math.max( repeats, 16 ); r ++ ) record[ name ]( encoder );
							device.queue.submit( [ encoder.finish() ] );

						}

						await device.queue.onSubmittedWorkDone();
						const result = { width, height, format, content, shader };
						if ( timestamps ) {

							const samples = { plain: [], storage: [] };
							for ( let round = 0; round < rounds; round ++ ) for ( const name of round % 2 ? [ ...names ].reverse() : names ) samples[ name ].push( await time( record[ name ] ) );
							result.gpuMs = { plain: median( samples.plain ), storage: median( samples.storage ) };
							result.ratio = median( samples.storage.map( ( value, round ) => value / samples.plain[ round ] ) );
							result.samples = samples;

						}

						if ( throughput ) {

							const measured = await measureUnits( device, { plain: encoder => record.plain( encoder ), storage: encoder => record.storage( encoder ) }, { rounds: Math.min( rounds, 5 ), reference: () => 'plain', ...throughput } );
							result.throughput = measured.units;
							result.throughputRatio = measured.units.storage.ratio;

						}

						results.push( result );
						onProgress( `${ width }x${ height } ${ format } ${ content } ${ shader }: storage ÷ plain ${ ( result.ratio ?? result.throughputRatio ).toFixed( 3 ) }` );

					}

				}

			} finally {

				textures.plain.destroy(); textures.storage.destroy(); target.destroy();

			}

		}

	} finally {

		const error = await device.popErrorScope();
		if ( error ) throw new Error( `GPU validation error: ${ error.message }` );

	}

	// Summary: the median storage ÷ plain ratio over sizes, per format, content and shader.
	const summary = {};
	for ( const row of results ) {

		const key = `${ row.format } ${ row.content } ${ row.shader }`;
		( summary[ key ] ??= [] ).push( row.ratio ?? row.throughputRatio );

	}

	return { mode: 'storage-usage', timestamps, rounds, repeats, throughput, results, summary: Object.fromEntries( Object.entries( summary ).map( ( [ key, ratios ] ) => [ key, { median: median( ratios ), min: Math.min( ...ratios ), max: Math.max( ...ratios ) } ] ) ) };

}
