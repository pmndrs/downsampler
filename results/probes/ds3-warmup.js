import { Downsampler, requiredLimits } from '/dist/webgpu/index.js';

const query = new URLSearchParams( location.search );
const sleep = ms => new Promise( resolve => setTimeout( resolve, ms ) );
const constant = [ 128, 64, 32, 255 ];
const fixture = { width: 512, height: 288, levels: 10, constant, tolerance: 1 };

function newCounters() {

	return { modules: 0, wgslBytes: 0, computeSync: 0, computeAsync: 0, renderSync: 0, renderAsync: 0 };

}

function subtract( after, before ) {

	return Object.fromEntries( Object.keys( after ).map( key => [ key, after[ key ] - before[ key ] ] ) );

}

function instrument( device, counters, pipelines, currentPhase ) {

	return new Proxy( device, {
		get( target, key ) {

			const method = Reflect.get( target, key, target );
			if ( key === 'createShaderModule' ) return descriptor => {

				counters.modules ++;
				counters.wgslBytes += new TextEncoder().encode( descriptor.code ).length;
				return method.call( target, descriptor );

			};
			if ( /^create(Compute|Render)Pipeline(Async)?$/.test( String( key ) ) ) return descriptor => {

				const asynchronous = String( key ).endsWith( 'Async' );
				const type = String( key ).includes( 'Compute' ) ? 'compute' : 'render';
				counters[ type + ( asynchronous ? 'Async' : 'Sync' ) ] ++;
				const entry = { phase: currentPhase(), method: key, label: descriptor.label, start: performance.now() };
				pipelines.push( entry );
				const result = method.call( target, descriptor );
				if ( asynchronous ) return result.then( pipeline => {

					entry.elapsedMs = performance.now() - entry.start;
					return pipeline;

				} );
				entry.elapsedMs = performance.now() - entry.start;
				return result;

			};
			return typeof method === 'function' ? method.bind( target ) : method;

		},
	} );

}

function observeGaps( currentPhase ) {

	let active = true;
	let lastBeat = performance.now();
	let lastFrame = lastBeat;
	const heartbeats = [];
	const frames = [];
	const interval = setInterval( () => {

		const now = performance.now();
		heartbeats.push( { phase: currentPhase(), gap: now - lastBeat } );
		lastBeat = now;

	}, 4 );
	const frame = now => {

		frames.push( { phase: currentPhase(), gap: now - lastFrame } );
		lastFrame = now;
		if ( active ) requestAnimationFrame( frame );

	};
	requestAnimationFrame( frame );
	function summary( events, phase ) {

		const gaps = events.filter( event => event.phase === phase ).map( event => event.gap );
		return { samples: gaps.length, maxMs: gaps.length ? Math.max( ...gaps ) : null, over50ms: gaps.filter( gap => gap > 50 ).length };

	}
	return {
		forPhase: phase => ( { heartbeat: summary( heartbeats, phase ), raf: summary( frames, phase ) } ),
		stop() {

			active = false;
			clearInterval( interval );

		},
	};

}

async function readMips( device, texture ) {

	const mips = [];
	for ( let level = 1; level < fixture.levels; level ++ ) {

		const width = Math.max( 1, fixture.width >> level );
		const height = Math.max( 1, fixture.height >> level );
		const stride = Math.ceil( width * 4 / 256 ) * 256;
		const buffer = device.createBuffer( { size: stride * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		try {

			const encoder = device.createCommandEncoder();
			encoder.copyTextureToBuffer( { texture, mipLevel: level }, { buffer, bytesPerRow: stride }, [ width, height ] );
			device.queue.submit( [ encoder.finish() ] );
			await buffer.mapAsync( GPUMapMode.READ );
			const bytes = new Uint8Array( buffer.getMappedRange() );
			let maxError = 0;
			let hash = 2166136261;
			for ( let y = 0; y < height; y ++ ) {

				for ( let x = 0; x < width * 4; x ++ ) {

					const value = bytes[ y * stride + x ];
					maxError = Math.max( maxError, Math.abs( value - constant[ x % 4 ] ) );
					hash = Math.imul( hash ^ value, 16777619 ) >>> 0;

				}

			}
			mips.push( { level, width, height, maxError, hash } );
			buffer.unmap();

		} finally {

			buffer.destroy();

		}

	}
	return mips;

}

async function run() {

	const adapter = await navigator.gpu.requestAdapter();
	if ( ! adapter ) throw new Error( 'No WebGPU adapter' );
	const device = await adapter.requestDevice( { requiredLimits: requiredLimits( adapter ) } );
	const errors = [];
	device.addEventListener( 'uncapturederror', event => errors.push( event.error.message ) );
	const counters = newCounters();
	const pipelines = [];
	let phase = 'setup';
	const measured = instrument( device, counters, pipelines, () => phase );
	const output = query.get( 'output' );
	const arm = query.get( 'arm' );
	const srgb = output === 'render';
	const texture = device.createTexture( {
		size: [ fixture.width, fixture.height ], mipLevelCount: fixture.levels,
		format: srgb ? 'rgba8unorm-srgb' : 'rgba8unorm',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | ( srgb ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.STORAGE_BINDING ),
	} );
	const downsampler = new Downsampler( measured, { output } );
	let observer;
	try {

		const input = new Uint8Array( fixture.width * fixture.height * 4 );
		for ( let offset = 0; offset < input.length; offset += 4 ) input.set( constant, offset );
		device.queue.writeTexture( { texture }, input, { bytesPerRow: fixture.width * 4 }, [ fixture.width, fixture.height ] );
		await device.queue.onSubmittedWorkDone();
		observer = observeGaps( () => phase );
		async function measurePhase( name, operation ) {

			phase = name;
			await sleep( 35 );
			const before = { ...counters };
			const start = performance.now();
			const timing = await operation();
			const elapsedMs = performance.now() - start;
			await sleep( 35 );
			return { elapsedMs, ...timing, created: subtract( counters, before ), ...observer.forPhase( name ) };

		}
		async function encode( name ) {

			return measurePhase( name, async () => {

				const start = performance.now();
				const encoder = device.createCommandEncoder();
				downsampler.encode( encoder, texture );
				const command = encoder.finish();
				const recorded = performance.now();
				device.queue.submit( [ command ] );
				const submitted = performance.now();
				await device.queue.onSubmittedWorkDone();
				const done = performance.now();
				return { recordMs: recorded - start, submitMs: submitted - recorded, submitToDoneMs: done - submitted, recordToDoneMs: done - start };

			} );

		}
		const warmup = arm === 'warmup' ? await measurePhase( 'warmup', async () => {

			if ( typeof downsampler.warmup !== 'function' ) throw new Error( 'Missing warmup API' );
			await downsampler.warmup( texture );
			return {};

		} ) : null;
		const first = await encode( 'first' );
		const second = await encode( 'second' );
		phase = 'validation';
		const pixels = await readMips( device, texture );
		await sleep( 20 );
		if ( errors.length ) throw new Error( errors.join( '\n' ) );
		if ( pixels.some( mip => mip.maxError > fixture.tolerance ) ) throw new Error( 'Constant-input pixel check failed' );
		const info = adapter.info;
		return {
			adapter: Object.fromEntries( [ 'vendor', 'architecture', 'device', 'description' ].map( key => [ key, info[ key ] ] ) ),
			fixture: { ...fixture, format: texture.format }, output, arm, warmup, first, second, counters, pipelines, pixels, errors,
		};

	} finally {

		observer?.stop();
		downsampler.release( texture );
		texture.destroy();
		device.destroy();

	}

}

run().then( result => { window.__probeResult = result; } ).catch( error => { window.__probeError = error.stack; } );
