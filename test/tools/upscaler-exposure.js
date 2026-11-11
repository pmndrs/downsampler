// pmndrs/upscaler's exposure pass ('upscaler/exposure'): one invocation takes
// 32×32 bilinear taps of the scene colour, averages log2 luma and eases the
// exposure. Dispatched as UpscalerCore does: one 8×8 workgroup.
import { LUMINANCE_PYRAMID_SHADER } from './upscaler-wgsl.js';
import { compile, createTexture, readTexture, uniformBuffer } from './gpu.js';
import { f32, halfUlp } from './half.js';

export const FLAG_RESET = 1, FLAG_AUTO_EXPOSURE = 64, FLAG_SHADING_CHANGE = 128;

/** The upscaler's 96-byte FsrConstants (internal/ConstantsBuffer.ts layout). Display size is 2× render. */
export function upscalerConstants( scene, { flags = FLAG_AUTO_EXPOSURE | FLAG_SHADING_CHANGE, exposure = 1, exposureTaps = 32 } = {} ) {

	const { width, height } = scene;
	const bytes = new ArrayBuffer( 96 ), f = new Float32Array( bytes ), u = new Uint32Array( bytes );
	f.set( [ width, height, width * 2, height * 2, 1 / width, 1 / height, 1 / ( width * 2 ), 1 / ( height * 2 ), ...scene.jitter, ...scene.previousJitter, 0.5, - 0.5, 0.1, 1000 ] );
	f[ 16 ] = 1; f[ 17 ] = 32; f[ 18 ] = exposure; f[ 19 ] = scene.deltaTime;
	u[ 20 ] = flags; u[ 21 ] = 1; u[ 22 ] = 0; u[ 23 ] = exposureTaps;
	return bytes;

}

// The previous frame's exposure texel: (exposure, average luma, host pre-exposure, 0).
export const previousExposure = [ 1, 0.5, 1, 0 ];

/**
 * The pass's resources in binding order (0-6 of luminancePyramid.ts) and a
 * check against an oracle. Shared with the generator versions.
 */
export function createUpscalerExposureResources( device, scene, colour ) {

	const constants = uniformBuffer( device, 'FsrConstants', upscalerConstants( scene ) );
	const previous = createTexture( device, { label: 'exposure prev', width: 1, height: 1, format: 'rgba16float', data: previousExposure } );
	const output = createTexture( device, { label: 'exposure', width: 1, height: 1, format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING } );
	const dummy = device.createTexture( { label: 'reactive dummy', size: [ 1, 1 ], format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING } );
	const sampler = device.createSampler( { magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' } );
	const views = [ { buffer: constants }, colour.createView(), sampler, previous.createView(), output.createView(), dummy.createView(), dummy.createView() ];
	return {
		/** Bind group entries 0-6, at `offset` + i. */
		groupEntries: ( offset = 0 ) => views.map( ( resource, binding ) => ( { binding: binding + offset, resource } ) ),
		/** Compares the exposure texel with `oracle` (the 1024-tap one by default). */
		async check( oracle = exposureOracle ) { return checkExposure( scene, Array.from( ( await readTexture( device, output ) ).data ), oracle ); },
		destroy() { for ( const resource of [ constants, previous, output, dummy ] ) resource.destroy(); }
	};

}

export async function createUpscalerExposure( device, scene, colour ) {

	const module = await compile( device, 'upscaler exposure', LUMINANCE_PYRAMID_SHADER );
	const pipeline = await device.createComputePipelineAsync( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
	const shared = createUpscalerExposureResources( device, scene, colour );
	const group = device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: shared.groupEntries() } );

	return {
		name: 'upscaler/exposure',
		pass: 'exposure',
		dispatches: 1,
		// ComputePass.dispatch( pass, group, 8, 8 ): ⌈8/8⌉ × ⌈8/8⌉ workgroups.
		encodePass( pass ) { pass.setPipeline( pipeline ); pass.setBindGroup( 0, group ); pass.dispatchWorkgroups( 1, 1 ); },
		check: () => shared.check(),
		destroy: () => shared.destroy()
	};

}

/**
 * Oracle with an error bound. GPU bilinear weights are fixed point (8
 * fractional bits at least), so each tap's channel can be off by 2·2^-8 of
 * its four texels' range; that propagates to log2 luma as Δluma / (luma·ln 2)
 * and is averaged over the 1024 taps. Outputs are rgba16float: plus one
 * float16 ulp. Plus 1e-5 relative for float32 arithmetic (log2/exp2 ulps).
 */
export function exposureOracle( scene ) {

	const { width: w, height: h, colour, deltaTime } = scene;
	const taps = 32;
	let logSum = 0, logBound = 0;
	const channel = ( x, y, c ) => colour[ ( Math.min( Math.max( y, 0 ), h - 1 ) * w + Math.min( Math.max( x, 0 ), w - 1 ) ) * 4 + c ];

	for ( let ty = 0; ty < taps; ty ++ ) for ( let tx = 0; tx < taps; tx ++ ) {

		const u = f32( f32( tx + 0.5 ) / taps ), v = f32( f32( ty + 0.5 ) / taps );
		const px = u * w - 0.5, py = v * h - 0.5, x0 = Math.floor( px ), y0 = Math.floor( py ), fx = px - x0, fy = py - y0;
		let luma = 0, lumaBound = 0;
		for ( const [ c, weight ] of [ [ 0, 0.2126 ], [ 1, 0.7152 ], [ 2, 0.0722 ] ] ) {

			const a = channel( x0, y0, c ), b = channel( x0 + 1, y0, c ), d = channel( x0, y0 + 1, c ), e = channel( x0 + 1, y0 + 1, c );
			luma += weight * ( ( a * ( 1 - fx ) + b * fx ) * ( 1 - fy ) + ( d * ( 1 - fx ) + e * fx ) * fy );
			lumaBound += weight * 2 * 2 ** - 8 * ( Math.max( a, b, d, e ) - Math.min( a, b, d, e ) );

		}

		const clamped = Math.max( luma, 1e-4 );
		logSum += Math.log2( clamped );
		logBound += lumaBound / ( clamped * Math.LN2 );

	}

	const avgLum = 2 ** ( logSum / ( taps * taps ) ); // host pre-exposure dummy reads 0 → 1
	return exposureFrom( avgLum, avgLum * Math.LN2 * logBound / ( taps * taps ), deltaTime );

}

/**
 * The same epilogue over every texel equally weighted (the generator's
 * `ScalarReduction`). Bound on the mean of log2 luma, per texel: the luma dot
 * product (3 f32 roundings, 3·2^-24 relative → /ln 2 in log2) and WGSL log2
 * (2^-21 absolute inside [0.5, 2], else 3 ulp); then the scalar's summation
 * bound (SPEC §4 Scalar reductions: γ_D·Σ|x|, D ≤ 27 to 4K). The result's
 * bound adds exp2 (3 + 2|x| ulp), one f16 ulp and the same 1e-5 relative as
 * the 1024-tap oracle.
 */
export function exposureOracleEveryTexel( scene ) {

	const { width: w, height: h, colour, deltaTime } = scene;
	const count = w * h, u = 2 ** - 24;
	let logSum = 0, absSum = 0, logBound = 0;
	for ( let i = 0; i < count; i ++ ) {

		const luma = f32( f32( f32( colour[ i * 4 ] * f32( 0.2126 ) ) + f32( colour[ i * 4 + 1 ] * f32( 0.7152 ) ) ) + f32( colour[ i * 4 + 2 ] * f32( 0.0722 ) ) );
		const clamped = Math.max( luma, f32( 1e-4 ) ), log = Math.log2( clamped );
		logSum += log; absSum += Math.abs( log );
		logBound += ( clamped > luma ? 0 : 3 * u / Math.LN2 ) + ( clamped >= 0.5 && clamped <= 2 ? 2 ** - 21 : 3 * 2 * u * Math.abs( log ) );

	}

	const mean = logSum / count, D = 27, gamma = D * u / ( 1 - D * u );
	const meanBound = logBound / count + gamma * absSum / count;
	const avgLum = 2 ** mean;
	return exposureFrom( avgLum, avgLum * ( Math.LN2 * meanBound + ( 3 + 2 * Math.abs( mean ) ) * 2 * u ), deltaTime );

}

/** luminancePyramid.ts after the metering: target, dead band, easing. Bounds carried from avgLum. */
function exposureFrom( avgLum, avgBound, deltaTime ) {

	const target = Math.min( Math.max( 0.18 / Math.max( avgLum, 1e-4 ), 0.02 ), 8 );
	const prev = previousExposure[ 0 ];
	let exposure = prev, rate = 0;
	if ( Math.abs( Math.log2( target / prev ) ) > 0.0625 ) {

		rate = Math.min( Math.max( 1 - 2 ** ( - deltaTime * 2.5 ), 0 ), 1 );
		exposure = prev + ( target - prev ) * rate;

	}

	const targetBound = target * avgBound / avgLum;
	return {
		value: [ exposure, avgLum, 1, 0 ],
		bound: [ rate * targetBound + halfUlp( exposure ) + 1e-5 * exposure, avgBound + halfUlp( avgLum ) + 1e-5 * avgLum, 0, 0 ],
		target, deadband: Math.abs( Math.log2( target / prev ) )
	};

}

export function checkExposure( scene, output, oracle = exposureOracle ) {

	const expected = oracle( scene );
	const checks = [ [ 0, 'exposure' ], [ 1, 'avgLum' ] ].map( ( [ i, name ] ) => {

		const error = Math.abs( output[ i ] - expected.value[ i ] );
		return { output: `exposure.${ name }`, actual: output[ i ], expected: expected.value[ i ], maxError: error, unit: 'absolute', tolerance: expected.bound[ i ], ok: error <= expected.bound[ i ] };

	} );
	checks.push( { output: 'exposure.host', actual: output[ 2 ], expected: 1, maxError: Math.abs( output[ 2 ] - 1 ), unit: 'absolute', tolerance: 0, ok: output[ 2 ] === 1 } );
	// The dead-band test is a discrete decision; a target within 1 % of it could flip legitimately.
	const margin = Math.abs( expected.deadband - 0.0625 ) > 0.01;
	return { checks, ok: checks.every( check => check.ok ) && margin, notes: { target: expected.target, deadbandStops: expected.deadband } };

}
