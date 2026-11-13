// Kernel-variant probes (SPEC §8, M3): the library's own mip chain with the 'exact' batches'
// main() swapped for a variant, timed against the unchanged chain. Everything else is the
// library's: the plan (Downsampler.prepare with its defaults, so the tail kernel and the device's
// exactTile), bind groups, helper WGSL (`first`, `quantize`, f16 rounding) and tail batches.
// Variants only replace batches of two or more levels (a 1-level batch has nothing to reduce).
//
// `probe: 'subgroups'`: what removing barriers and workgroup memory from levels 2+ could buy.
// - 'lds': the library's kernel (two barriers per level, the whole tile in workgroup memory).
// - 'slot': one barrier per level: each level's texel goes to the slot of its first input
//   (2x, 2y) at the previous level's stride, so no reader of a level is overwritten by it
//   (SPEC §13 "cheaper first"). Same results, no subgroups.
// - 'shuffle' (needs 'subgroups'): the cheapest real subgroup kernel on the existing layout.
//   The first reduction (tile → half) comes from subgroupShuffleXor with masks 1 (x) and the
//   workgroup side (y), assuming lane = local_invocation_index mod subgroup size and subgroups
//   of at least two rows (WGSL guarantees neither: the check below says whether it held); then
//   'slot' for the rest. Barriers per 6-level batch: 4 against 'lds''s 10.
// - 'nosync' (timing only, wrong values): levels 2+ from each invocation's own registers, no
//   barriers and no workgroup memory at all, stores unchanged. No reduction scheme can be
//   cheaper, so (lds − nosync) ÷ lds bounds what any subgroup kernel can gain. SPEC M3 rule:
//   build the subgroup kernel only if the bound is at least 5 %.
//
// `probe: 'f16'` (needs 'shader-f16'): the averaging in f16 instead of f32.
// - 'f32': the library's kernel.
// - 'f16': workgroup memory as vec4h (half the bytes) and levels 2+ averaged in f16,
//   `(a + b + c + d) * 0.25` as the f32 kernel orders it. The first output is still sampled and
//   rounded in f32, then converted (exact: integer units for unorm8, halves for float16).
//   unorm8 sums (≤ 1020) and quarters are exact in f16, so rgba8 should match bit for bit;
//   float16 sums round at each add and overflow above 65504.
// - 'f16safe': workgroup memory holds each texel × 0.25, so a level is the plain sum of four
//   quarters: no overflow below 65504, the same rounding otherwise (scaling by a power of two is
//   exact above the subnormals). Scaling the inputs in the sum instead (`a * 0.25 + b * 0.25 …`)
//   overflowed just the same on the M5 Pro: Metal's fast math refactors it to `(a + b …) * 0.25`.
// Nothing here relies on `(x + c) − c`-style rounding, which fast math also folds away.
//
// Check: every variant except 'nosync' runs on the timed input and on a second pattern
// ('tiles' for unorm8; 'large', halves in [16384, 65504], for float16) and is compared level by
// level with the library's chain on the same input: values that differ, the worst difference in
// format steps (unorm8 steps, f16 ulps) and non-finite values where the library's are finite.
// The library's f32 chain is what the GPU suite checks against the CPU oracle, so a difference
// from it is the variant's own cost. Timing: timeGpu (render markers, marker cost subtracted),
// rotated order, medians, ratios to the library's kernel paired per round.
import { formats, toHalf } from '/dist/index.js';
import { Downsampler, createKernel, kernelLayoutEntries } from '/dist/webgpu/index.js';
import { createTexture, makeData, median, readMips, upload } from '../browser/gpu.js';
import { measureUnits } from './throughput.js';
import { timeGpu } from './timing.js';

export const variantSizes = [ [ 256, 256 ], [ 512, 512 ], [ 1024, 1024 ], [ 2048, 2048 ], [ 4096, 4096 ], [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1920, 1080 ], [ 2560, 1440 ] ];
export const variantFormats = [ 'rgba8unorm', 'rgba16float' ];
export const probes = {
	subgroups: { baseline: 'lds', variants: [ 'lds', 'slot', 'shuffle', 'nosync' ], feature: 'subgroups', needs: { shuffle: 'subgroups' } },
	f16: { baseline: 'f32', variants: [ 'f32', 'f16', 'f16safe' ], feature: 'shader-f16', needs: { f16: 'shader-f16', f16safe: 'shader-f16' } }
};
const timingOnly = new Set( [ 'nosync' ] );

/** Integer-unit rounding (ties to even) in f16, as the kernel's `quantize` does in f32. */
function quantize16( quantization ) {

	if ( quantization === 'float16' ) return 'fn quantize16(v: vec4h) -> vec4h { return v; }\n';
	if ( quantization !== 'unorm8' ) throw new Error( `f16 probe: ${ quantization } formats are not covered (rgba8unorm and rgba16float are)` );
	return `fn quantize16(value: vec4h) -> vec4h {
	let scaled = clamp(value, vec4h(0.0h), vec4h(255.0h));
	let lower = floor(scaled);
	let fraction = scaled - lower;
	let nearest = select(lower, lower + vec4h(1.0h), fraction > vec4h(0.5h));
	return select(nearest, lower + vec4h(vec4i(lower) & vec4i(1)), fraction == vec4h(0.5h));
}
`;

}

/**
 * WGSL for `variant` of an 'exact' batch (`options`: format, count, tile). The library's
 * helpers (everything before its entry point) are kept; main() is the variant's.
 */
export function variantCode( variant, options ) {

	const real = createKernel( options );
	if ( variant === 'lds' || variant === 'f32' ) return real;
	const { tile, count, format } = options;
	const side = Math.min( tile, 16 ), samples = tile / side;
	const half = variant === 'f16' || variant === 'f16safe';
	let prefix = real.slice( 0, real.indexOf( '\n@compute' ) + 1 );
	if ( half ) prefix = 'enable f16;\n' + prefix.replace( /var<workgroup> tile: array<vec4f, (\d+)>;/, 'var<workgroup> tile: array<vec4h, $1>;' ) + quantize16( formats[ format ].quantization );
	if ( variant === 'shuffle' ) prefix = 'enable subgroups;\n' + prefix;
	const dims = i => i === 0 ? 'textureDimensions(mip0)' : `max(size0 >> vec2u(${ i }u), vec2u(1u))`;
	const offset = s => `vec2u(${ ( s % samples ) * side }u, ${ Math.floor( s / samples ) * side }u)`;
	const n = samples * samples;
	const writesTile = variant !== 'nosync' && variant !== 'shuffle';

	let code = prefix + `
@compute @workgroup_size(${ side }, ${ side })
fn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {
	let size0 = ${ dims( 0 ) };
	var value: vec4f;
	var value16: ${ half ? 'vec4h' : 'vec4f' };
	var v: array<vec4f, ${ n }>;
`;
	for ( let s = 0; s < n; s ++ ) code += `
	{
		let p = local.xy + ${ offset( s ) };
		let pos0 = group.xy * ${ tile }u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		v[${ s }] = value;
		${ writesTile ? `tile[p.y * ${ tile }u + p.x] = ${ variant === 'f16safe' ? 'vec4h(value) * 0.25h' : half ? 'vec4h(value)' : 'value' };` : '' }
		if (all(pos0 < size0)) { textureStore(mip0, vec2i(pos0), value / SCALE); }
	}
`;

	// Level i's texel (x, y) at the slot of its first input at level i − 1's stride (2^(i−1)).
	const slotLevel = i => {

		const extent = tile >> i, shift = i - 1;
		const at = ( x, y ) => `tile[((${ y }) << ${ shift }u) * ${ tile }u + ((${ x }) << ${ shift }u)]`;
		return `
	workgroupBarrier();
	{
		let last = ${ dims( i - 1 ) } - group.xy * ${ extent * 2 }u - vec2u(1);
		if (local.x < ${ extent }u && local.y < ${ extent }u && all(local.xy * 2u <= last)) {
			let lo = local.xy * 2u;
			let hi = min(lo + vec2u(1), last);
			value = quantize((${ at( 'lo.x', 'lo.y' ) } + ${ at( 'hi.x', 'lo.y' ) } + ${ at( 'lo.x', 'hi.y' ) } + ${ at( 'hi.x', 'hi.y' ) }) * 0.25);
			${ at( 'lo.x', 'lo.y' ) } = value;
			textureStore(mip${ i }, vec2i(group.xy * ${ extent }u + local.xy), value / SCALE);
		}
	}
`;

	};

	for ( let i = 1; i < count; i ++ ) {

		const extent = tile >> i;
		const store = ( v ) => `let pos = group.xy * ${ extent }u + local.xy;
		if (all(pos < ${ dims( i ) })) { textureStore(mip${ i }, vec2i(pos), ${ v } / SCALE); }`;
		if ( variant === 'slot' || ( variant === 'shuffle' && i > 1 ) ) {

			code += slotLevel( i );

		} else if ( variant === 'shuffle' ) {

			// Level 1 from the first output in registers: x neighbour lane ^ 1, y neighbour lane ^ side.
			code += '\n	{\n		let even = ((local.x | local.y) & 1u) == 0u;\n';
			for ( let s = 0; s < n; s ++ ) code += `		{
			let a = v[${ s }];
			let b = subgroupShuffleXor(a, 1u);
			let c = subgroupShuffleXor(a, ${ side }u);
			let d = subgroupShuffleXor(b, ${ side }u);
			let r = quantize((a + b + c + d) * 0.25);
			let p = local.xy + ${ offset( s ) };
			if (even) {
				tile[p.y * ${ tile }u + p.x] = r;
				let pos = group.xy * ${ extent }u + p / 2u;
				if (all(pos < ${ dims( 1 ) })) { textureStore(mip1, vec2i(pos), r / SCALE); }
			}
		}
`;
			code += '	}\n';

		} else if ( variant === 'nosync' ) {

			const mix = i === 1 && n === 4 ? '(v[0] + v[1] + v[2] + v[3])' : i === 1 ? '(v[0] + v[0].yzwx + v[0].zwxy + v[0].wxyz)' : '(value + value.yzwx + value.zwxy + value.wxyz)';
			code += `
	value = quantize(${ mix } * 0.25);
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		${ store( 'value' ) }
	}
`;

		} else {

			// f16 / f16safe: the library's two-barrier structure, in f16.
			const at = ( x, y ) => `tile[${ y }.y * ${ tile }u + ${ x }.x]`;
			// f16safe: the tile holds quarters, so the sum is the average.
			const sum = `${ at( 'lo', 'lo' ) } + ${ at( 'hi', 'lo' ) } + ${ at( 'lo', 'hi' ) } + ${ at( 'hi', 'hi' ) }`;
			code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let last = ${ dims( i - 1 ) } - group.xy * ${ extent * 2 }u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value16 = quantize16(${ variant === 'f16safe' ? sum : `(${ sum }) * 0.25h` });
	}
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		tile[local.y * ${ tile }u + local.x] = value16${ variant === 'f16safe' ? ' * 0.25h' : '' };
		${ store( 'vec4f(value16)' ) }
	}
`;

		}

	}

	return code + '}\n';

}

/**
 * The library's chain for `texture` with `variant` swapped in: encode( encoder ) records one
 * compute pass per batch, as Downsampler.encode does. Pipelines are cached per WGSL.
 */
function variantChain( device, downsampler, texture, variant, cache ) {

	const batches = downsampler.prepare( texture );
	const steps = batches.map( batch => {

		if ( batch.copies?.length ) throw new Error( 'kernel variants: storage output only' );
		if ( batch.kernel !== 'exact' || batch.count < 2 || variant === 'lds' || variant === 'f32' ) return { pipeline: batch.pipeline, batch };
		const options = { format: texture.format, dimension: '2d', count: batch.count, tile: batch.tile, f16Rounding: downsampler.f16Rounding };
		const code = variantCode( variant, options );
		let pipeline = cache.get( code );
		if ( ! pipeline ) {

			// An equivalent layout: the library's bind groups are compatible with it.
			const layout = device.createPipelineLayout( { bindGroupLayouts: [ device.createBindGroupLayout( { entries: kernelLayoutEntries( options ) } ) ] } );
			pipeline = device.createComputePipeline( { label: `variant ${ variant } ${ texture.format } ${ batch.count }/${ batch.tile }`, layout, compute: { module: device.createShaderModule( { label: `variant ${ variant }`, code } ), entryPoint: 'main' } } );
			cache.set( code, pipeline );

		}

		return { pipeline, batch };

	} );
	return {
		batches: batches.map( batch => `${ batch.kernel } ${ batch.count }×${ batch.tile }` ),
		encode( encoder ) {

			for ( const { pipeline, batch } of steps ) {

				const pass = encoder.beginComputePass();
				pass.setPipeline( pipeline );
				pass.setBindGroup( 0, batch.bindGroup );
				pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, texture.depthOrArrayLayers );
				pass.end();

			}

		}
	};

}

/** Halves (as decoded floats) to an ordered integer, so ulp distance is a subtraction. */
const halfOrder = value => {

	const bits = toHalf( value );
	return bits & 0x8000 ? - ( bits & 0x7fff ) : bits;

};

/** Level-by-level difference of a variant's chain from the library's, in format steps. */
function difference( actual, expected, quantization ) {

	const out = { values: 0, differing: 0, worstSteps: 0, nonFinite: 0, worstLevel: 0 };
	actual.forEach( ( level, l ) => level.forEach( ( a, i ) => {

		const e = expected[ l ][ i ];
		out.values ++;
		if ( Object.is( a, e ) ) return;
		out.differing ++;
		if ( Number.isFinite( e ) && ! Number.isFinite( a ) ) { out.nonFinite ++; return; }
		const steps = quantization === 'unorm8' ? Math.round( Math.abs( a - e ) * 255 ) : Math.abs( halfOrder( a ) - halfOrder( e ) );
		if ( steps > out.worstSteps ) { out.worstSteps = steps; out.worstLevel = l + 1; }

	} ) );
	return out;

}

/** Halves in [16384, 65504]: f16 sums of four overflow. */
function largeHalves( width, height, seed ) {

	const data = new Uint16Array( width * height * 4 );
	let state = seed;
	for ( let i = 0; i < data.length; i ++ ) {

		state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0;
		data[ i ] = toHalf( 16384 + ( state >>> 8 ) / 2 ** 24 * ( 65504 - 16384 ) );

	}

	return data;

}

export async function kernelVariants( device, { probe, sizes = variantSizes, formats: benchFormats = variantFormats, rounds = 11, repeats = 16, variants, throughput = null, onProgress = () => {} } = {} ) {

	const spec = probes[ probe ];
	const chosen = variants?.length ? [ ...new Set( [ spec.baseline, ...variants ] ) ] : spec.variants;
	const unknown = chosen.filter( name => ! spec.variants.includes( name ) );
	if ( unknown.length ) throw new Error( `--compare=${ probe }: unknown variants ${ unknown.join( ', ' ) }; expected ${ spec.variants.join( ', ' ) }` );
	const skipped = chosen.filter( name => spec.needs[ name ] && ! device.features.has( spec.needs[ name ] ) );
	const names = chosen.filter( name => ! skipped.includes( name ) );
	const timestamps = device.features.has( 'timestamp-query' );
	const downsampler = new Downsampler( device );
	const cache = new Map();
	const results = [];
	device.pushErrorScope( 'validation' );

	try {

		for ( const format of benchFormats ) for ( const [ width, height ] of sizes ) {

			const quantization = formats[ format ].quantization;
			const textures = Object.fromEntries( names.map( name => [ name, createTexture( device, { width, height, format, render: true } ) ] ) );

			try {

				const chains = Object.fromEntries( names.map( name => [ name, variantChain( device, downsampler, textures[ name ], name, cache ) ] ) );
				const run = name => {

					const encoder = device.createCommandEncoder();
					chains[ name ].encode( encoder );
					device.queue.submit( [ encoder.finish() ] );

				};

				// Check on a second pattern, then on the timed input (left in place for timing).
				const checks = {};
				const patterns = [ [ quantization === 'float16' ? 'large' : 'tiles', quantization === 'float16' ? largeHalves( width, height, 7 ) : makeData( width, height, 1, format, 'tiles', 7 ) ], [ 'random', makeData( width, height, 1, format, 'random', 1337 ) ] ];
				for ( const [ pattern, data ] of patterns ) {

					for ( const name of names ) { upload( device, textures[ name ], data ); run( name ); }
					const expected = await readMips( device, textures[ spec.baseline ], 1 );
					for ( const name of names.filter( name => name !== spec.baseline && ! timingOnly.has( name ) ) ) {

						( checks[ name ] ??= {} )[ pattern ] = difference( await readMips( device, textures[ name ], 1 ), expected, quantization );

					}

				}

				for ( let i = 0; i < 8; i ++ ) for ( const name of names ) run( name );
				await device.queue.onSubmittedWorkDone();
				const result = { width, height, format, batches: chains[ spec.baseline ].batches, checks };

				if ( timestamps ) {

					const order = [ 'marker', ...names ];
					const samples = Object.fromEntries( order.map( name => [ name, [] ] ) );
					for ( let round = 0; round < rounds; round ++ ) {

						for ( const name of [ ...order.slice( round % order.length ), ...order.slice( 0, round % order.length ) ] ) {

							const texture = textures[ name === 'marker' ? spec.baseline : name ];
							samples[ name ].push( await timeGpu( device, texture, name === 'marker' ? () => {} : encoder => chains[ name ].encode( encoder ), repeats ) );

						}

					}

					const marker = median( samples.marker );
					result.gpuMs = Object.fromEntries( names.map( name => [ name, median( samples[ name ] ) - marker ] ) );
					result.ratios = Object.fromEntries( names.map( name => [ name, median( samples[ name ].map( ( value, round ) => ( value - marker ) / ( samples[ spec.baseline ][ round ] - marker ) ) ) ] ) );
					result.markerMs = marker;
					result.samples = samples;

				}

				if ( throughput ) {

					const measured = await measureUnits( device, Object.fromEntries( names.map( name => [ name, encoder => chains[ name ].encode( encoder ) ] ) ), { rounds: Math.min( rounds, 5 ), reference: () => spec.baseline, ...throughput } );
					result.throughput = measured.units;

				}

				results.push( result );
				onProgress( `${ probe } ${ width }x${ height } ${ format }: ${ names.map( name => `${ name } ${ ( result.ratios ?? {} )[ name ]?.toFixed( 3 ) ?? result.throughput?.[ name ].ratio.toFixed( 3 ) }` ).join( ', ' ) }` );

			} finally {

				for ( const texture of Object.values( textures ) ) texture.destroy();

			}

		}

	} finally {

		const error = await device.popErrorScope();
		if ( error ) throw new Error( `GPU validation error: ${ error.message }` );

	}

	// Summary per format and variant: median ratio over the render sizes and over the squares.
	const render = row => row.width !== row.height;
	const summary = {};
	for ( const format of benchFormats ) for ( const name of names ) {

		const rows = results.filter( row => row.format === format );
		const ratio = row => row.ratios?.[ name ] ?? row.throughput?.[ name ].ratio;
		const of = list => list.length ? median( list.map( ratio ) ) : null;
		const checked = rows.flatMap( row => Object.values( row.checks[ name ] ?? {} ) );
		summary[ `${ format } ${ name }` ] = {
			renderSizes: of( rows.filter( render ) ), squares: of( rows.filter( row => ! render( row ) ) ),
			...( checked.length ? { identical: checked.every( c => c.differing === 0 ), worstSteps: Math.max( ...checked.map( c => c.worstSteps ) ), nonFinite: checked.reduce( ( n, c ) => n + c.nonFinite, 0 ), differing: checked.reduce( ( n, c ) => n + c.differing, 0 ), values: checked.reduce( ( n, c ) => n + c.values, 0 ) } : {} )
		};

	}

	return { mode: probe, timestamps, rounds, repeats, throughput, variants: names, skipped, exactTile: downsampler.exactTile, tail: downsampler.tail, results, summary };

}
