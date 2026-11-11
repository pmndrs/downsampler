/**
 * Scalar reductions: a whole image (per layer) to one vec4 and a weight, in two
 * dispatches (SPEC.md §4 Scalar reductions). This file holds the backend-neutral
 * part: per-channel ops, the dispatch geometry, and the CPU oracle, which pairs
 * values in exactly the order the GPU kernels do and reports an error bound.
 *
 * Order of combination (the oracle reproduces it):
 * 1. Invocation (lx, ly) of workgroup (gx, gy) reads `block × block` texels at
 *    `tileOrigin + (lx + 16 jx, ly + 16 jy)`, j = jy · block + jx, and combines
 *    them as a stride-halving tree: v[j] ∘= v[j + stride], stride = block²/2 … 1.
 * 2. The workgroup combines its 256 invocation values (slot ly · 16 + lx) with
 *    the same stride-halving tree, 128 … 1, and writes one partial.
 * 3. One workgroup per layer: invocation t folds partials t, t + 256, t + 512 …
 *    in sequence, then the 256 results are combined with the same tree.
 * Off-image reads are clamped to the edge; sum-like channels zero them, and
 * min/max see a duplicate of an in-image texel, which is harmless.
 */

export type ScalarOp = 'sum' | 'average' | 'min' | 'max' | 'weight' | 'weighted';
export type ScalarReduce = ScalarOp | readonly [ ScalarOp, ScalarOp, ScalarOp, ScalarOp ];

const scalarOpNames: readonly ScalarOp[] = [ 'sum', 'average', 'min', 'max', 'weight', 'weighted' ];

/** Invocations per workgroup side (16 × 16 = 256). */
export const scalarSide = 16;
/** Invocations per workgroup. */
export const scalarInvocations = scalarSide * scalarSide;
/** Default texels per invocation along each axis: 4 × 4, a 64 × 64 tile per workgroup (§4 Scalar reductions). */
export const scalarBlock = 4;

/** Validated per-channel ops. */
export function scalarOps( reduce: ScalarReduce = 'average' ): ScalarOp[] {

	const ops = typeof reduce === 'string' ? [ reduce, reduce, reduce, reduce ] : Array.from( reduce );
	if ( ops.length !== 4 ) throw new Error( 'downsampler: a scalar reduction takes one op or four (one per channel).' );
	for ( const op of ops ) if ( ! scalarOpNames.includes( op ) ) throw new Error( `downsampler: unknown scalar op ${ JSON.stringify( op ) }.` );
	const weights = ops.filter( op => op === 'weight' ).length;
	if ( weights > 1 ) throw new Error( 'downsampler: at most one channel can be the weight.' );
	if ( weights === 0 && ops.includes( 'weighted' ) ) throw new Error( 'downsampler: "weighted" channels need a "weight" channel.' );
	return ops;

}

/** The channel holding the weight, or -1. */
export function scalarWeightChannel( ops: readonly ScalarOp[] ): number {

	return ops.indexOf( 'weight' );

}

/** Channels combined by addition (everything except min and max). */
export function isSumOp( op: ScalarOp ): boolean {

	return op !== 'min' && op !== 'max';

}

/** Workgroups per layer of the partials dispatch. */
export function scalarWorkgroups( width: number, height: number, block = scalarBlock ): [ number, number ] {

	const tile = scalarSide * block;
	return [ Math.ceil( width / tile ), Math.ceil( height / tile ) ];

}

/**
 * Longest chain of roundings any texel's term passes through on its way to the
 * total: the in-invocation tree, the workgroup tree, the sequential fold over
 * partials in the combine pass, and its tree.
 */
export function scalarDepth( width: number, height: number, block = scalarBlock ): number {

	const [ gx, gy ] = scalarWorkgroups( width, height, block );
	const sequential = Math.ceil( gx * gy / scalarInvocations ) - 1;
	return Math.log2( block * block ) + Math.log2( scalarInvocations ) + sequential + Math.log2( scalarInvocations );

}

export interface ScalarReferenceOptions {
	/** Texels per invocation along each axis. Default {@link scalarBlock}. */
	block?: number;
	/**
	 * Error allowed in each input value before it reaches the reduction, e.g. a
	 * load hook's `log` (not correctly rounded on GPUs) or unorm → float
	 * conversion: `absolute + relative · |v|` per value. Default 0.
	 */
	inputError?: { absolute?: number, relative?: number };
}

export interface ScalarReferenceResult {
	/** Emitted value: sums, averages (÷ texel count), weighted means (÷ Σ w), Σ w, min, max. */
	value: number[];
	/** Σ w with a weight channel, otherwise the texel count. */
	weight: number;
	/** The same results computed in float64, in plain order: the exact answer for the stated bound. */
	exact: number[];
	exactWeight: number;
	/** Bound on |GPU − exact| per channel (see {@link scalarReference}). */
	bound: number[];
	weightBound: number;
}

const u = 2 ** - 24;
const gamma = ( n: number ) => n * u / ( 1 - n * u );

/**
 * The scalar reduction of `texels` (RGBA floats, layer-major rows, as `toFloats`
 * returns them), one result per layer, combined in the GPU's order with float32
 * rounding after every operation.
 *
 * Error bound (per channel, against the exact float64 result of the same f32
 * inputs). With D = {@link scalarDepth}, γₙ = n·u/(1 − n·u), u = 2⁻²⁴ and the
 * sums A = Σ|tᵢ| of the channel's terms:
 * - 'sum': γ_D · A. This is the classic bound for a summation tree of depth D
 *   (Higham, *Accuracy and Stability*, §4.2); every term passes through at most
 *   D roundings.
 * - 'average': γ_D · A / n + u·|r| (the division rounds once; a count above 2²⁴
 *   rounds once more).
 * - 'weight': γ_D · Σ|w|.
 * - 'weighted': (γ_{D+1} · Σ|w·v| + |r| · γ_D · Σ|w|) / W + u·|r|: the product
 *   rounds once more, and the denominator carries its own sum error.
 * - 'min', 'max': exact (they select a value).
 * Input error adds `(absolute · n + relative · A) / normaliser` to sum-like
 * channels and `absolute + relative · |r|` to min and max.
 */
export function scalarReference( texels: ArrayLike<number>, width: number, height: number, layers: number, reduce: ScalarReduce = 'average', options: ScalarReferenceOptions = {} ): ScalarReferenceResult[] {

	const ops = scalarOps( reduce );
	const block = options.block ?? scalarBlock;
	const absolute = options.inputError?.absolute ?? 0, relative = options.inputError?.relative ?? 0;
	const weightChannel = scalarWeightChannel( ops );
	const sums = ops.map( isSumOp );
	const [ gx, gy ] = scalarWorkgroups( width, height, block );
	const partialCount = gx * gy;
	const tile = scalarSide * block;
	const blockCount = block * block;
	const depth = scalarDepth( width, height, block );
	const count = width * height;
	const results: ScalarReferenceResult[] = [];

	// Float32Array stores round to float32; a sum or product of two float32
	// values computed in float64 and then rounded is correctly rounded.
	const combineInto = ( target: Float32Array, a: number, b: number ) => {

		for ( let c = 0; c < 4; c ++ ) {

			const x = target[ a * 4 + c ], y = target[ b * 4 + c ];
			target[ a * 4 + c ] = ops[ c ] === 'min' ? Math.min( x, y ) : ops[ c ] === 'max' ? Math.max( x, y ) : x + y;

		}

	};

	const tree = ( values: Float32Array, length: number ) => {

		for ( let stride = length >> 1; stride > 0; stride >>= 1 ) for ( let i = 0; i < stride; i ++ ) combineInto( values, i, i + stride );

	};

	const values = new Float32Array( blockCount * 4 );
	const slots = new Float32Array( scalarInvocations * 4 );
	const partials = new Float32Array( partialCount * 4 );

	for ( let layer = 0; layer < layers; layer ++ ) {

		const base = layer * width * height;
		const texel = ( x: number, y: number, c: number ) => texels[ ( base + y * width + x ) * 4 + c ];
		const term = ( x: number, y: number, c: number ) => ops[ c ] === 'weighted' ? Math.fround( texel( x, y, c ) * texel( x, y, weightChannel ) ) : texel( x, y, c );

		for ( let py = 0; py < gy; py ++ ) for ( let px = 0; px < gx; px ++ ) {

			for ( let ly = 0; ly < scalarSide; ly ++ ) for ( let lx = 0; lx < scalarSide; lx ++ ) {

				for ( let jy = 0; jy < block; jy ++ ) for ( let jx = 0; jx < block; jx ++ ) {

					const x = px * tile + lx + jx * scalarSide, y = py * tile + ly + jy * scalarSide;
					const inside = x < width && y < height;
					const cx = Math.min( x, width - 1 ), cy = Math.min( y, height - 1 );
					for ( let c = 0; c < 4; c ++ ) values[ ( jy * block + jx ) * 4 + c ] = sums[ c ] && ! inside ? 0 : term( cx, cy, c );

				}

				tree( values, blockCount );
				slots.set( values.subarray( 0, 4 ), ( ly * scalarSide + lx ) * 4 );

			}

			tree( slots, scalarInvocations );
			partials.set( slots.subarray( 0, 4 ), ( py * gx + px ) * 4 );

		}

		for ( let t = 0; t < scalarInvocations; t ++ ) {

			const first = Math.min( t, partialCount - 1 );
			for ( let c = 0; c < 4; c ++ ) slots[ t * 4 + c ] = sums[ c ] && t >= partialCount ? 0 : partials[ first * 4 + c ];
			for ( let i = t + scalarInvocations; i < partialCount; i += scalarInvocations ) {

				for ( let c = 0; c < 4; c ++ ) {

					const x = slots[ t * 4 + c ], y = partials[ i * 4 + c ];
					slots[ t * 4 + c ] = ops[ c ] === 'min' ? Math.min( x, y ) : ops[ c ] === 'max' ? Math.max( x, y ) : x + y;

				}

			}

		}

		tree( slots, scalarInvocations );
		const total = Array.from( slots.subarray( 0, 4 ) );

		// Exact (float64) totals and the magnitudes the bound needs.
		const exactTotal = ops.map( op => op === 'min' ? Infinity : op === 'max' ? - Infinity : 0 );
		const magnitude = [ 0, 0, 0, 0 ];
		for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) for ( let c = 0; c < 4; c ++ ) {

			const v = ops[ c ] === 'weighted' ? texel( x, y, c ) * texel( x, y, weightChannel ) : texel( x, y, c );
			if ( ops[ c ] === 'min' ) exactTotal[ c ] = Math.min( exactTotal[ c ], v );
			else if ( ops[ c ] === 'max' ) exactTotal[ c ] = Math.max( exactTotal[ c ], v );
			else { exactTotal[ c ] += v; magnitude[ c ] += Math.abs( v ); }

		}

		const weight = weightChannel >= 0 ? total[ weightChannel ] : Math.fround( count );
		const exactWeight = weightChannel >= 0 ? exactTotal[ weightChannel ] : count;
		const weightMagnitude = weightChannel >= 0 ? magnitude[ weightChannel ] : count;
		const weightBound = weightChannel >= 0 ? gamma( depth ) * weightMagnitude + absolute * count + relative * weightMagnitude : 0;
		const countRounding = count > 2 ** 24 ? u : 0;
		const value = [ 0, 0, 0, 0 ], exact = [ 0, 0, 0, 0 ], bound = [ 0, 0, 0, 0 ];

		for ( let c = 0; c < 4; c ++ ) {

			const op = ops[ c ], input = absolute * count + relative * magnitude[ c ];

			if ( op === 'min' || op === 'max' ) {

				value[ c ] = total[ c ]; exact[ c ] = exactTotal[ c ];
				bound[ c ] = absolute + relative * Math.abs( exact[ c ] );

			} else if ( op === 'sum' || op === 'weight' ) {

				value[ c ] = total[ c ]; exact[ c ] = exactTotal[ c ];
				bound[ c ] = gamma( depth ) * magnitude[ c ] + input;

			} else if ( op === 'average' ) {

				value[ c ] = Math.fround( total[ c ] / Math.fround( count ) ); exact[ c ] = exactTotal[ c ] / count;
				bound[ c ] = ( gamma( depth ) * magnitude[ c ] + input ) / count + ( u + countRounding ) * Math.abs( exact[ c ] );

			} else {

				value[ c ] = weight === 0 ? 0 : Math.fround( total[ c ] / weight );
				exact[ c ] = exactWeight === 0 ? 0 : exactTotal[ c ] / exactWeight;
				bound[ c ] = exactWeight === 0 ? 0 : ( gamma( depth + 1 ) * magnitude[ c ] + input + Math.abs( exact[ c ] ) * weightBound ) / Math.abs( exactWeight ) + u * Math.abs( exact[ c ] );

			}

		}

		results.push( { value, weight, exact, exactWeight, bound, weightBound } );

	}

	return results;

}
