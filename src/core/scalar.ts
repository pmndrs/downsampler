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

/** Rounding depth of the combine pass over `count` partials: the sequential fold, then the 256-slot tree. */
export function scalarCombineDepth( count: number ): number {

	return Math.ceil( count / scalarInvocations ) - 1 + Math.log2( scalarInvocations );

}

/**
 * Combines vector `b` of `source` into vector `a` of `target` (`ops.length`
 * channels each), in float32: Float32Array stores round, and a sum of two
 * float32 values computed in float64 and then rounded is correctly rounded.
 */
function combiner( ops: readonly ScalarOp[] ) {

	const n = ops.length;
	return ( target: Float32Array, a: number, source: Float32Array, b: number ) => {

		for ( let c = 0; c < n; c ++ ) {

			const x = target[ a * n + c ], y = source[ b * n + c ];
			target[ a * n + c ] = ops[ c ] === 'min' ? Math.min( x, y ) : ops[ c ] === 'max' ? Math.max( x, y ) : x + y;

		}

	};

}

/** The combine pass of one layer, in the GPU's order: the per-invocation sequential fold, then the tree. */
function combinePartials( ops: readonly ScalarOp[], partials: Float32Array, offset: number, count: number ): number[] {

	const n = ops.length;
	const combine = combiner( ops );
	const slots = new Float32Array( scalarInvocations * n );
	for ( let t = 0; t < scalarInvocations; t ++ ) {

		const first = offset + Math.min( t, count - 1 );
		for ( let c = 0; c < n; c ++ ) slots[ t * n + c ] = isSumOp( ops[ c ] ) && t >= count ? 0 : partials[ first * n + c ];
		for ( let i = t + scalarInvocations; i < count; i += scalarInvocations ) combine( slots, t, partials, offset + i );

	}

	for ( let stride = scalarInvocations >> 1; stride > 0; stride >>= 1 ) for ( let i = 0; i < stride; i ++ ) combine( slots, i, slots, i + stride );
	return Array.from( slots.subarray( 0, n ) );

}

interface Exact {
	/** Per channel: Σ terms (sum-like) or the min/max, in float64. */
	total: number[];
	/** Per channel: Σ |terms| (sum-like). */
	magnitude: number[];
}

/**
 * Normalises a GPU-order total and an exact total into a result, with the
 * error bound for `depth` roundings (see {@link scalarReference}). `count` is
 * the texel count; `inputs` the number of inputs that carry `absolute` error.
 */
function scalarResult( ops: readonly ScalarOp[], total: number[], exactTotals: Exact, count: number, depth: number, absolute: number, relative: number, inputs = count ): ScalarReferenceResult {

	const weightChannel = scalarWeightChannel( ops );
	const { total: exactTotal, magnitude } = exactTotals;
	const n = ops.length;
	const weight = weightChannel >= 0 ? total[ weightChannel ] : Math.fround( count );
	const exactWeight = weightChannel >= 0 ? exactTotal[ weightChannel ] : count;
	const weightMagnitude = weightChannel >= 0 ? magnitude[ weightChannel ] : count;
	const weightBound = weightChannel >= 0 ? gamma( depth ) * weightMagnitude + absolute * inputs + relative * weightMagnitude : 0;
	const countRounding = count > 2 ** 24 ? u : 0;
	const value = new Array<number>( n ).fill( 0 ), exact = new Array<number>( n ).fill( 0 ), bound = new Array<number>( n ).fill( 0 );

	for ( let c = 0; c < n; c ++ ) {

		const op = ops[ c ], input = absolute * inputs + relative * magnitude[ c ];

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

	return { value, weight, exact, exactWeight, bound, weightBound };

}

/** Exact totals over `length` vectors from `term( i, c )`. */
function exactOf( ops: readonly ScalarOp[], length: number, term: ( i: number, c: number ) => number ): Exact {

	const total = ops.map( op => op === 'min' ? Infinity : op === 'max' ? - Infinity : 0 );
	const magnitude = ops.map( () => 0 );
	for ( let i = 0; i < length; i ++ ) for ( let c = 0; c < ops.length; c ++ ) {

		const v = term( i, c );
		if ( ops[ c ] === 'min' ) total[ c ] = Math.min( total[ c ], v );
		else if ( ops[ c ] === 'max' ) total[ c ] = Math.max( total[ c ], v );
		else { total[ c ] += v; magnitude[ c ] += Math.abs( v ); }

	}

	return { total, magnitude };

}

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
	const n = ops.length;
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
	const combine = combiner( ops );
	const results: ScalarReferenceResult[] = [];

	const tree = ( values: Float32Array, length: number ) => {

		for ( let stride = length >> 1; stride > 0; stride >>= 1 ) for ( let i = 0; i < stride; i ++ ) combine( values, i, values, i + stride );

	};

	const values = new Float32Array( blockCount * n );
	const slots = new Float32Array( scalarInvocations * n );
	const partials = new Float32Array( partialCount * n );

	for ( let layer = 0; layer < layers; layer ++ ) {

		const base = layer * width * height;
		const texel = ( x: number, y: number, c: number ) => texels[ ( base + y * width + x ) * n + c ];
		const term = ( x: number, y: number, c: number ) => ops[ c ] === 'weighted' ? Math.fround( texel( x, y, c ) * texel( x, y, weightChannel ) ) : texel( x, y, c );

		for ( let py = 0; py < gy; py ++ ) for ( let px = 0; px < gx; px ++ ) {

			for ( let ly = 0; ly < scalarSide; ly ++ ) for ( let lx = 0; lx < scalarSide; lx ++ ) {

				for ( let jy = 0; jy < block; jy ++ ) for ( let jx = 0; jx < block; jx ++ ) {

					const x = px * tile + lx + jx * scalarSide, y = py * tile + ly + jy * scalarSide;
					const inside = x < width && y < height;
					const cx = Math.min( x, width - 1 ), cy = Math.min( y, height - 1 );
					for ( let c = 0; c < n; c ++ ) values[ ( jy * block + jx ) * n + c ] = sums[ c ] && ! inside ? 0 : term( cx, cy, c );

				}

				tree( values, blockCount );
				slots.set( values.subarray( 0, n ), ( ly * scalarSide + lx ) * n );

			}

			tree( slots, scalarInvocations );
			partials.set( slots.subarray( 0, n ), ( py * gx + px ) * n );

		}

		const total = combinePartials( ops, partials, 0, partialCount );
		const at = ( i: number, c: number ) => texels[ ( base + i ) * n + c ];
		const exact = exactOf( ops, count, ( i, c ) => ops[ c ] === 'weighted' ? at( i, c ) * at( i, weightChannel ) : at( i, c ) );
		results.push( scalarResult( ops, total, exact, count, depth, absolute, relative ) );

	}

	return results;

}

export interface ScalarCombineReferenceOptions {
	/**
	 * Texels behind the partials: the divisor of 'average' channels, and the
	 * weight passed on without a 'weight' channel. Default 0.
	 */
	texels?: number;
	/**
	 * Roundings the partials' terms already went through before the combine
	 * pass (e.g. 12 for a Reducer's level 6: two per level), added to the
	 * combine depth in the bound. Default 0: the partials are exact inputs.
	 */
	partialDepth?: number;
	/**
	 * Per channel, Σ|terms| behind the partials, for the bound when
	 * `partialDepth` > 0 (it is at least Σ|partials|). Default Σ|partials|.
	 */
	magnitude?: readonly number[];
	/** As {@link ScalarReferenceOptions.inputError}, per partial. */
	inputError?: { absolute?: number, relative?: number };
}

/**
 * A combine-only scalar reduction (`ScalarRun.partials`): `partials` holds
 * `count` accumulations per layer (`ops.length` floats each, layer-major), as
 * a tool or a Reducer's store hook wrote them. Sum-like channels hold sums
 * (for 'weighted' channels Σ w·v), min/max channels the min/max. The result
 * per layer is combined in the GPU's order, with the exact float64 result and
 * the bound of {@link scalarReference} at depth `partialDepth` +
 * {@link scalarCombineDepth}.
 */
export function scalarCombineReference( partials: ArrayLike<number>, count: number, layers: number, reduce: ScalarReduce = 'average', options: ScalarCombineReferenceOptions = {} ): ScalarReferenceResult[] {

	const ops = scalarOps( reduce );
	const n = ops.length;
	if ( ! Number.isInteger( count ) || count < 1 ) throw new Error( 'downsampler: a combine needs at least one partial.' );
	const values = Float32Array.from( partials );
	const absolute = options.inputError?.absolute ?? 0, relative = options.inputError?.relative ?? 0;
	const depth = ( options.partialDepth ?? 0 ) + scalarCombineDepth( count );
	const results: ScalarReferenceResult[] = [];

	for ( let layer = 0; layer < layers; layer ++ ) {

		const total = combinePartials( ops, values, layer * count, count );
		const exact = exactOf( ops, count, ( i, c ) => values[ ( layer * count + i ) * n + c ] );
		if ( options.magnitude !== undefined ) exact.magnitude = ops.map( ( _, c ) => Math.max( exact.magnitude[ c ], options.magnitude![ c ] ) );
		results.push( scalarResult( ops, total, exact, options.texels ?? 0, depth, absolute, relative, count ) );

	}

	return results;

}
