import { test } from 'node:test';
import assert from 'node:assert/strict';
import { clampDependents, exactDependents, footprint, layersProblem, mipLevelCountFor, mipSize, planBatches, planRegions, rectProblem, reduceReference, spdChainReference } from '../../dist/index.js';

// Brute force: every texel of a `size`-texel level whose source texels (`reads( x )`) meet [ a, b ].
function bruteForce( size, reads, a, b ) {

	let lo = Infinity, hi = - Infinity;
	for ( let x = 0; x < size; x ++ ) if ( reads( x ).some( s => s >= a && s <= b ) ) { lo = Math.min( lo, x ); hi = Math.max( hi, x ); }
	return lo === Infinity ? null : [ lo, hi ];

}

const span = ( [ lo, hi ] ) => Array.from( { length: hi - lo + 1 }, ( _, i ) => lo + i );
// The texels the 'exact' kernels' shader-weighted bilinear sample reads (integer position, as in kernel.ts).
function bilinear( x, sourceSize, size ) {

	const numerator = ( 2 * x + 1 ) * sourceSize - size, denominator = 2 * size;
	const lo = Math.min( Math.floor( numerator / denominator ), sourceSize - 1 );
	return [ lo, Math.min( lo + 1, sourceSize - 1 ) ];

}

test( 'exact dependents are the inverse of the fold footprint, and cover the bilinear sample', () => {

	for ( let S = 1; S <= 41; S ++ ) {

		const n = Math.max( 1, S >> 1 );
		for ( let a = 0; a < S; a ++ ) for ( let b = a; b < S; b ++ ) {

			assert.deepEqual( exactDependents( [ a, b ], S, n ), bruteForce( n, x => span( footprint( x, S ) ), a, b ), `S ${ S } [${ a }, ${ b }]` );
			// Copies read their own texel.
			assert.deepEqual( exactDependents( [ a, b ], S, S ), [ a, b ] );

		}

		// The average's two taps lie in the footprint, so the fold rule is conservative for it.
		for ( let x = 0; x < n; x ++ ) {

			const [ lo, hi ] = footprint( x, S );
			for ( const s of bilinear( x, S, n ) ) assert.ok( s >= lo && s <= hi, `S ${ S } x ${ x }` );
			// A sampler that truncates its position (Intel) moves a tap of weight 0 one texel lower: still inside.
			if ( ( ( 2 * x + 1 ) * S - n ) % ( 2 * n ) === 0 ) assert.ok( Math.max( 0, bilinear( x, S, n )[ 0 ] - 1 ) >= lo, `S ${ S } x ${ x } truncated` );

		}

	}

	assert.equal( exactDependents( null, 9, 4 ), null );

} );

test( 'clamp dependents are the texels whose clamped block meets the range', () => {

	for ( let S = 1; S <= 40; S ++ ) for ( let l = 1; l <= 6; l ++ ) {

		const n = Math.max( 1, S >> l );
		const block = x => Array.from( { length: 2 ** l }, ( _, i ) => Math.min( x * 2 ** l + i, S - 1 ) );
		for ( let a = 0; a < S; a ++ ) for ( let b = a; b < S; b ++ ) assert.deepEqual( clampDependents( [ a, b ], l, n ), bruteForce( n, block, a, b ), `S ${ S } l ${ l } [${ a }, ${ b }]` );

	}

	// 5 → 2: texel 4 is dropped; 3 → 1 → 1: texel 2 is dropped at level 1 but reaches level 2 through the clamp.
	assert.equal( clampDependents( [ 4, 4 ], 1, 2 ), null );
	assert.equal( clampDependents( [ 2, 2 ], 1, 1 ), null );
	assert.deepEqual( clampDependents( [ 2, 2 ], 2, 1 ), [ 0, 0 ] );

} );

/**
 * Brute-force propagation of a dirty mask through a plan: per level, the texels whose footprint
 * ('exact', 'tail': from the previous level; 'clamp': the clamped block of the batch source) meets
 * the dirty texels. Returns level → Set of 'x,y'.
 */
function propagate( batches, rect, width, height, sourceWidth, sourceHeight ) {

	const masks = new Map();
	const sizeOf = level => level === null ? [ sourceWidth, sourceHeight ] : [ mipSize( width, level ), mipSize( height, level ) ];
	const start = new Set();
	for ( let y = rect.y; y < rect.y + rect.height; y ++ ) for ( let x = rect.x; x < rect.x + rect.width; x ++ ) start.add( `${ x },${ y }` );
	masks.set( batches[ 0 ].sourceMipLevel, start );
	const hit = ( mask, xs, ys ) => xs.some( sx => ys.some( sy => mask.has( `${ sx },${ sy }` ) ) );

	for ( const batch of batches ) {

		const source = masks.get( batch.sourceMipLevel ), [ sw, sh ] = sizeOf( batch.sourceMipLevel );
		let previous = source, [ pw, ph ] = [ sw, sh ];
		for ( let i = 0; i < batch.count; i ++ ) {

			const level = batch.firstMipLevel + i, [ w, h ] = sizeOf( level ), mask = new Set();
			for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

				let xs, ys, from = previous;
				if ( batch.kernel === 'clamp' ) {

					const block = ( c, size ) => Array.from( { length: 2 ** ( i + 1 ) }, ( _, k ) => Math.min( c * 2 ** ( i + 1 ) + k, size - 1 ) );
					xs = block( x, sw ); ys = block( y, sh ); from = source;

				} else {

					xs = w === pw ? [ x ] : span( footprint( x, pw ) );
					ys = h === ph ? [ y ] : span( footprint( y, ph ) );

				}

				if ( hit( from, xs, ys ) ) mask.add( `${ x },${ y }` );

			}

			masks.set( level, mask );
			previous = mask; pw = w; ph = h;

		}

	}

	return masks;

}

const bounds = mask => {

	if ( mask.size === 0 ) return null;
	const points = [ ...mask ].map( key => key.split( ',' ).map( Number ) );
	const xs = points.map( p => p[ 0 ] ), ys = points.map( p => p[ 1 ] );
	const x = Math.min( ...xs ), y = Math.min( ...ys );
	const rect = { x, y, width: Math.max( ...xs ) - x + 1, height: Math.max( ...ys ) - y + 1 };
	// Separable footprints keep the dirty set a rectangle.
	assert.equal( mask.size, rect.width * rect.height );
	return rect;

};

function rectsFor( width, height, seed ) {

	let state = seed;
	const next = n => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) % n;
	const rects = [
		{ x: 0, y: 0, width: 1, height: 1 }, { x: width - 1, y: height - 1, width: 1, height: 1 },
		{ x: width - 1, y: 0, width: 1, height }, { x: 0, y: height - 1, width, height: 1 }, { x: 0, y: 0, width, height }
	];
	for ( let i = 0; i < 4; i ++ ) {

		const x = next( width ), y = next( height );
		rects.push( { x, y, width: 1 + next( width - x ), height: 1 + next( height - y ) } );

	}

	return rects;

}

const planShapes = [ [ 1, 1 ], [ 3, 7 ], [ 17, 9 ], [ 31, 33 ], [ 65, 129 ], [ 1, 37 ], [ 77, 1 ], [ 133, 75 ], [ 255, 257 ], [ 90, 60 ] ];
const planVariants = [
	...[ 1, 4, 6 ].map( levels => ( { levelsPerDispatch: levels } ) ),
	{ levelsPerDispatch: 4, exactMinTile: 32 },
	{ levelsPerDispatch: 6, tail: true }, { levelsPerDispatch: 4, tail: true, tailMaxTile: 8 },
	{ levelsPerDispatch: 4, oddSizes: 'spd' }, { levelsPerDispatch: 6, oddSizes: 'spd' }, { levelsPerDispatch: 1, oddSizes: 'spd' },
	{ levelsPerDispatch: 4, baseMipLevel: 1 }, { levelsPerDispatch: 6, baseMipLevel: 1, oddSizes: 'spd' }
];

test( 'planRegions matches brute-force propagation, and its workgroups cover every written texel', () => {

	for ( const [ width, height ] of planShapes ) for ( const variant of planVariants ) for ( const source of [ null, 'same', 'half' ] ) {

		const { baseMipLevel = 0 } = variant;
		const levels = mipLevelCountFor( width, height );
		if ( baseMipLevel >= levels ) continue;
		// A separate source: the destination's first level the source's size, or half of it.
		const bw = mipSize( width, baseMipLevel ), bh = mipSize( height, baseMipLevel );
		const sw = source === 'half' ? bw * 2 + 1 : bw, sh = source === 'half' ? bh * 2 : bh;
		const fromSource = source !== null;
		const batches = planBatches( width, height, { ...variant, mipLevelCount: levels - baseMipLevel, fromSource, copySource: source === 'same' } );
		if ( batches.length === 0 ) continue;
		const rw = fromSource ? sw : mipSize( width, baseMipLevel ), rh = fromSource ? sh : mipSize( height, baseMipLevel );

		for ( const rect of rectsFor( rw, rh, width * 31 + height ) ) {

			const label = `${ width }x${ height } ${ JSON.stringify( variant ) } ${ source } ${ JSON.stringify( rect ) }`;
			const regions = planRegions( batches, rect, width, height, sw, sh );
			const masks = propagate( batches, rect, width, height, sw, sh );
			batches.forEach( ( batch, b ) => {

				const region = regions[ b ];
				assert.equal( region.levels.length, batch.count, label );
				region.levels.forEach( ( level, i ) => {

					assert.deepEqual( level, bounds( masks.get( batch.firstMipLevel + i ) ), `${ label } level ${ batch.firstMipLevel + i }` );
					if ( level === null ) return;
					const extent = batch.kernel === 'tail' ? Infinity : batch.tile >> i;
					const group = c => batch.kernel === 'tail' ? 0 : Math.floor( c / extent );
					for ( const [ x, y ] of [ [ level.x, level.y ], [ level.x + level.width - 1, level.y + level.height - 1 ] ] ) {

						assert.ok( group( x ) >= region.groupX && group( x ) < region.groupX + region.groupsX, `${ label } x ${ x }` );
						assert.ok( group( y ) >= region.groupY && group( y ) < region.groupY + region.groupsY, `${ label } y ${ y }` );

					}

				} );
				// No more workgroups than the full dispatch; none for a batch that writes nothing.
				assert.ok( region.groupX + region.groupsX <= batch.groupsX && region.groupY + region.groupsY <= batch.groupsY, label );
				if ( region.levels.every( level => level === null ) ) assert.equal( region.groupsX * region.groupsY, 0, label );

			} );

		}

	}

} );

test( 'exact regions are always non-empty, and a whole-level rect writes whole levels', () => {

	for ( const [ width, height ] of planShapes ) {

		const batches = planBatches( width, height, { mipLevelCount: mipLevelCountFor( width, height ), levelsPerDispatch: 4, tail: true } );
		const regions = planRegions( batches, { x: 0, y: 0, width, height }, width, height );
		regions.forEach( ( region, b ) => {

			const batch = batches[ b ];
			region.levels.forEach( ( level, i ) => assert.deepEqual( level, { x: 0, y: 0, width: mipSize( width, batch.firstMipLevel + i ), height: mipSize( height, batch.firstMipLevel + i ) } ) );
			assert.deepEqual( [ region.groupX, region.groupY, region.groupsX, region.groupsY ], [ 0, 0, batch.groupsX, batch.groupsY ] );

		} );

	}

} );

// CPU chains of A and of B (A changed inside a rect): every texel that differs lies in its level's region.
function changedOutside( chainA, chainB, regions, batches, width, height, layers ) {

	const outside = [];
	batches.forEach( ( batch, b ) => regions[ b ].levels.forEach( ( region, i ) => {

		const level = batch.firstMipLevel + i, w = mipSize( width, level ), h = mipSize( height, level );
		for ( let layer = 0; layer < layers; layer ++ ) for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

			const inside = region !== null && x >= region.x && x < region.x + region.width && y >= region.y && y < region.y + region.height;
			if ( inside ) continue;
			const t = ( ( layer * h + y ) * w + x ) * 4;
			for ( let c = 0; c < 4; c ++ ) if ( ! Object.is( chainA[ level - 1 ][ t + c ], chainB[ level - 1 ][ t + c ] ) ) outside.push( `level ${ level } (${ x }, ${ y })` );

		}

	} ) );
	return outside;

}

function data( width, height, layers, seed ) {

	let state = seed;
	return Float32Array.from( { length: width * height * layers * 4 }, () => ( state = ( Math.imul( state, 1664525 ) + 1013904223 ) >>> 0 ) / 2 ** 32 );

}

function modified( values, width, height, layers, rect, seed ) {

	const out = Float32Array.from( values );
	const fresh = data( width, height, layers, seed );
	for ( let layer = 0; layer < layers; layer ++ ) for ( let y = rect.y; y < rect.y + rect.height; y ++ ) for ( let x = rect.x; x < rect.x + rect.width; x ++ ) for ( let c = 0; c < 4; c ++ ) {

		const i = ( ( layer * height + y ) * width + x ) * 4 + c;
		out[ i ] = fresh[ i ];

	}

	return out;

}

test( 'regions are conservative against the CPU oracles (exact average/max, spd average)', () => {

	for ( const [ width, height, layers ] of [ [ 37, 23, 1 ], [ 129, 65, 2 ], [ 45, 91, 1 ] ] ) {

		const levels = mipLevelCountFor( width, height );
		const base = data( width, height, layers, 7 );
		for ( const rect of rectsFor( width, height, 3 ) ) {

			const changed = modified( base, width, height, layers, rect, 99 );
			for ( const reduction of [ 'average', 'max' ] ) {

				const chain = values => {

					const out = [];
					let previous = values, w = width, h = height;
					for ( let level = 1; level < levels; level ++ ) {

						const next = reduceReference( previous, w, h, layers, 'float32', reduction );
						out.push( next.values ); previous = next.values; w = next.width; h = next.height;

					}

					return out;

				};
				const batches = planBatches( width, height, { mipLevelCount: levels, levelsPerDispatch: 4 } );
				const regions = planRegions( batches, rect, width, height );
				assert.deepEqual( changedOutside( chain( base ), chain( changed ), regions, batches, width, height, layers ), [], `${ width }x${ height } ${ reduction } ${ JSON.stringify( rect ) }` );

			}

			for ( const levelsPerDispatch of [ 1, 4, 6 ] ) {

				const chain = values => spdChainReference( values, width, height, layers, 'float32', 'average', levels - 1, levelsPerDispatch ).map( level => level.values );
				const batches = planBatches( width, height, { mipLevelCount: levels, levelsPerDispatch, oddSizes: 'spd' } );
				const regions = planRegions( batches, rect, width, height );
				assert.deepEqual( changedOutside( chain( base ), chain( changed ), regions, batches, width, height, layers ), [], `${ width }x${ height } spd L${ levelsPerDispatch } ${ JSON.stringify( rect ) }` );

			}

		}

	}

} );

test( 'invalid rects and layer ranges are named', () => {

	assert.equal( rectProblem( { x: 0, y: 0, width: 8, height: 4 }, 8, 4 ), null );
	assert.match( rectProblem( { x: 0, y: 0, width: 0, height: 4 }, 8, 4 ), /at least 1×1/ );
	assert.match( rectProblem( { x: 1, y: 0, width: 8, height: 4 }, 8, 4 ), /outside the 8×4 level/ );
	assert.match( rectProblem( { x: - 1, y: 0, width: 2, height: 2 }, 8, 4 ), /outside/ );
	assert.match( rectProblem( { x: 0.5, y: 0, width: 2, height: 2 }, 8, 4 ), /integers/ );
	assert.match( rectProblem( null, 8, 4 ), /x, y, width, height/ );
	assert.equal( layersProblem( [ 1, 2 ], 3 ), null );
	assert.match( layersProblem( [ 2, 2 ], 3 ), /outside the texture's 3 layers/ );
	assert.match( layersProblem( [ 0, 0 ], 3 ), /at least 1/ );
	assert.match( layersProblem( [ 0 ], 3 ), /\[ first, count \]/ );
	const batches = planBatches( 16, 16, { mipLevelCount: 5, levelsPerDispatch: 4 } );
	assert.throws( () => planRegions( batches, { x: 0, y: 0, width: 17, height: 1 }, 16, 16 ), /outside the 16×16 level/ );

} );
