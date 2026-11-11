// Synthetic but realistic upscaler inputs, generated on the CPU from a seed.
//
// A static camera looks at a sky (with a sun) over a textured ground plane.
// Six objects move between the previous frame (t = 0) and the current one
// (t = 1), one of them a small emissive light; a ground region is lit 2.5×
// brighter in the current frame (a genuine shading change); a 1.3 px checker
// aliases under the jitter. Colour is HDR (luma 0.05 to ~100) and every input
// is rounded to the format it is uploaded in, so the oracles read exactly what
// the GPU reads.
//
// Velocities and jitters are chosen so that no reprojected sample position
// falls within 0.02 px of a texel boundary (floor() is a discrete decision in
// both shading-change passes; motion vectors are float16).
import { f16, f32 } from './half.js';

export const defaultSizes = [ [ 960, 540 ], [ 1280, 720 ], [ 1129, 635 ], [ 1707, 960 ], [ 1920, 1080 ] ];

// Render-pixel jitters, as the FSR constants take them: texel i samples the scene at i + 0.5 + jitter.
export const jitter = [ - 0.375, - 0.1111 ];
export const previousJitter = [ 0.125, 0.2222 ];
export const deltaTime = 1 / 60;

const objects = [
	{ shape: 'disc', centre: [ 0.30, 0.62 ], radius: 0.09, velocity: [ 2.25, - 1.25 ], colour: [ 2.4, 1.1, 0.5 ], depth: 6 },
	{ shape: 'disc', centre: [ 0.95, 0.70 ], radius: 0.13, velocity: [ - 1.25, 0.25 ], colour: [ 0.4, 0.9, 1.6 ], depth: 9 },
	{ shape: 'box', centre: [ 1.30, 0.55 ], half: [ 0.10, 0.06 ], velocity: [ 3.25, 1.25 ], colour: [ 1.5, 1.5, 1.4 ], depth: 4, stripes: true },
	{ shape: 'disc', centre: [ 0.55, 0.30 ], radius: 0.012, velocity: [ 1.25, 2.25 ], colour: [ 120, 90, 60 ], depth: 20 },
	{ shape: 'box', centre: [ 1.55, 0.82 ], half: [ 0.05, 0.12 ], velocity: [ - 2.25, - 1.25 ], colour: [ 0.2, 0.2, 0.25 ], depth: 3 },
	{ shape: 'disc', centre: [ 1.10, 0.25 ], radius: 0.05, velocity: [ 0.25, 1.25 ], colour: [ 6, 5, 3 ], depth: 30 }
];

function hash( x, y, z ) {

	let h = ( Math.imul( x | 0, 374761393 ) + Math.imul( y | 0, 668265263 ) + Math.imul( z | 0, 1274126177 ) ) | 0;
	h = Math.imul( h ^ ( h >>> 13 ), 1274126177 );
	return ( ( h ^ ( h >>> 16 ) ) >>> 0 ) / 4294967296;

}

function valueNoise( x, y, seed ) {

	const ix = Math.floor( x ), iy = Math.floor( y ), fx = x - ix, fy = y - iy;
	const sx = fx * fx * ( 3 - 2 * fx ), sy = fy * fy * ( 3 - 2 * fy );
	const a = hash( ix, iy, seed ), b = hash( ix + 1, iy, seed ), c = hash( ix, iy + 1, seed ), d = hash( ix + 1, iy + 1, seed );
	return ( a + ( b - a ) * sx ) + ( ( c + ( d - c ) * sx ) - ( a + ( b - a ) * sx ) ) * sy;

}

/** Index of the nearest object covering scene point (sx, sy) at time t, or -1. */
function hit( sx, sy, t, height ) {

	let best = - 1, depth = Infinity;
	for ( let i = 0; i < objects.length; i ++ ) {

		const object = objects[ i ];
		const cx = object.centre[ 0 ] + object.velocity[ 0 ] * t / height, cy = object.centre[ 1 ] + object.velocity[ 1 ] * t / height;
		const dx = sx - cx, dy = sy - cy;
		const inside = object.shape === 'disc' ? dx * dx + dy * dy < object.radius * object.radius : Math.abs( dx ) < object.half[ 0 ] && Math.abs( dy ) < object.half[ 1 ];
		if ( inside && object.depth < depth ) { best = i; depth = object.depth; }

	}

	return best;

}

function shade( px, py, t, frame, width, height, seed, out ) {

	const sx = px / height, sy = py / height, aspect = width / height;
	const id = hit( sx, sy, t, height );
	let r, g, b, depth;

	if ( id >= 0 ) {

		const object = objects[ id ];
		[ r, g, b ] = object.colour;
		if ( object.stripes ) {

			const local = sx - ( object.centre[ 0 ] + object.velocity[ 0 ] * t / height );
			const stripe = 0.6 + 0.4 * Math.sin( local * 60 );
			r *= stripe; g *= stripe; b *= stripe;

		}

		depth = object.depth;

	} else if ( sy < 0.4 ) {

		const base = 1.5 + 4 * ( 1 - sy / 0.4 );
		[ r, g, b ] = [ base * 0.6, base * 0.8, base * 1.2 ];
		const dx = sx - 0.2 * aspect, dy = sy - 0.12;
		if ( dx * dx + dy * dy < 0.03 * 0.03 ) [ r, g, b ] = [ 60, 55, 45 ];
		depth = 1000;

	} else {

		depth = 1.5 / ( sy - 0.4 + 0.004 );
		const texture = valueNoise( ( sx - aspect / 2 ) * depth * 2, depth * 2, seed );
		const albedo = 0.15 + 0.8 * texture;
		[ r, g, b ] = [ albedo * 1.6, albedo * 1.44, albedo * 1.2 ];
		// Sub-Nyquist checker: aliases under the jitter.
		if ( sx > 0.05 * aspect && sx < 0.3 * aspect && sy > 0.75 && sy < 0.95 ) {

			const on = ( Math.floor( px / 1.3 ) + Math.floor( py / 1.3 ) ) & 1;
			const value = on ? 1.8 : 0.15;
			[ r, g, b ] = [ value, value, value * 0.9 ];

		}

		// A light switched on between the frames.
		if ( t === 1 && sx > 0.55 * aspect && sx < 0.8 * aspect && sy > 0.45 && sy < 0.7 ) { r *= 2.5; g *= 2.5; b *= 2.5; }

	}

	// Render noise, ±3 %, different each frame.
	const noise = 1 + 0.06 * ( hash( Math.floor( px ), Math.floor( py ), seed * 7 + frame ) - 0.5 );
	out[ 0 ] = r * noise; out[ 1 ] = g * noise; out[ 2 ] = b * noise; out[ 3 ] = depth;
	return id;

}

/** RGBToLuma(max(0, rgb)) in float32, as FSR's prepare-inputs pass computes it. */
export function luma( r, g, b ) {

	return f32( f32( f32( Math.max( 0, r ) * f32( 0.2126 ) ) + f32( Math.max( 0, g ) * f32( 0.7152 ) ) ) + f32( Math.max( 0, b ) * f32( 0.0722 ) ) );

}

/**
 * Every input both upscalers' pyramid passes read, at render size width × height.
 * Float arrays hold the values as uploaded (float16-rounded where the texture is float16).
 */
export function createScene( width, height, { seed = 1 } = {} ) {

	const count = width * height;
	const colour = new Float32Array( count * 4 ), previousColour = new Float32Array( count * 4 );
	const depth = new Float32Array( count ), ids = new Int8Array( count );
	const sample = [ 0, 0, 0, 0 ];

	for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

		const i = y * width + x;
		ids[ i ] = shade( x + 0.5 + jitter[ 0 ], y + 0.5 + jitter[ 1 ], 1, 1, width, height, seed, sample );
		for ( let c = 0; c < 3; c ++ ) colour[ i * 4 + c ] = f16( sample[ c ] );
		colour[ i * 4 + 3 ] = 1;
		depth[ i ] = sample[ 3 ];
		shade( x + 0.5 + previousJitter[ 0 ], y + 0.5 + previousJitter[ 1 ], 0, 0, width, height, seed, sample );
		for ( let c = 0; c < 3; c ++ ) previousColour[ i * 4 + c ] = f16( sample[ c ] );
		previousColour[ i * 4 + 3 ] = 1;

	}

	// Per-pixel velocity (render px per frame) and disocclusion.
	const velocity = new Float32Array( count * 2 ), disocclusion = new Uint8Array( count );
	for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

		const i = y * width + x, id = ids[ i ];
		if ( id < 0 ) continue;
		const [ vx, vy ] = objects[ id ].velocity;
		velocity[ i * 2 ] = vx; velocity[ i * 2 + 1 ] = vy;
		const before = hit( ( x + 0.5 + previousJitter[ 0 ] - vx ) / height, ( y + 0.5 + previousJitter[ 1 ] - vy ) / height, 0, height );
		disocclusion[ i ] = before !== id ? 1 : 0;

	}

	// Dilate: the motion of the nearest-depth texel of the 3×3 neighbourhood
	// (FSR FindDepthExtents order; a sample replaces the nearest only if strictly nearer).
	const offsets = [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 0, - 1 ], [ - 1, 0 ], [ - 1, 1 ], [ 1, 1 ], [ - 1, - 1 ], [ 1, - 1 ] ];
	const fsrMotion = new Float32Array( count * 2 ), upscalerMotion = new Float32Array( count * 4 );
	for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

		let nearest = depth[ y * width + x ], nx = x, ny = y;
		for ( let s = 1; s < 9; s ++ ) {

			const sx = x + offsets[ s ][ 0 ], sy = y + offsets[ s ][ 1 ];
			if ( sx < 0 || sy < 0 || sx >= width || sy >= height ) continue;
			if ( depth[ sy * width + sx ] < nearest ) { nearest = depth[ sy * width + sx ]; nx = sx; ny = sy; }

		}

		const j = ny * width + nx, i = y * width + x;
		// FSR: previous uv = uv + mv. The upscaler: previous uv = uv - motion.
		fsrMotion[ i * 2 ] = f16( - velocity[ j * 2 ] / width ); fsrMotion[ i * 2 + 1 ] = f16( - velocity[ j * 2 + 1 ] / height );
		upscalerMotion[ i * 4 ] = f16( velocity[ j * 2 ] / width ); upscalerMotion[ i * 4 + 1 ] = f16( velocity[ j * 2 + 1 ] / height );

	}

	const currentLuma = new Float32Array( count ), previousLuma = new Float32Array( count );
	const hostLuma = new Float32Array( count ), lumaHistory = new Float32Array( count );
	const farthestDepth = new Float32Array( count ), masks = new Uint8Array( count * 4 );
	for ( let i = 0; i < count; i ++ ) {

		const current = luma( colour[ i * 4 ], colour[ i * 4 + 1 ], colour[ i * 4 + 2 ] );
		const previous = luma( previousColour[ i * 4 ], previousColour[ i * 4 + 1 ], previousColour[ i * 4 + 2 ] );
		currentLuma[ i ] = f16( current ); previousLuma[ i ] = f16( previous );
		hostLuma[ i ] = current; lumaHistory[ i ] = previous;
		// FindDepthExtents' "farthest" only grows from a sample that is also nearer
		// than the nearest so far, so it always equals the centre depth (non-inverted).
		farthestDepth[ i ] = f16( Math.min( depth[ i ], 65504 ) );
		masks[ i * 4 ] = disocclusion[ i ] * 255; masks[ i * 4 + 3 ] = 255;

	}

	return {
		width, height, seed, jitter, previousJitter, deltaTime,
		colour, previousColour, currentLuma, previousLuma, farthestDepth, fsrMotion,
		hostLuma, lumaHistory, upscalerMotion, masks,
		stats: { disoccluded: disocclusion.reduce( ( a, b ) => a + b, 0 ) / count, moving: ids.reduce( ( a, id ) => a + ( id >= 0 ), 0 ) / count }
	};

}
