// float16 and float32 rounding for the CPU oracles. Browser and Node.

const scratch = new Float32Array( 1 );
const bits = new Uint32Array( scratch.buffer );

/** float32 → float16 bits, round to nearest even (what writeTexture data and most GPUs produce). */
export function toHalfBits( value ) {

	scratch[ 0 ] = value;
	const x = bits[ 0 ];
	const sign = ( x >>> 16 ) & 0x8000;
	const exponent = ( x >>> 23 ) & 0xff;
	let mantissa = x & 0x7fffff;
	if ( exponent === 0xff ) return sign | 0x7c00 | ( mantissa ? 0x200 : 0 );
	const e = exponent - 127 + 15;
	if ( e >= 0x1f ) return sign | 0x7c00;
	if ( e <= 0 ) {

		if ( e < - 10 ) return sign;
		mantissa |= 0x800000;
		const shift = 14 - e;
		let half = mantissa >>> shift;
		const rest = mantissa & ( ( 1 << shift ) - 1 ), midpoint = 1 << ( shift - 1 );
		if ( rest > midpoint || ( rest === midpoint && ( half & 1 ) ) ) half ++;
		return sign | half;

	}

	let half = ( e << 10 ) | ( mantissa >>> 13 );
	const rest = mantissa & 0x1fff;
	if ( rest > 0x1000 || ( rest === 0x1000 && ( half & 1 ) ) ) half ++;
	return sign | half;

}

export function fromHalfBits( half ) {

	const sign = half & 0x8000 ? - 1 : 1;
	const exponent = ( half >>> 10 ) & 0x1f, mantissa = half & 0x3ff;
	if ( exponent === 0 ) return sign * mantissa * 2 ** - 24;
	if ( exponent === 0x1f ) return mantissa ? NaN : sign * Infinity;
	return sign * ( 1 + mantissa / 1024 ) * 2 ** ( exponent - 15 );

}

/** Round to the nearest float16 value. */
export const f16 = Math.f16round ?? ( value => fromHalfBits( toHalfBits( value ) ) );

/** Round to the nearest float32 value. */
export const f32 = Math.fround;

/** One float16 ulp at `value` (the spacing of float16 numbers there). */
export function halfUlp( value ) {

	const magnitude = Math.abs( value );
	if ( magnitude < 2 ** - 14 ) return 2 ** - 24;
	return 2 ** ( Math.floor( Math.log2( magnitude ) ) - 10 );

}

/** Float16 bits of a typed float array, for upload. */
export function halfArray( floats ) {

	const output = new Uint16Array( floats.length );
	for ( let i = 0; i < floats.length; i ++ ) output[ i ] = toHalfBits( floats[ i ] );
	return output;

}
