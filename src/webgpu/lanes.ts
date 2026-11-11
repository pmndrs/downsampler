/**
 * WGSL for accumulator values of 4 or 8 channels (SPEC.md §14.1 Wide
 * accumulators). Four channels are a `vec4f`, exactly as before; eight are
 * the struct `DsWide { lo: vec4f, hi: vec4f }` (channels 0–3 in `lo`, 4–7 in
 * `hi`), which hooks take and return in place of `vec4f`.
 */
import type { Channels } from '../core/reduction.js';

const xyzw = [ 'x', 'y', 'z', 'w' ];

/** The 8-channel value type, declared by kernels with `channels: 8`. */
export const wideStruct = 'struct DsWide {\n\tlo: vec4f,\n\thi: vec4f,\n}\n';

export interface Lanes {
	channels: Channels;
	/** The value type: `vec4f` or `DsWide`. */
	type: string;
	/** Channel `c` of the value expression `v`. */
	channel( v: string, c: number ): string;
	/** A value from one scalar expression per channel. */
	build( exprs: readonly string[] ): string;
	/** A value from one `vec4f` expression per half (`lo`, `hi`); for 4 channels the one `vec4f`. */
	halves( v: string ): string[];
	/** Rebuilds a value from per-half `vec4f` expressions. */
	join( halves: readonly string[] ): string;
	/** Declarations the kernel needs for the type. */
	declarations: string;
}

export function lanes( channels: Channels ): Lanes {

	if ( channels === 4 ) return {
		channels, type: 'vec4f', declarations: '',
		channel: ( v, c ) => `${ v }.${ xyzw[ c ] }`,
		build: exprs => `vec4f(${ exprs.join( ', ' ) })`,
		halves: v => [ v ],
		join: halves => halves[ 0 ]
	};

	return {
		channels, type: 'DsWide', declarations: wideStruct,
		channel: ( v, c ) => `${ v }.${ c < 4 ? 'lo' : 'hi' }.${ xyzw[ c % 4 ] }`,
		build: exprs => `DsWide(vec4f(${ exprs.slice( 0, 4 ).join( ', ' ) }), vec4f(${ exprs.slice( 4 ).join( ', ' ) }))`,
		halves: v => [ `${ v }.lo`, `${ v }.hi` ],
		join: halves => `DsWide(${ halves.join( ', ' ) })`
	};

}
