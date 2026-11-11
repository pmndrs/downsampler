// AMD FidelityFX SPD, ported to WGSL for the FSR 3.1 reference fixtures.
//
// Source: FidelityFX SDK, Kits/FidelityFX/upscalers/fsr3/include/gpu/spd/ffx_spd.h
// (FSR3UPSCALER 3.1.6, non-packed float path with FFX_SPD_NO_WAVE_OPERATIONS:
// the LDS reductions, not the quad intrinsics). Function names follow the
// original so the two can be read side by side.
//
// Differences from AMD, deliberate and listed in test/tools/README.md:
// - The atomic tail is a second dispatch (`spdTail`, one workgroup) instead of
//   "the last workgroup to finish continues". It runs the same code
//   (SpdDownsampleMips_6_7 + SpdDownsampleNextFour from mip 8) on the same data.
// - SpdLoadSourceImage's ClampLoad lives here (the `'clamp'` edge policy of
//   SPEC §14.1), so the tool's `load` hook sees in-bounds texels only.
// - The callbacks are the §14.1 hooks: `load( texel, layer )` for
//   SpdLoadSourceImage, `store( level, texel, layer, value )` for SpdStore with
//   level = mip index + 1, and `spdLoad( texel )` for SpdLoad (tail input).
//
// ---------------------------------------------------------------------------
// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// ---------------------------------------------------------------------------

/** ffxSpdSetup for a full-texture rect: groups, numWorkGroups and mips = min(⌊log2 max(w, h)⌋, 12). */
export function spdSetup( width, height ) {

	const groups = [ Math.floor( ( width - 1 ) / 64 ) + 1, Math.floor( ( height - 1 ) / 64 ) + 1 ];
	return { groups, numWorkGroups: groups[ 0 ] * groups[ 1 ], mips: Math.min( Math.floor( Math.log2( Math.max( width, height ) ) ), 12 ) };

}

/** cbSPD: mips, numWorkGroups, workGroupOffset, renderSize (32 bytes). */
export function spdConstants( { mips, numWorkGroups }, width, height ) {

	return new Uint32Array( [ mips, numWorkGroups, 0, 0, width, height, 0, 0 ] );

}

/**
 * The SPD skeleton in @group(0). The tool's WGSL (in @group(1)) must define
 * `load`, `store` and `spdLoad`.
 */
export const spdWgsl = /* wgsl */ `
struct SpdConstants {
	mips: u32,
	numWorkGroups: u32,
	workGroupOffset: vec2u,
	renderSize: vec2u,
}
@group(0) @binding(0) var<uniform> spd: SpdConstants;

var<workgroup> spdIntermediate: array<vec4f, 256>; // [16][16], indexed [x][y] as in ffx_spd.h

// ffxRemapForWaveReduction (ffx_core): a 64-thread index to an 8×8 position
// such that threads 0..15 cover 4×4 and threads 0..3 cover 2×2.
fn ffxRemapForWaveReduction(a: u32) -> vec2u {
	return vec2u((((a >> 2u) & 7u) & ~1u) | (a & 1u), (((a >> 3u) & 7u) & ~3u) | ((a >> 1u) & 3u));
}

fn SpdLoadIntermediate(x: u32, y: u32) -> vec4f { return spdIntermediate[x * 16u + y]; }
fn SpdStoreIntermediate(x: u32, y: u32, value: vec4f) { spdIntermediate[x * 16u + y] = value; }

// Both FSR 3.1 SPD passes define SpdReduce4 as the average.
fn SpdReduce4(v0: vec4f, v1: vec4f, v2: vec4f, v3: vec4f) -> vec4f { return (v0 + v1 + v2 + v3) * 0.25; }

// SpdLoadSourceImage: the tool's load, behind FSR's ClampLoad(p, 0, RenderSize()).
fn SpdLoadSourceImage(p: vec2i) -> vec4f {
	let size = vec2i(spd.renderSize);
	return load(clamp(p, vec2i(0), size - 1), 0u);
}
fn SpdLoad(p: vec2i) -> vec4f { return spdLoad(p); }
fn SpdStore(pix: vec2i, value: vec4f, index: u32) { store(index + 1u, pix, 0u, value); }

fn SpdReduceIntermediate(i0: vec2u, i1: vec2u, i2: vec2u, i3: vec2u) -> vec4f {
	let v0 = SpdLoadIntermediate(i0.x, i0.y);
	let v1 = SpdLoadIntermediate(i1.x, i1.y);
	let v2 = SpdLoadIntermediate(i2.x, i2.y);
	let v3 = SpdLoadIntermediate(i3.x, i3.y);
	return SpdReduce4(v0, v1, v2, v3);
}

fn SpdReduceLoad4(base: vec2u) -> vec4f {
	let v0 = SpdLoad(vec2i(base + vec2u(0u, 0u)));
	let v1 = SpdLoad(vec2i(base + vec2u(0u, 1u)));
	let v2 = SpdLoad(vec2i(base + vec2u(1u, 0u)));
	let v3 = SpdLoad(vec2i(base + vec2u(1u, 1u)));
	return SpdReduce4(v0, v1, v2, v3);
}

fn SpdReduceLoadSourceImage(base: vec2u) -> vec4f {
	let v0 = SpdLoadSourceImage(vec2i(base + vec2u(0u, 0u)));
	let v1 = SpdLoadSourceImage(vec2i(base + vec2u(0u, 1u)));
	let v2 = SpdLoadSourceImage(vec2i(base + vec2u(1u, 0u)));
	let v3 = SpdLoadSourceImage(vec2i(base + vec2u(1u, 1u)));
	return SpdReduce4(v0, v1, v2, v3);
}

fn SpdDownsampleMips_0_1_LDS(x: u32, y: u32, workGroupID: vec2u, localInvocationIndex: u32, mip: u32) {
	var v: array<vec4f, 4>;

	var tex = workGroupID * 64u + vec2u(x * 2u, y * 2u);
	var pix = vec2i(workGroupID * 32u + vec2u(x, y));
	v[0] = SpdReduceLoadSourceImage(tex);
	SpdStore(pix, v[0], 0u);

	tex = workGroupID * 64u + vec2u(x * 2u + 32u, y * 2u);
	pix = vec2i(workGroupID * 32u + vec2u(x + 16u, y));
	v[1] = SpdReduceLoadSourceImage(tex);
	SpdStore(pix, v[1], 0u);

	tex = workGroupID * 64u + vec2u(x * 2u, y * 2u + 32u);
	pix = vec2i(workGroupID * 32u + vec2u(x, y + 16u));
	v[2] = SpdReduceLoadSourceImage(tex);
	SpdStore(pix, v[2], 0u);

	tex = workGroupID * 64u + vec2u(x * 2u + 32u, y * 2u + 32u);
	pix = vec2i(workGroupID * 32u + vec2u(x + 16u, y + 16u));
	v[3] = SpdReduceLoadSourceImage(tex);
	SpdStore(pix, v[3], 0u);

	if (mip <= 1u) { return; }

	for (var i = 0u; i < 4u; i++) {
		SpdStoreIntermediate(x, y, v[i]);
		workgroupBarrier();
		if (localInvocationIndex < 64u) {
			v[i] = SpdReduceIntermediate(vec2u(x * 2u + 0u, y * 2u + 0u), vec2u(x * 2u + 1u, y * 2u + 0u), vec2u(x * 2u + 0u, y * 2u + 1u), vec2u(x * 2u + 1u, y * 2u + 1u));
			SpdStore(vec2i(workGroupID * 16u + vec2u(x + (i % 2u) * 8u, y + (i / 2u) * 8u)), v[i], 1u);
		}
		workgroupBarrier();
	}

	if (localInvocationIndex < 64u) {
		SpdStoreIntermediate(x + 0u, y + 0u, v[0]);
		SpdStoreIntermediate(x + 8u, y + 0u, v[1]);
		SpdStoreIntermediate(x + 0u, y + 8u, v[2]);
		SpdStoreIntermediate(x + 8u, y + 8u, v[3]);
	}
}

fn SpdDownsampleMip_2(x: u32, y: u32, workGroupID: vec2u, localInvocationIndex: u32, mip: u32) {
	if (localInvocationIndex < 64u) {
		let v = SpdReduceIntermediate(vec2u(x * 2u + 0u, y * 2u + 0u), vec2u(x * 2u + 1u, y * 2u + 0u), vec2u(x * 2u + 0u, y * 2u + 1u), vec2u(x * 2u + 1u, y * 2u + 1u));
		SpdStore(vec2i(workGroupID * 8u + vec2u(x, y)), v, mip);
		// store to LDS, try to reduce bank conflicts
		SpdStoreIntermediate(x * 2u + y % 2u, y * 2u, v);
	}
}

fn SpdDownsampleMip_3(x: u32, y: u32, workGroupID: vec2u, localInvocationIndex: u32, mip: u32) {
	if (localInvocationIndex < 16u) {
		let v = SpdReduceIntermediate(vec2u(x * 4u + 0u + 0u, y * 4u + 0u), vec2u(x * 4u + 2u + 0u, y * 4u + 0u), vec2u(x * 4u + 0u + 1u, y * 4u + 2u), vec2u(x * 4u + 2u + 1u, y * 4u + 2u));
		SpdStore(vec2i(workGroupID * 4u + vec2u(x, y)), v, mip);
		SpdStoreIntermediate(x * 4u + y, y * 4u, v);
	}
}

fn SpdDownsampleMip_4(x: u32, y: u32, workGroupID: vec2u, localInvocationIndex: u32, mip: u32) {
	if (localInvocationIndex < 4u) {
		let v = SpdReduceIntermediate(vec2u(x * 8u + 0u + 0u + y * 2u, y * 8u + 0u),
			vec2u(x * 8u + 4u + 0u + y * 2u, y * 8u + 0u),
			vec2u(x * 8u + 0u + 1u + y * 2u, y * 8u + 4u),
			vec2u(x * 8u + 4u + 1u + y * 2u, y * 8u + 4u));
		SpdStore(vec2i(workGroupID * 2u + vec2u(x, y)), v, mip);
		SpdStoreIntermediate(x + y * 2u, 0u, v);
	}
}

fn SpdDownsampleMip_5(workGroupID: vec2u, localInvocationIndex: u32, mip: u32) {
	if (localInvocationIndex < 1u) {
		let v = SpdReduceIntermediate(vec2u(0u, 0u), vec2u(1u, 0u), vec2u(2u, 0u), vec2u(3u, 0u));
		SpdStore(vec2i(workGroupID), v, mip);
	}
}

fn SpdDownsampleMips_6_7(x: u32, y: u32, mips: u32) {
	var tex = vec2u(x * 4u + 0u, y * 4u + 0u);
	var pix = vec2i(vec2u(x * 2u + 0u, y * 2u + 0u));
	let v0 = SpdReduceLoad4(tex);
	SpdStore(pix, v0, 6u);

	tex = vec2u(x * 4u + 2u, y * 4u + 0u);
	pix = vec2i(vec2u(x * 2u + 1u, y * 2u + 0u));
	let v1 = SpdReduceLoad4(tex);
	SpdStore(pix, v1, 6u);

	tex = vec2u(x * 4u + 0u, y * 4u + 2u);
	pix = vec2i(vec2u(x * 2u + 0u, y * 2u + 1u));
	let v2 = SpdReduceLoad4(tex);
	SpdStore(pix, v2, 6u);

	tex = vec2u(x * 4u + 2u, y * 4u + 2u);
	pix = vec2i(vec2u(x * 2u + 1u, y * 2u + 1u));
	let v3 = SpdReduceLoad4(tex);
	SpdStore(pix, v3, 6u);

	if (mips <= 7u) { return; }
	// no barrier needed, working on values only from the same thread

	let v = SpdReduce4(v0, v1, v2, v3);
	SpdStore(vec2i(vec2u(x, y)), v, 7u);
	SpdStoreIntermediate(x, y, v);
}

fn SpdDownsampleNextFour(x: u32, y: u32, workGroupID: vec2u, localInvocationIndex: u32, baseMip: u32, mips: u32) {
	if (mips <= baseMip) { return; }
	workgroupBarrier();
	SpdDownsampleMip_2(x, y, workGroupID, localInvocationIndex, baseMip);

	if (mips <= baseMip + 1u) { return; }
	workgroupBarrier();
	SpdDownsampleMip_3(x, y, workGroupID, localInvocationIndex, baseMip + 1u);

	if (mips <= baseMip + 2u) { return; }
	workgroupBarrier();
	SpdDownsampleMip_4(x, y, workGroupID, localInvocationIndex, baseMip + 2u);

	if (mips <= baseMip + 3u) { return; }
	workgroupBarrier();
	SpdDownsampleMip_5(workGroupID, localInvocationIndex, baseMip + 3u);
}

fn spdThread(localInvocationIndex: u32) -> vec2u {
	let sub_xy = ffxRemapForWaveReduction(localInvocationIndex % 64u);
	return vec2u(sub_xy.x + 8u * ((localInvocationIndex >> 6u) % 2u), sub_xy.y + 8u * (localInvocationIndex >> 7u));
}

// SpdDownsample up to the atomic counter: mips 1-6 of one 64×64 tile.
@compute @workgroup_size(256)
fn spdTile(@builtin(workgroup_id) workGroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
	let workGroupID = workGroupId.xy + spd.workGroupOffset;
	let xy = spdThread(localInvocationIndex);
	let mips = spd.mips;
	SpdDownsampleMips_0_1_LDS(xy.x, xy.y, workGroupID, localInvocationIndex, mips);
	SpdDownsampleNextFour(xy.x, xy.y, workGroupID, localInvocationIndex, 2u, mips);
}

// SpdDownsample after the atomic counter, as its own one-workgroup dispatch:
// mips 7-12 from mip 6 (read back through SpdLoad).
@compute @workgroup_size(256)
fn spdTail(@builtin(local_invocation_index) localInvocationIndex: u32) {
	let mips = spd.mips;
	if (mips <= 6u) { return; }
	let xy = spdThread(localInvocationIndex);
	SpdDownsampleMips_6_7(xy.x, xy.y, mips);
	SpdDownsampleNextFour(xy.x, xy.y, vec2u(0u, 0u), localInvocationIndex, 8u, mips);
}
`;

const f32 = Math.fround;

/** SpdReduce4 in float32, adding in the order the kernel does. */
function reduce4( out, o, a, b, c, d ) {

	for ( let k = 0; k < 4; k ++ ) out[ o + k ] = f32( f32( f32( f32( a[ k ] + b[ k ] ) + c[ k ] ) + d[ k ] ) * 0.25 );

}

/**
 * CPU SPD with AMD's semantics, for the oracles. `source` holds the tool's
 * load for every in-bounds texel (4 floats each); out-of-bounds reads clamp
 * (ClampLoad). Every texel the kernel computes is passed to `store( level, x,
 * y, value )`, including texels outside the image, as on the GPU. Mips above 6
 * read back through `spdLoad( x, y )` (the tail). Returns the float32 levels.
 */
export function spdReference( { width, height, source, mips, store = () => {}, spdLoad = null } ) {

	const { groups } = spdSetup( width, height );
	const levels = [];
	const a = [ 0, 0, 0, 0 ], b = [ 0, 0, 0, 0 ], c = [ 0, 0, 0, 0 ], d = [ 0, 0, 0, 0 ], value = [ 0, 0, 0, 0 ];
	const read = ( data, w, x, y, out ) => { const i = ( y * w + x ) * 4; out[ 0 ] = data[ i ]; out[ 1 ] = data[ i + 1 ]; out[ 2 ] = data[ i + 2 ]; out[ 3 ] = data[ i + 3 ]; };
	const sourceAt = ( x, y, out ) => read( source, width, Math.min( x, width - 1 ), Math.min( y, height - 1 ), out );

	// Mip 1 over the whole tile grid: (0,0), (0,1), (1,0), (1,1).
	let w = groups[ 0 ] * 32, h = groups[ 1 ] * 32;
	let data = new Float32Array( w * h * 4 );
	for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

		sourceAt( 2 * x, 2 * y, a ); sourceAt( 2 * x, 2 * y + 1, b ); sourceAt( 2 * x + 1, 2 * y, c ); sourceAt( 2 * x + 1, 2 * y + 1, d );
		reduce4( data, ( y * w + x ) * 4, a, b, c, d );
		read( data, w, x, y, value ); store( 1, x, y, value );

	}

	levels[ 1 ] = { width: w, height: h, data };

	// Mips 2-6 inside each tile, then 8-12 in the tail: (0,0), (1,0), (0,1), (1,1).
	// Mip 7 reads mip 6 back through SpdLoad: (0,0), (0,1), (1,0), (1,1).
	for ( let level = 2; level <= mips; level ++ ) {

		const tail = level === 7;
		if ( tail ) {

			if ( ! spdLoad ) break;
			// The tail workgroup covers 64×64 mip-6 texels.
			const next = new Float32Array( 32 * 32 * 4 );
			for ( let y = 0; y < 32; y ++ ) for ( let x = 0; x < 32; x ++ ) {

				spdLoad( 2 * x, 2 * y, a ); spdLoad( 2 * x, 2 * y + 1, b ); spdLoad( 2 * x + 1, 2 * y, c ); spdLoad( 2 * x + 1, 2 * y + 1, d );
				reduce4( next, ( y * 32 + x ) * 4, a, b, c, d );
				read( next, 32, x, y, value ); store( 7, x, y, value );

			}

			levels[ 7 ] = { width: 32, height: 32, data: next };
			w = 32; h = 32; data = next;
			continue;

		}

		const nw = w >> 1, nh = h >> 1, next = new Float32Array( Math.max( 1, nw * nh ) * 4 );
		for ( let y = 0; y < nh; y ++ ) for ( let x = 0; x < nw; x ++ ) {

			read( data, w, 2 * x, 2 * y, a ); read( data, w, 2 * x + 1, 2 * y, b ); read( data, w, 2 * x, 2 * y + 1, c ); read( data, w, 2 * x + 1, 2 * y + 1, d );
			reduce4( next, ( y * nw + x ) * 4, a, b, c, d );
			read( next, nw, x, y, value ); store( level, x, y, value );

		}

		levels[ level ] = { width: nw, height: nh, data: next };
		w = nw; h = nh; data = next;

	}

	return levels;

}
