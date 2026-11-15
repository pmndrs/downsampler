// `--compare=partial`: what a partial update costs against regenerating the whole chain.
// - `full`: the default chain (the reference for the ratios).
// - `rect32`, `rect256`: a 32×32 or 256×256 rectangle of level 0 changed, away from the
//   edges; each level rewrites only the texels that depend on it (`rect`).
// - `layer`: one layer of an array texture (`layers: [ 2, 1 ]`), for cube maps; on a
//   single-layer texture it is the full chain.
// - `layerRect32`: a 32×32 rectangle on that one layer.
// Ratios are paired per round against `full`.
import { Downsampler } from '/dist/webgpu/index.js';

export const partialSizes = [ [ 1920, 1080 ], [ 2560, 1440 ], [ 1024, 1024 ], [ 4096, 4096 ], [ 512, 512, 6 ], [ 1024, 1024, 6 ] ];

const rect = ( texture, size ) => ( { x: Math.floor( texture.width / 3 ), y: Math.floor( texture.height / 3 ), width: Math.min( size, texture.width ), height: Math.min( size, texture.height ) } );

export function partialStrategies( device ) {

	const downsampler = new Downsampler( device );
	const run = options => ( {
		passes: texture => downsampler.prepare( texture, options( texture ) ).length,
		encode: ( encoder, texture ) => downsampler.encode( encoder, texture, options( texture ) ),
		parity: false,
	} );
	const layer = texture => texture.depthOrArrayLayers > 1 ? { layers: [ 2, 1 ] } : {};
	const strategies = {
		full: { ...run( () => ( {} ) ), parity: true },
		rect32: run( texture => ( { rect: rect( texture, 32 ) } ) ),
		rect256: run( texture => ( { rect: rect( texture, 256 ) } ) ),
		layer: run( layer ),
		layerRect32: run( texture => ( { ...layer( texture ), rect: rect( texture, 32 ) } ) ),
	};
	return { strategies, reference: () => 'full' };

}
