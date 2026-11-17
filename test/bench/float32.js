// `--compare=float32` (DIVERGES #13): 'average' chains of float32 formats, sampled against
// loaded. Sampling float32 needs the optional `float32-filterable` feature, which this mode
// requests (test/browser/index.html); the Downsampler samples whenever the device has it.
// - `load`: `averageAccess: 'load'`, four loads with weights in the shader (what every
//   float32 chain does on a device without the feature).
// - `sample`: the default, one bilinear sample per first-output texel.
// Ratios are paired per round against `load`.
import { Downsampler } from '/dist/webgpu/index.js';

export const float32Formats = [ 'r32float', 'rg32float', 'rgba32float' ];

export function float32Strategies( device ) {

	if ( ! device.features.has( 'float32-filterable' ) ) throw new Error( '--compare=float32 needs a device with float32-filterable; this adapter does not offer it.' );
	const run = downsampler => ( { passes: texture => downsampler.prepare( texture ).length, encode: ( encoder, texture ) => downsampler.encode( encoder, texture ) } );
	return {
		strategies: {
			load: { ...run( new Downsampler( device, { averageAccess: 'load' } ) ), parity: true },
			sample: { ...run( new Downsampler( device ) ), parity: true }
		},
		reference: () => 'load'
	};

}
