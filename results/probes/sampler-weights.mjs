// Probe: the bilinear weights a GPU's sampler really uses. A 2×1 texture holding 0 and 1 is
// sampled at N positions between the two texel centres in a compute shader (f32 results to a
// buffer, so nothing is stored at reduced precision); each result is the hardware's weight for
// texel 1. Reports, per format: distinct weights (256 ⇒ 8-bit), the largest error against the
// exact weight, and whether weights look truncated or rounded. Also samples the coordinates the
// library's 'exact' odd-size reduction uses (15 → 7, 65 → 32) and reports the error there.
// Run: node results/probes/sampler-weights.mjs [--power=high|low] [--swiftshader]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';

const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const swift = process.argv.includes( '--swiftshader' );
const power = process.argv.find( arg => arg.startsWith( '--power=' ) )?.split( '=' )[ 1 ];
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', ...( swift ? [ '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' ] : gpuChoiceArgs( power ) ) ] } );
const page = await browser.newPage();
await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
console.log( JSON.stringify( await page.evaluate( async () => {

	const adapter = await navigator.gpu.requestAdapter();
	const features = [ 'float32-filterable' ].filter( name => adapter.features.has( name ) );
	const device = await adapter.requestDevice( { requiredFeatures: features } );
	const N = 1024;
	// Texel centres of a 2-wide texture are at u = 0.25 and 0.75; sample u = 0.25 + t / 2, t in [0, 1].
	const fractions = Array.from( { length: N + 1 }, ( _, i ) => i / N );
	const module = device.createShaderModule( { code: `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var linear: sampler;
@group(0) @binding(2) var<storage, read> us: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
	if (id.x >= arrayLength(&us)) { return; }
	out[id.x] = textureSampleLevel(image, linear, vec2f(us[id.x], 0.5), 0.0).r;
}` } );
	const pipeline = device.createComputePipeline( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
	const sampler = device.createSampler( { magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge' } );

	async function sample( format, us, pair = 'unit' ) {

		const texture = device.createTexture( { size: [ 2, 1 ], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
		const upload = {
			rgba8unorm: () => pair === 'unit' ? new Uint8Array( [ 0, 0, 0, 255, 255, 0, 0, 255 ] ) : new Uint8Array( [ 51, 0, 0, 255, 230, 0, 0, 255 ] ),
			// 'wide': 4 and 5.5 (f16 0x4400, 0x4580), where an f16 ulp is 1/256. 'skew': 0.25 and 40 (0x3400, 0x5100).
			rgba16float: () => new Uint16Array( { unit: [ 0, 0, 0, 0x3c00, 0x3c00, 0, 0, 0x3c00 ], wide: [ 0x4400, 0, 0, 0x3c00, 0x4580, 0, 0, 0x3c00 ], skew: [ 0x3400, 0, 0, 0x3c00, 0x5100, 0, 0, 0x3c00 ] }[ pair ] ),
			rgba32float: () => new Float32Array( { unit: [ 0, 0, 0, 1, 1, 0, 0, 1 ], wide: [ 4, 0, 0, 1, 5.5, 0, 0, 1 ], skew: [ 0.25, 0, 0, 1, 40, 0, 0, 1 ] }[ pair ] ),
			// rg11b10ufloat: r is an 11-bit float (5-bit exponent, 6-bit mantissa); 1.0 = 15 << 6.
			rg11b10ufloat: () => new Uint32Array( [ 0, 15 << 6 ] )
		}[ format ]();
		device.queue.writeTexture( { texture }, upload, { bytesPerRow: upload.byteLength }, [ 2, 1 ] );
		const input = device.createBuffer( { size: us.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } );
		device.queue.writeBuffer( input, 0, new Float32Array( us ) );
		const output = device.createBuffer( { size: us.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
		const read = device.createBuffer( { size: us.length * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
		const bind = device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: texture.createView() }, { binding: 1, resource: sampler }, { binding: 2, resource: { buffer: input } }, { binding: 3, resource: { buffer: output } } ] } );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline( pipeline ); pass.setBindGroup( 0, bind ); pass.dispatchWorkgroups( Math.ceil( us.length / 64 ) ); pass.end();
		encoder.copyBufferToBuffer( output, 0, read, 0, us.length * 4 );
		device.queue.submit( [ encoder.finish() ] );
		await read.mapAsync( GPUMapMode.READ );
		const values = Array.from( new Float32Array( read.getMappedRange().slice( 0 ) ) );
		[ texture, input, output, read ].forEach( resource => resource.destroy() );
		return values;

	}

	// Weights the 'exact' reduction asks for: output x of a 2n+1 → n level samples source u = (2x + 1) / n / 2
	// in normalized coordinates; with texel width 1/(2n+1) its fractional position past a centre is the weight.
	function oddWeights( size ) {

		const n = size >> 1, weights = [];
		for ( let x = 0; x < n; x ++ ) {

			const texel = ( ( 2 * x + 1 ) / ( 2 * n ) ) * size - 0.5;
			weights.push( texel - Math.floor( texel ) );

		}
		return [ ...new Set( weights.map( w => Math.round( w * 1e9 ) / 1e9 ) ) ];

	}

	const report = { adapter: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, description: adapter.info?.description }, formats: {} };
	const formats = [ 'rgba8unorm', 'rgba16float', 'rg11b10ufloat', ...( features.includes( 'float32-filterable' ) ? [ 'rgba32float' ] : [] ) ];
	for ( const format of formats ) {

		const values = await sample( format, fractions.map( t => 0.25 + t / 2 ) );
		const errors = values.map( ( value, i ) => value - fractions[ i ] );
		const distinct = new Set( values.map( value => value.toFixed( 7 ) ) ).size;
		const below = errors.filter( error => error < - 1e-6 ).length, above = errors.filter( error => error > 1e-6 ).length;
		const odd = {};
		for ( const size of [ 15, 17, 31, 65, 129, 255 ] ) {

			const weights = oddWeights( size );
			const got = await sample( format, weights.map( w => 0.25 + w / 2 ) );
			odd[ size ] = { maxError: Math.max( ...got.map( ( value, i ) => Math.abs( value - weights[ i ] ) ) ), weights: weights.length };

		}
		// Filter arithmetic: two non-trivial values, against a + w (b − a) with this sampler's own weights w.
		let arithmetic = null;
		if ( format !== 'rg11b10ufloat' ) {

			const [ a, b, ulp ] = format === 'rgba8unorm' ? [ 51 / 255, 230 / 255, 1 / 255 ] : [ 4, 5.5, 1 / 256 ];
			const wide = await sample( format, fractions.map( t => 0.25 + t / 2 ), 'wide' );
			const residuals = wide.map( ( value, i ) => value - ( a + values[ i ] * ( b - a ) ) );
			const worst = Math.max( ...residuals.map( Math.abs ) );
			arithmetic = { values: [ a, b ], maxResidual: worst, maxResidualSteps: worst / ulp, step: format === 'rgba8unorm' ? '1/255' : 'f16 ulp at 4–8 (1/256)' };
			if ( format !== 'rgba8unorm' ) {

				// 0.25 and 40: f16 filter arithmetic rounds the 40 · w product at 40's ulp (1/32), many
				// f16 ulps of a small result; f32 arithmetic doesn't. Errors in f16 ulps of the result.
				const skew = await sample( format, fractions.map( t => 0.25 + t / 2 ), 'skew' );
				const ulpOf = value => 2 ** ( Math.floor( Math.log2( Math.max( value, 2 ** - 14 ) ) ) - 10 );
				const errors = skew.map( ( value, i ) => { const exact = 0.25 + values[ i ] * 39.75; return { t: fractions[ i ], value, exact, resultUlps: ( value - exact ) / ulpOf( exact ) }; } );
				arithmetic.skew = { values: [ 0.25, 40 ], worst: errors.reduce( ( a, b ) => Math.abs( b.resultUlps ) > Math.abs( a.resultUlps ) ? b : a ) };

			}

		}
		report.formats[ format ] = {
			arithmetic,
			distinctWeights: distinct, maxError: Math.max( ...errors.map( Math.abs ) ),
			belowExact: below, aboveExact: above, of: values.length,
			// The weight at t = 0.5 + one 1/1024 step: 8-bit truncation gives 0.5, rounding 0.5 or 0.50390625.
			samples: [ 0, 1, 2, 3, 4, 256, 511, 512, 513, 768, 1023, 1024 ].map( i => [ fractions[ i ], values[ i ] ] ),
			oddSizes: odd
		};

	}
	return report;

} ), null, 1 ) );
await browser.close();
server.close();
