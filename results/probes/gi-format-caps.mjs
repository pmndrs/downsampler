// Probe: which usages rg11b10ufloat and rgb9e5ufloat allow, with and without
// texture-formats-tier1, and whether they are filterable.
// Run: node results/probes/gi-format-caps.mjs [--swiftshader]
import puppeteer from 'puppeteer';
import { createServer } from '../../test/serve.js';
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const swift = process.argv.includes( '--swiftshader' );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', ...( swift ? [ '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' ] : [] ) ] } );
const page = await browser.newPage();
await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
console.log( JSON.stringify( await page.evaluate( async () => {
	const adapter = await navigator.gpu.requestAdapter();
	const out = { userAgent: navigator.userAgent, features: [ ...adapter.features ].sort() };
	for ( const tier1 of [ false, true ] ) {
		if ( tier1 && ! adapter.features.has( 'texture-formats-tier1' ) ) continue;
		const device = await ( await navigator.gpu.requestAdapter() ).requestDevice( { requiredFeatures: tier1 ? [ 'texture-formats-tier1' ] : [] } );
		const result = {};
		const scope = async ( name, fn ) => { device.pushErrorScope( 'validation' ); fn(); result[ name ] = ( await device.popErrorScope() )?.message ?? 'ok'; };
		for ( const format of [ 'rg11b10ufloat', 'rgb9e5ufloat' ] ) {
			for ( const usage of [ 'STORAGE_BINDING', 'RENDER_ATTACHMENT', 'COPY_DST' ] ) await scope( `${ format } ${ usage }`, () => device.createTexture( { size: [ 4, 4 ], format, usage: GPUTextureUsage[ usage ] | GPUTextureUsage.TEXTURE_BINDING } ).destroy() );
			await scope( `${ format } filterable`, () => {
				const texture = device.createTexture( { size: [ 4, 4 ], format, usage: GPUTextureUsage.TEXTURE_BINDING } );
				const layout = device.createBindGroupLayout( { entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } } ] } );
				device.createBindGroup( { layout, entries: [ { binding: 0, resource: texture.createView() }, { binding: 1, resource: device.createSampler( { minFilter: 'linear', magFilter: 'linear' } ) } ] } );
			} );
		}
		out[ tier1 ? 'tier1' : 'core' ] = result;
		device.destroy();
	}
	return out;
} ), null, 1 ) );
await browser.close(); server.close();
