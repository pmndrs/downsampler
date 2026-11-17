// Runs test/webgpu/compat-cube-mips.html in headless Chrome pinned to the Intel UHD.
// usage: node run-cube.mjs <worktree> [query...]
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const worktree = path.resolve( process.argv[ 2 ] );
const queries = process.argv.slice( 3 );
const require = createRequire( 'C:/Users/denni/Documents/GitHub/three.js/.claude/worktrees/compute-mipmaps-bounds-b02970/package.json' );
const puppeteer = require( 'puppeteer' );
const { createServer } = await import( pathToFileURL( path.join( worktree, 'utils/server.js' ) ).href );

const INTEL_LUID = process.env.INTEL_LUID || '0,66946';
const server = createServer( { root: worktree } );
await new Promise( resolve => server.listen( 0, resolve ) );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', `--use-adapter-luid=${ INTEL_LUID }` ] } );

try {

	const port = server.address().port;
	const probe = await browser.newPage();
	await probe.goto( `http://localhost:${ port }/` );
	const adapters = await probe.evaluate( async () => {

		const out = [];
		for ( const options of [ {}, { powerPreference: 'high-performance' }, { featureLevel: 'compatibility' } ] ) {

			const a = await navigator.gpu.requestAdapter( options );
			out.push( a ? `${ a.info.vendor } ${ a.info.architecture } ${ a.info.description }` : 'null' );

		}

		return out;

	} );
	await probe.close();
	console.log( 'adapters:', adapters );
	if ( adapters.some( a => ! a.startsWith( 'intel' ) ) ) throw new Error( 'not pinned to Intel; stopping' );

	for ( const query of queries ) {

		const page = await browser.newPage();
		page.on( 'pageerror', e => console.log( 'pageerror', e.message ) );
		page.on( 'console', m => console.log( 'console', m.text() ) );
		await page.goto( `http://localhost:${ port }/test/webgpu/compat-cube-mips.html${ query === '-' ? '' : query }` );
		await page.waitForFunction( () => window.testResults !== undefined, { timeout: 120000 } );
		const r = await page.evaluate( () => window.testResults );
		await page.close();
		console.log( `\n== ${ query } | ${ r.adapter } | compat ${ r.compat } | core-features ${ r.coreFeatures } | errors ${ JSON.stringify( r.errors ) } | textureBindingViewDimension reported ${ JSON.stringify( r.reported ) }` );
		for ( const { size, perLevel } of r.results ) {

			for ( const l of perLevel ) {

				const w = l.cubeVsArray.worst ? ` worst L${ l.cubeVsArray.worst.layer } (${ l.cubeVsArray.worst.x },${ l.cubeVsArray.worst.y }) c${ l.cubeVsArray.worst.channel } cube ${ l.cubeVsArray.worst.cube } array ${ l.cubeVsArray.worst.array }` : '';
				console.log( `${ size }² L${ l.level } ${ l.size }²: cube vs array max ${ l.cubeVsArray.maxDiff } (${ l.cubeVsArray.differ }/${ l.cubeVsArray.of })${ w } | vs ref: cube ${ l.cubeVsRef.maxErr } (${ l.cubeVsRef.overOneStep } >1), array ${ l.arrayVsRef.maxErr } (${ l.arrayVsRef.overOneStep } >1)${ l.cubeEqualsTexel ? ` | cube equals texel (0,0),(1,0),(0,1),(1,1) of 2²: ${ l.cubeEqualsTexel.join( '/' ) } of 24` : '' }` );

			}

		}

	}

} finally {

	await browser.close();
	server.close();

}
