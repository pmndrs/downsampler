// Minimal static server for the GPU suite and device lab (repository root).
// `npm run lab` serves http://localhost:8080/test/browser/ — open it on any
// device on your network (WebGPU needs a secure context: localhost or HTTPS).
import { createServer as createHttpServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath( new URL( '..', import.meta.url ) );
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.map': 'application/json', '.css': 'text/css' };

export function createServer() {

	return createHttpServer( async ( request, response ) => {

		// Resolve directory URLs before normalize(): on Windows it turns '/' into '\', so a
		// trailing-slash check after it never matches and the directory itself 404s.
		const pathname = decodeURIComponent( new URL( request.url, 'http://localhost' ).pathname );
		const file = join( root, normalize( pathname.endsWith( '/' ) ? pathname + 'index.html' : pathname ) );
		if ( ! file.startsWith( root ) ) return response.writeHead( 403 ).end();

		try {

			const body = await readFile( file );
			response.writeHead( 200, { 'Content-Type': types[ extname( file ) ] || 'application/octet-stream', 'Cache-Control': 'no-store' } ).end( body );

		} catch {

			response.writeHead( 404 ).end( 'not found' );

		}

	} );

}

if ( process.argv[ 1 ] === fileURLToPath( import.meta.url ) ) {

	const port = Number( process.env.PORT || 8080 );
	createServer().listen( port, () => console.log( `device lab: http://localhost:${ port }/test/browser/` ) );

}
