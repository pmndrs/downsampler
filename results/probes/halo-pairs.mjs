// Print capped halo / plain GPU time from the two runs in docs/M5-HANDOFF-SRGB.md.
// Both strategies in each ratio must come from the same row of the same run.
import { readFile } from 'node:fs/promises';

const files = process.argv.slice( 2 );
if ( files.length !== 2 ) throw new Error( 'Usage: node results/probes/halo-pairs.mjs exact.json box.json' );
const [ exact, box ] = await Promise.all( files.map( async path => JSON.parse( await readFile( path, 'utf8' ) ) ) );
for ( const field of [ 'vendor', 'architecture', 'device', 'description' ] ) {
	if ( ! exact.adapter || ! box.adapter || exact.adapter[ field ] !== box.adapter[ field ] ) throw new Error( 'The reports must use the same adapter' );
}

function ratios( report, plain, halo ) {

	if ( ! report.timing?.timestamps ) throw new Error( 'The report has no GPU timestamps' );
	const rows = new Map();
	for ( const row of report.timing.results ) {
		const key = `${ row.width }×${ row.height }×${ row.layers } ${ row.format }`;
		const baseline = row.gpuMs?.[ plain ], candidate = row.gpuMs?.[ halo ];
		if ( rows.has( key ) || ! Number.isFinite( baseline ) || ! Number.isFinite( candidate ) || baseline <= 0 || candidate <= 0 ) throw new Error( `Invalid or duplicate ${ plain } / ${ halo } timings for ${ key }` );
		rows.set( key, candidate / baseline );
	}
	if ( ! rows.size ) throw new Error( 'No timing rows' );
	return rows;

}

const a = ratios( exact, 'tail', 'halo' ), b = ratios( box, 'tailBox', 'haloBox' );
if ( a.size !== b.size || [ ...a.keys() ].some( key => ! b.has( key ) ) ) throw new Error( 'Exact and box reports cover different sizes or formats' );
console.log( [ exact.adapter.vendor, exact.adapter.architecture ].filter( Boolean ).join( ' ' ) );
console.table( [ ...a ].map( ( [ workload, value ] ) => ( { workload, 'exact halo / plain': value.toFixed( 3 ), 'box halo / plain': b.get( workload ).toFixed( 3 ) } ) ) );
