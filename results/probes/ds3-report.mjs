import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';

const args = process.argv.slice( 2 );
const outputOption = args.find( arg => arg.startsWith( '--output=' ) );
const output = resolve( outputOption?.slice( '--output='.length ) ?? 'results/windows/ds3/summary.md' );
const inputs = args.filter( arg => ! arg.startsWith( '--' ) );

if ( args.includes( '--help' ) ) {

	console.log( 'Usage: node results/probes/ds3-report.mjs [result.json ...] [--output=summary.md]\nDefaults to results/windows/ds3/{nvidia,intel,swiftshader}.json. Legacy adapter names are also accepted.' );
	process.exit( 0 );

}

const files = ( inputs.length ? inputs : [ 'nvidia', 'intel', 'swiftshader' ] ).map( input => input.endsWith( '.json' ) ? resolve( input ) : resolve( `results/windows/ds3/${ input }.json` ) );
const median = values => {

	const sorted = [ ...values ].sort( ( a, b ) => a - b );
	if ( ! sorted.length ) return null;
	const middle = Math.floor( sorted.length / 2 );
	return sorted.length % 2 ? sorted[ middle ] : ( sorted[ middle - 1 ] + sorted[ middle ] ) / 2;

};
const format = value => value === null || value === undefined ? 'n/a' : value.toFixed( 1 );
const unique = values => [ ...new Set( values ) ];
const pipelineCount = counters => [ 'computeSync', 'computeAsync', 'renderSync', 'renderAsync' ].reduce( ( sum, key ) => sum + ( counters[ key ] ?? 0 ), 0 );
const maxGap = ( runs, phase, type ) => {

	const values = runs.map( run => run[ phase ]?.[ type ]?.maxMs ).filter( Number.isFinite );
	return values.length ? Math.max( ...values ) : null;

};

function compiler( run ) {

	if ( ! run.trace || run.trace.status === 'disabled' ) return null;
	if ( run.trace.status === 'failed' ) throw new Error( `Trace failed: ${ run.trace.error }` );
	const events = run.trace.compilerEvents ?? [];
	const spans = run.trace.compilerSpans ?? events.filter( event => event.phase === 'X' );
	return {
		count: events.filter( event => [ 'X', 'B' ].includes( event.phase ) ).length,
		ms: spans.reduce( ( sum, event ) => sum + event.durationUs / 1000, 0 ),
		names: unique( events.map( event => event.name ) ),
	};

}

function validate( report, label ) {

	const runs = report.runs;
	if ( report.error || ! runs?.length ) throw new Error( `${ label }: missing or failed run: ${ report.error ?? '' }` );
	const paths = report.options?.paths ?? unique( runs.map( run => run.output ) );
	const repeats = report.options?.repeats ?? unique( runs.map( run => run.pair ) ).length;
	const caches = [ 'fresh-profile', 'same-browser-reload' ];
	const expected = paths.length * repeats * caches.length * 2;
	if ( runs.length !== expected ) throw new Error( `${ label }: ${ runs.length } runs; expected ${ expected }` );
	for ( let pair = 0; pair < repeats; pair ++ ) {

		for ( const path of paths ) for ( const cache of caches ) for ( const arm of [ 'sync', 'warmup' ] ) {

			if ( runs.filter( run => run.pair === pair && run.output === path && run.cache === cache && run.arm === arm ).length !== 1 ) throw new Error( `${ label }: missing/duplicate ${ pair }/${ path }/${ cache }/${ arm }` );

		}

	}
	for ( const run of runs ) {

		if ( run.error || run.errors?.length ) throw new Error( `${ label }: browser/validation errors in ${ run.output }/${ run.arm }` );
		if ( ! run.pixels?.length || run.pixels.some( mip => mip.maxError > ( run.fixture?.tolerance ?? 1 ) ) ) throw new Error( `${ label }: pixel oracle failed` );
		if ( run.arm === 'warmup' && ( ! run.warmup || pipelineCount( run.first.created ) !== 0 ) ) throw new Error( `${ label }: warm-up readiness failed` );
		if ( pipelineCount( run.second.created ) !== 0 ) throw new Error( `${ label }: second encode created a pipeline` );
		compiler( run );

	}
	for ( const path of paths ) {

		const hashes = runs.filter( run => run.output === path ).map( run => JSON.stringify( run.pixels.map( mip => [ mip.level, mip.width, mip.height, mip.hash ] ) ) );
		if ( new Set( hashes ).size !== 1 ) throw new Error( `${ label }/${ path }: A/B pixel hashes differ` );

	}
	return { paths, repeats, caches };

}

const lines = [
	'# DS3 pipeline warm-up measurements', '',
	'Same-build A/B compares synchronous first encode with awaited `warmup(texture)` before first encode. Each pair uses separate private profiles for both arms, followed by a same-browser reload. OS/driver caches remain. Values below are medians in milliseconds; traces add overhead when enabled.', '',
	'| Dataset/cache | Path | First recording sync → warmed | First submit-to-done sync → warmed | Warm-up wait | Second record-to-done sync → warmed | Compiler events sync / warmed | Compiler CPU ms sync → warmed |',
	'|---|---|---:|---:|---:|---:|---:|---:|',
];
const details = [];
const buildDigests = [];
for ( const file of files ) {

	const report = JSON.parse( await readFile( file, 'utf8' ) );
	const label = basename( file, '.json' );
	const { paths, repeats, caches } = validate( report, label );
	const runs = report.runs;
	const build = report.build?.sha256 ?? report.buildSha256 ?? 'unrecorded';
	buildDigests.push( build );
	for ( const cache of caches ) {

		for ( const path of paths ) {

			const group = arm => runs.filter( run => run.cache === cache && run.output === path && run.arm === arm );
			const sync = group( 'sync' );
			const warm = group( 'warmup' );
			const pair = extract => `${ format( median( sync.map( extract ) ) ) } → ${ format( median( warm.map( extract ) ) ) }`;
			const traced = runs.every( run => compiler( run ) !== null );
			const events = traced ? pair( run => compiler( run ).count ) : 'n/a';
			const cpu = traced ? pair( run => compiler( run ).ms ) : 'n/a';
			lines.push( `| ${ label }/${ cache } | ${ path } | ${ pair( run => run.first.recordMs ) } | ${ pair( run => run.first.submitToDoneMs ) } | ${ format( median( warm.map( run => run.warmup.elapsedMs ) ) ) } | ${ pair( run => run.second.recordToDoneMs ) } | ${ events } | ${ cpu } |` );

		}

	}
	const sync = runs.filter( run => run.arm === 'sync' );
	const warm = runs.filter( run => run.arm === 'warmup' );
	const logMessages = unique( runs.flatMap( run => run.logs ?? [] ) );
	if ( logMessages.length ) details.push( `CDP error log messages requiring review: ${ JSON.stringify( logMessages ) }.` );
	const provenance = report.schema >= 2 && report.build?.files?.length ? 'immutable dist snapshot' : 'legacy end-of-run hash; build isolation unverified';
	details.push( `**${ label }**: ${ runs.length } page runs, ${ repeats } alternating pairs per path, ${ runs.reduce( ( sum, run ) => sum + run.pixels.length, 0 ) } generated mip readbacks. Browser ${ unique( runs.map( run => run.browser ) ).join( ', ' ) }; platform ${ report.platform ?? 'unrecorded' }; adapter ${ JSON.stringify( runs[ 0 ].adapter ) }. Provenance: ${ provenance }; SHA256 \`${ build }\`; captured ${ report.build?.capturedAt ?? 'unrecorded' }.` );
	details.push( `First-use maximum rAF gap sync → warmed: ${ format( maxGap( sync, 'first', 'raf' ) ) } → ${ format( maxGap( warm, 'first', 'raf' ) ) } ms; preceding warm-up maximum ${ format( maxGap( warm, 'warmup', 'raf' ) ) } ms. First-use heartbeat sync → warmed: ${ format( maxGap( sync, 'first', 'heartbeat' ) ) } → ${ format( maxGap( warm, 'first', 'heartbeat' ) ) } ms; warm-up maximum ${ format( maxGap( warm, 'warmup', 'heartbeat' ) ) } ms. First submit() call medians sync → warmed: ${ format( median( sync.map( run => run.first.submitMs ) ) ) } → ${ format( median( warm.map( run => run.first.submitMs ) ) ) } ms.` );
	for ( const path of paths ) {

		const group = runs.filter( run => run.output === path );
		const signatures = unique( group.map( run => `${ run.counters.modules } modules, ${ run.counters.wgslBytes } WGSL bytes, ${ pipelineCount( run.counters ) } pipelines` ) );
		const apiModes = unique( group.map( run => `${ run.arm }: ${ JSON.stringify( Object.fromEntries( [ 'computeSync', 'computeAsync', 'renderSync', 'renderAsync' ].map( key => [ key, run.counters[ key ] ] ) ) ) }` ) );
		const compilerNames = unique( group.flatMap( run => compiler( run )?.names ?? [] ) );
		details.push( `${ path }: ${ signatures.join( '; ' ) }. API modes: ${ apiModes.join( '; ' ) }. Observed compiler events: ${ compilerNames.join( ', ' ) || 'none or tracing disabled' }. Fixture ${ JSON.stringify( group[ 0 ].fixture ?? 'legacy 512×288 constant RGBA input; nine generated mips' ) }.` );

	}
	const stalls = phase => [ 'sync', 'warmup' ].map( arm => {

		const selected = runs.filter( run => run.arm === arm );
		return `${ arm } ${ selected.reduce( ( sum, run ) => sum + ( run[ phase ]?.raf?.over50ms ?? 0 ), 0 ) } rAF / ${ selected.reduce( ( sum, run ) => sum + ( run[ phase ]?.heartbeat?.over50ms ?? 0 ), 0 ) } heartbeat`;

	} ).join( '; ' );
	details.push( `Gaps above 50 ms, first-use: ${ stalls( 'first' ) }; preparation: ${ stalls( 'warmup' ) }.` );
	details.push( 'Validation passed: constant-input oracle, identical per-path A/B mip hashes, zero pipeline calls at first encode after warm-up, and zero pipeline calls at second encode in either arm.' );

}
lines.push( '', ...details.flatMap( detail => [ detail, '' ] ) );
lines.push( `Cross-dataset build digests: ${ unique( buildDigests ).length }. Compare timing deltas within an adapter. Different digests require reviewing implementation changes before comparing builds.`, '',
	'Compiler events count DXC/FXC complete events or begin events. CPU duration sums complete spans and reconstructed begin/end spans when available. No observed DXC/FXC event does not prove a cache hit: backend support and trace availability differ. Pipeline API calls are distinct from shader compiler invocations.', '',
	'Warm-up moves compilation before first submission; it does not remove compilation or guarantee lower total startup time. Headless rAF describes browser scheduling, not physical presentation. The constant-input fixture is a narrow execution check; nonconstant GPU oracle tests remain necessary.', '',
	'Reproduce with `npm run build`, then `node results/probes/ds3-warmup.mjs --repeats=3 --trace --output=PATH.json` and optional `--power=high|low`, `--adapter=swiftshader` or `--chrome=PATH`. Run adapters sequentially. Regenerate using `node results/probes/ds3-report.mjs PATH.json ... --output=SUMMARY.md`.', '',
	'Skill feedback: none.', '' );
await mkdir( dirname( output ), { recursive: true } );
await writeFile( output, lines.join( '\n' ) );
console.log( `Verified ${ files.length } datasets; report: ${ output }` );
