// The GPU suite in parallel shards on one machine, unattended: every shard capped in time, one summary.
//
//   node test/parallel.js [--jobs=6] [--shards=18] [--cap=10] [--budget=30] [--out=results/local/<name>] -- <test/run.js flags>
//   node test/parallel.js -- --quick --power=high
//   node test/parallel.js --cap=8 -- --power=low --durations=results/windows/durations-intel-uhd.json
//
// The suite is CPU-bound (each case's oracle is computed in JavaScript on the page), so one page
// uses one core while the GPU idles. This runs `--shards` disjoint shards (test/ci/shard.js),
// `--jobs` at a time, each in its own headless Chrome with --keep-going, and kills any shard that
// runs past `--cap` minutes (it is reported as timed out, with the groups it didn't finish).
// `--budget` minutes bounds the whole run: past it, running shards are killed and queued ones
// skipped, all reported. Measured on a laptop RTX 3060 (6 jobs): --quick 2 min, full 17 min.
//
// Writes to --out: shard-<i>.log, report-<i>.json, timing-<i>.json, and
//   summary.md / summary.json   failures, timeouts, crashes, cases, wall time, slowest groups
//   durations.json              this device's per-group times: pass it back with --durations=
//                               (after the --) to balance the next run's shards
// Exits 1 on any failure, timeout or crash. Shards share the GPU: fine for correctness, never
// for timing (benchmarks stay in test/run.js --bench, alone on the GPU).
import { spawn, spawnSync } from 'node:child_process';
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const argv = process.argv.slice( 2 );
const split = argv.indexOf( '--' );
const own = split < 0 ? argv : argv.slice( 0, split ), passed = split < 0 ? [] : argv.slice( split + 1 );
const value = name => own.find( arg => arg.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];

const jobs = Number( value( 'jobs' ) ?? Math.max( 1, Math.min( 6, availableParallelism() - 2 ) ) );
// More shards than jobs: a queue evens out shards whose recorded durations are off on this device.
const total = Number( value( 'shards' ) ?? jobs * 3 );
const capMs = Number( value( 'cap' ) ?? 10 ) * 60000;
const budgetMs = Number( value( 'budget' ) ?? 30 ) * 60000;
const stamp = new Date().toISOString().slice( 0, 16 ).replace( /[:T]/g, '-' );
const out = value( 'out' ) ?? join( 'results', 'local', stamp );
const runner = fileURLToPath( new URL( './run.js', import.meta.url ) );
for ( const flag of [ '--shard=', '--output=', '--timing=', '--list', '--bench' ] ) if ( passed.some( arg => arg.startsWith( flag ) ) ) throw new Error( `test/parallel.js sets ${ flag } itself (or doesn't support it)` );
mkdirSync( out, { recursive: true } );

const started = Date.now();
const shards = Array.from( { length: total }, ( _, i ) => ( { index: i + 1 } ) );
console.log( `${ total } shards, ${ jobs } at a time, ${ capMs / 60000 } min cap each, ${ budgetMs / 60000 } min budget → ${ out }` );
const running = new Set();
let overBudget = false;
const budgetTimer = setTimeout( () => {

	overBudget = true;
	for ( const { shard, child } of running ) { shard.overBudget = true; kill( child ); }

}, budgetMs );

/** Kill a shard and its Chrome (a process tree on Windows; a process group elsewhere). */
function kill( child ) {

	if ( process.platform === 'win32' ) spawnSync( 'taskkill', [ '/pid', String( child.pid ), '/T', '/F' ], { stdio: 'ignore' } );
	else try { process.kill( - child.pid, 'SIGKILL' ); } catch { child.kill( 'SIGKILL' ); }

}

function runShard( shard ) {

	return new Promise( resolve => {

		const files = { log: join( out, `shard-${ shard.index }.log` ), report: join( out, `report-${ shard.index }.json` ), timing: join( out, `timing-${ shard.index }.json` ) };
		const args = [ runner, ...passed, '--keep-going', `--shard=${ shard.index }/${ total }`, `--output=${ files.report }`, `--timing=${ files.timing }` ];
		const log = createWriteStream( files.log );
		const begin = Date.now();
		const child = spawn( process.execPath, args, { detached: process.platform !== 'win32', windowsHide: true } );
		child.stdout.pipe( log, { end: false } );
		child.stderr.pipe( log, { end: false } );
		const timer = setTimeout( () => { shard.timedOut = true; kill( child ); }, capMs );
		const entry = { shard, child };
		running.add( entry );
		child.on( 'close', code => {

			clearTimeout( timer );
			running.delete( entry );
			log.end();
			Object.assign( shard, { code, seconds: Math.round( ( Date.now() - begin ) / 1000 ), files } );
			const state = shard.overBudget ? 'KILLED (budget)' : shard.timedOut ? 'TIMED OUT' : code === 0 ? 'passed' : 'FAILED';
			console.log( `  shard ${ shard.index }/${ total }: ${ state } in ${ shard.seconds } s` );
			resolve();

		} );

	} );

}

// A queue of shards, `jobs` running at once.
const queue = [ ...shards ];
await Promise.all( Array.from( { length: Math.min( jobs, total ) }, async () => {

	while ( queue.length && ! overBudget ) await runShard( queue.shift() );

} ) );
clearTimeout( budgetTimer );

// Merge.
const read = path => existsSync( path ) ? JSON.parse( readFileSync( path, 'utf8' ) ) : null;
const failures = [], groups = {}, problems = [];
let cases = 0, checks = 0, adapter = null, listing = null;
const skipped = new Set();

for ( const shard of shards ) {

	if ( ! shard.files ) { problems.push( `shard ${ shard.index }: not run (budget of ${ budgetMs / 60000 } min spent)` ); continue; }
	const report = read( shard.files.report ), timing = read( shard.files.timing );
	if ( report ) {

		const { validation } = report;
		adapter ??= report.adapter;
		cases += validation.cases ?? 0;
		checks += validation.checks ?? 0;
		for ( const failure of validation.failures ?? [] ) failures.push( { shard: shard.index, ...failure } );
		for ( const reason of validation.skipped ?? [] ) skipped.add( reason );

	}
	if ( timing?.groups ) Object.assign( groups, timing.groups );
	listing ??= timing?.listing ?? null;
	if ( shard.overBudget ) problems.push( `shard ${ shard.index }: killed when the ${ budgetMs / 60000 } min budget ran out (log ${ shard.files.log })` );
	else if ( shard.timedOut ) problems.push( `shard ${ shard.index }: timed out after ${ capMs / 60000 } min (log ${ shard.files.log })` );
	else if ( ! report ) {

		// A crash: the last lines of its log say why.
		const tail = existsSync( shard.files.log ) ? readFileSync( shard.files.log, 'utf8' ).split( /\r?\n/ ).filter( line => line && ! /^\s+at /.test( line ) ).slice( - 3 ).join( ' | ' ) : '';
		problems.push( `shard ${ shard.index }: exit ${ shard.code }, no report: ${ tail.slice( 0, 400 ) }` );

	}

}

const ran = Object.keys( groups ).length;
const seconds = Math.round( ( Date.now() - started ) / 1000 );
const slowest = Object.entries( groups ).sort( ( a, b ) => b[ 1 ].ms - a[ 1 ].ms ).slice( 0, 10 );
const device = adapter ? [ adapter.vendor, adapter.architecture, adapter.description ].filter( Boolean ).join( ' ' ) : 'unknown';
const summary = {
	date: new Date().toISOString(), flags: passed, jobs, shards: total, capMinutes: capMs / 60000, seconds, adapter,
	ok: failures.length === 0 && problems.length === 0,
	cases, checks, groups: { ran, listed: listing?.groups ?? null },
	failures, problems, skipped: [ ...skipped ],
	perShard: shards.map( ( { index, code, seconds, timedOut, overBudget, files } ) => ( { index, ran: Boolean( files ), code, seconds, timedOut: Boolean( timedOut ), overBudget: Boolean( overBudget ) } ) ),
	slowest: slowest.map( ( [ id, group ] ) => ( { id, ms: Math.round( group.ms ), cases: group.cases } ) )
};
writeFileSync( join( out, 'summary.json' ), JSON.stringify( summary, null, '\t' ) + '\n' );
writeFileSync( join( out, 'durations.json' ), JSON.stringify( {
	about: `Per-group ms on ${ device } (${ passed.join( ' ' ) || 'full' }), from test/parallel.js; pass with --durations= to balance shards on this device. Shards ran ${ jobs } at a time, so times include contention.`,
	recorded: { date: summary.date.slice( 0, 10 ), run: passed.join( ' ' ), adapter: device },
	groups: Object.fromEntries( Object.keys( groups ).sort().map( id => [ id, { ms: Math.round( groups[ id ].ms ), cases: groups[ id ].cases } ] ) )
}, null, '\t' ) + '\n' );

const md = [
	`# GPU suite, ${ summary.ok ? 'passed' : 'FAILED' }`,
	'',
	`- Device: ${ device }`,
	`- Flags: \`${ passed.join( ' ' ) || '(full)' }\`; ${ total } shards, ${ jobs } at a time, cap ${ capMs / 60000 } min`,
	`- ${ cases } cases, ${ checks } pattern runs, ${ ran }${ listing ? ` of ${ listing.groups }` : '' } case groups; wall ${ Math.floor( seconds / 60 ) } min ${ seconds % 60 } s`,
	'',
	...( problems.length ? [ '## Shards that did not finish', '', ...problems.map( line => `- ${ line }` ), '' ] : [] ),
	...( failures.length ? [ `## Failures (${ failures.length })`, '', ...failures.map( f => `- ${ f.first }${ f.thrown ? '' : ` (${ f.failing }/${ f.values } values outside, max error ${ f.maxError })` }` ), '' ] : [] ),
	...( skipped.size ? [ '## Skipped (feature unavailable)', '', ...[ ...skipped ].map( line => `- ${ line }` ), '' ] : [] ),
	'## Slowest groups', '',
	...slowest.map( ( [ id, group ] ) => `- \`${ id }\` ${ ( group.ms / 1000 ).toFixed( 1 ) } s, ${ group.cases } cases` ),
	''
].join( '\n' );
writeFileSync( join( out, 'summary.md' ), md );
console.log( '\n' + md );
process.exitCode = summary.ok ? 0 : 1;
