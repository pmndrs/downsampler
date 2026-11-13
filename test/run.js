// Headless GPU suite runner.
//   node test/run.js                     validation suite
//   node test/run.js --quick             small shapes only
//   node test/run.js --only=scalar       one part of the suite: downsampler, reducer, scalar, fixtures
//                                        (a partition), or mips (downsampler + reducer); CI uses --shard instead
//                                        or box (the 'box' and resample checks of downsampler, for debugging)
//   node test/run.js --bench             validation + benchmark
//   node test/run.js --bench --sizes=1024x1024,512x512x6 --formats=rgba8unorm --rounds=11 --repeats=16
//   node test/run.js --output=results/report.json --headful
//   node test/run.js --adapter=swiftshader   software adapter (CI, no GPU)
//   node test/run.js --bench --compare=spd --skip-validation   M2T.4: exact / tail / spd / atomic probe / render
//   node test/run.js --bench --compare=render --skip-validation   DIVERGES #17: render passes, then 'exact'; level 1 alone
//   node test/run.js --quality --skip-validation              M2T.4: 'spd' vs 'exact' per level, Hi-Z violations
//   node test/run.js --compat            on a compatibility-mode device (featureLevel 'compatibility')
//   node test/run.js --power=low         force the low-power GPU (default high); for laptops with two GPUs
//   node test/run.js --keep-going        report every failing check instead of stopping at the first
//   node test/run.js --bench --compare=webgpu-spd --formats=rgba8unorm,rgba16float,r32float --skip-validation
//                                        webgpu-spd 3.0.0 vs exact / tail / spd / render (results/compare/webgpu-spd.md)
//   node test/run.js --webgpu-spd-check --skip-validation [--frames=64]   its correctness against our oracles
// M3 measurement modes (test/bench/README.md; all with --bench --skip-validation, i.e. npm run bench -- …):
//   --compare=throughput (or --throughput)   chains per second without timestamps: render / exact / spd / tail,
//                                        then the same by timestamps and whether both order them alike
//       [--window=1000] [--pacing=queue|raf] [--in-flight=2] [--batch-ms=8] [--max-batch=1024] [--textures=1]
//       [--no-timestamps] skip the timestamp check; [--no-timestamp-query] a device without timestamp-query
//   --compare=storage-usage              sampling a texture with vs without STORAGE_BINDING (--strategies=fullscreen,material,write)
//   --compare=subgroups                  upper bound for a subgroup kernel (--strategies=lds,slot,shuffle,nosync)
//   --compare=f16                        averaging in f16 vs f32, with the precision cost (--strategies=f32,f16,f16safe)
//   --throughput with --compare=storage-usage|subgroups|f16 adds per-frame throughput to the timestamps
//   --strategies=a,b,c                   time a subset (any --compare mode; the ratios' reference is always kept)
// Shards (test/ci/README.md):
//   node test/run.js --quick --adapter=swiftshader --shard=2/8   one of 8 disjoint shards, balanced by test/ci/durations.json
//   node test/run.js --quick --adapter=swiftshader --list --shards=4,8   check the partitions (runs nothing); --verbose lists ids
//   node test/run.js --quick --adapter=swiftshader --list --shard=2/8    the case groups of one shard
//   node test/run.js --quick --adapter=swiftshader --record-durations    rewrite test/ci/durations.json (sharded: merge)
//   node test/run.js ... --timing=ci-timing.json                         per-group timing and the listing, for the CI aggregate
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { loadavg } from 'node:os';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { gpuChoiceArgs } from './gpu-choice.js';
import { assignShards, parseShard, partitionProblems, shardLoads } from './ci/shard.js';
import { createServer } from './serve.js';

const args = process.argv.slice( 2 );
const value = name => args.find( arg => arg.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];
const options = { bench: args.includes( '--bench' ), quick: args.includes( '--quick' ), compat: args.includes( '--compat' ) };
if ( value( 'only' ) ) options.only = value( 'only' );
if ( value( 'sizes' ) ) options.sizes = value( 'sizes' ).split( ',' ).map( size => size.split( 'x' ).map( Number ) );
if ( value( 'formats' ) ) options.formats = value( 'formats' ).split( ',' );
if ( value( 'rounds' ) ) options.rounds = Number( value( 'rounds' ) );
if ( value( 'repeats' ) ) options.repeats = Number( value( 'repeats' ) );
if ( value( 'compare' ) ) options.compare = value( 'compare' );
if ( args.includes( '--quality' ) ) options.quality = true;
if ( args.includes( '--skip-validation' ) ) options.skipValidation = true;
if ( args.includes( '--webgpu-spd-check' ) ) options.webgpuSpdCheck = true;
if ( args.includes( '--keep-going' ) ) options.keepGoing = true;
if ( value( 'frames' ) ) options.frames = Number( value( 'frames' ) );
if ( value( 'strategies' ) ) options.strategies = value( 'strategies' ).split( ',' );
if ( args.includes( '--throughput' ) ) {

	options.throughput = true;
	options.compare ??= 'throughput';

}
if ( args.includes( '--no-timestamps' ) ) options.noTimestamps = true;
if ( args.includes( '--no-timestamp-query' ) ) options.timestampQuery = false;
for ( const [ flag, key ] of [ [ 'window', 'windowMs' ], [ 'in-flight', 'inFlight' ], [ 'batch-ms', 'batchMs' ], [ 'max-batch', 'maxBatch' ], [ 'textures', 'textures' ] ] ) if ( value( flag ) ) options[ key ] = Number( value( flag ) );
if ( value( 'pacing' ) ) options.pacing = value( 'pacing' );
const loadBefore = loadavg();
const startedAt = Date.now();

// Shards: recorded per-group durations balance them (round-robin without any).
const durationsPath = value( 'durations' ) ?? fileURLToPath( new URL( './ci/durations.json', import.meta.url ) );
const recorded = existsSync( durationsPath ) ? JSON.parse( readFileSync( durationsPath, 'utf8' ) ) : { groups: {} };
const durations = Object.fromEntries( Object.entries( recorded.groups ).map( ( [ id, group ] ) => [ id, group.ms ] ) );
const list = args.includes( '--list' );
if ( value( 'shard' ) ) Object.assign( options, { shard: parseShard( value( 'shard' ) ), durations } );
if ( list ) options.list = true;
const recordPath = args.includes( '--record-durations' ) ? durationsPath : value( 'record-durations' );
const timingPath = value( 'timing' );

const browserArgs = [ '--enable-unsafe-webgpu' ];
if ( process.env.CI ) browserArgs.push( '--no-sandbox' );
if ( value( 'adapter' ) === 'swiftshader' ) browserArgs.push( '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan' );
// --power=high|low picks the GPU on machines with two (e.g. a laptop's NVIDIA and Intel; see
// test/gpu-choice.js for Windows); the printed adapter info confirms which one ran.
browserArgs.push( ...gpuChoiceArgs( value( 'power' ) ) );
// Measurement only: Dawn toggles, e.g. --dawn-features=disable_robustness,skip_validation to see how
// much of a gap to native is the browser's bounds checks and validation.
if ( value( 'dawn-features' ) ) browserArgs.push( `--enable-dawn-features=${ value( 'dawn-features' ) }` );

const server = createServer();
let browser;

try {

	await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
	// The full matrix takes minutes (over 30 on a laptop RTX 3060); the default 180 s protocol timeout would cut it off.
	browser = await puppeteer.launch( { headless: ! args.includes( '--headful' ), args: browserArgs, protocolTimeout: 120 * 60 * 1000 } );
	const page = await browser.newPage();
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => {

		if ( message.type() === 'error' ) errors.push( message.text() );
		else if ( args.includes( '--verbose' ) && message.text().startsWith( 'PROGRESS ' ) ) console.log( message.text().slice( 9 ) );

	} );
	page.setDefaultTimeout( 600000 );
	await page.goto( `http://127.0.0.1:${ server.address().port }/test/browser/` );
	// Loading the suite page takes seconds: fail fast, with the page's errors, if it never loads.
	await page.waitForFunction( () => typeof window.runSuite === 'function', { timeout: 60000 } ).catch( error => {

		throw new Error( `suite page did not load: ${ error.message }${ errors.length ? `\n${ errors.join( '\n' ) }` : '' }` );

	} );
	let report;

	try {

		report = await page.evaluate( options => window.runSuite( options ), options );
		if ( errors.length ) throw new Error( errors.join( '\n' ) );

	} catch ( error ) {

		// A failed shard still leaves its wall time for the CI summary.
		if ( timingPath ) await writeTiming( { ok: false, error: String( error.message ?? error ).slice( 0, 2000 ) } );
		throw error;

	}

	const { validation, timing, adapter } = report;
	if ( validation.listed ) {

		listShards( validation.ids );
		await browser.close();
		server.close();
		process.exit( process.exitCode ?? 0 );

	}

	report.loadAverage = { before: loadBefore, after: loadavg() };
	console.log( JSON.stringify( adapter ) );
	console.log( `load average (1, 5, 15 min): before ${ loadBefore.map( v => v.toFixed( 1 ) ).join( ' ' ) }, after ${ report.loadAverage.after.map( v => v.toFixed( 1 ) ).join( ' ' ) }` );
	if ( validation.failures?.length ) {

		// --keep-going: every failing check, then a failed exit.
		for ( const failure of validation.failures ) console.log( failure.thrown ? `FAIL ${ failure.first }` : `FAIL ${ failure.first } (${ failure.failing }/${ failure.values } values outside, max error ${ failure.maxError })` );
		console.log( `validation FAILED: ${ validation.failures.length } failing checks in ${ validation.cases } cases, ${ validation.checks } pattern runs, ${ validation.ms } ms` );
		process.exitCode = 1;

	} else console.log( `validation passed: ${ validation.cases } cases, ${ validation.checks } pattern runs, ${ validation.ms } ms` );
	if ( validation.scalarPrecision ) console.log( `scalar float sums: worst error ${ ( validation.scalarPrecision.worst * 100 ).toFixed( 2 ) } % of the oracle's bound; ${ validation.scalarPrecision.pairing } results equal the oracle's float32 pairing bit for bit` );
	if ( validation.skipped.length ) console.log( `skipped (feature unavailable): ${ validation.skipped.join( ', ' ) }` );
	const ran = Object.keys( validation.groups ?? {} );
	if ( ! validation.listing ) { /* --skip-validation: no case groups ran */ } else if ( validation.shard ) console.log( `shard ${ validation.shard.index }/${ validation.shard.total }: ${ ran.length } of ${ validation.listing.groups } case groups (listing ${ validation.listing.digest }), ${ validation.cases } cases` );
	else console.log( `${ ran.length } case groups (listing ${ validation.listing.digest })` );
	if ( timingPath ) await writeTiming( { ok: true, adapter, cases: validation.cases, checks: validation.checks, validationMs: validation.ms, listing: validation.listing, groups: validation.groups, skipped: validation.skipped } );
	if ( recordPath ) await recordDurations( recordPath, validation, adapter );

	if ( timing?.timestamps === false && ! timing.mode ) console.log( 'timestamp-query unavailable: benchmark skipped' );
	if ( timing?.mode && ! timing.timestamps ) console.log( 'timestamp-query unavailable or off: throughput only' );
	const fixed = ( x, digits = 4 ) => x === undefined || x === null ? '—' : x.toFixed( digits );

	if ( timing?.mode === 'throughput' ) {

		// Wall ms per chain ×(÷ render, paired per round) [CPU share], then timestamps for the check.
		console.table( timing.results.map( row => ( {
			size: `${ row.width }x${ row.height }`, format: row.format,
			...Object.fromEntries( timing.strategies.map( name => {

				const t = row.throughput[ name ];
				return [ name, `${ fixed( t.msPerUnit ) } ×${ fixed( t.ratio, 2 ) } [cpu ${ Math.round( t.cpuShare * 100 ) } %${ t.cpuBound ? ' BOUND' : '' }, n ${ t.batch }]${ row.gpuMs ? ` ts ${ fixed( row.gpuMs[ name ] ) }` : '' }` ];

			} ) ),
			...( row.agreement ? { order: `${ row.agreement.agree }/${ row.agreement.counted } τ ${ fixed( row.agreement.tau, 2 ) }` } : {} )
		} ) ) );
		if ( timing.agreement ) console.log( `throughput vs timestamps: ${ timing.agreement.agree } of ${ timing.agreement.counted } strategy pairs ordered alike (pairs within ${ timing.agreement.tie * 100 } % by timestamps are ties); mean Kendall τ ${ fixed( timing.agreement.meanTau, 2 ) }` );
		for ( const row of timing.results ) for ( const line of row.agreement?.disagree ?? [] ) console.log( `  ${ row.width }x${ row.height } ${ row.format }: ${ line }` );

	} else if ( timing?.mode === 'storage-usage' ) {

		console.table( timing.results.map( row => ( {
			size: `${ row.width }x${ row.height }`, format: row.format, content: row.content, shader: row.shader,
			'plain ms': fixed( row.gpuMs?.plain ), 'storage ms': fixed( row.gpuMs?.storage ), 'storage ÷ plain': fixed( row.ratio, 3 ),
			...( row.throughput ? { 'throughput ÷': fixed( row.throughputRatio, 3 ) } : {} )
		} ) ) );
		console.log( 'median storage ÷ plain over sizes (min–max):' );
		console.table( Object.entries( timing.summary ).map( ( [ key, v ] ) => ( { case: key, median: fixed( v.median, 3 ), range: `${ fixed( v.min, 3 ) }–${ fixed( v.max, 3 ) }` } ) ) );

	} else if ( timing?.mode === 'subgroups' || timing?.mode === 'f16' ) {

		if ( timing.skipped.length ) console.log( `skipped (feature unavailable): ${ timing.skipped.join( ', ' ) }` );
		console.log( `plan: exactTile ${ timing.exactTile }, tail ${ timing.tail }` );
		const check = c => c ? Object.entries( c ).map( ( [ pattern, d ] ) => `${ pattern } ${ d.differing ? `${ d.differing }/${ d.values } differ${ d.worstSteps ? `, worst ${ d.worstSteps } steps (L${ d.worstLevel })` : '' }${ d.nonFinite ? `, ${ d.nonFinite } non-finite` : '' }` : 'identical' }` ).join( '; ' ) : '';
		console.table( timing.results.map( row => ( {
			size: `${ row.width }x${ row.height }`, format: row.format, batches: row.batches.join( ', ' ),
			...Object.fromEntries( timing.variants.map( name => [ name, `${ fixed( row.gpuMs?.[ name ] ) } ×${ fixed( row.ratios?.[ name ] ?? row.throughput?.[ name ].ratio, 3 ) }${ row.throughput && row.gpuMs ? ` (tp ×${ fixed( row.throughput[ name ].ratio, 3 ) })` : '' } ${ check( row.checks[ name ] ) }` ] ) )
		} ) ) );
		console.log( 'median ratio to the library kernel (render sizes / squares), and the check over every size and pattern:' );
		console.table( Object.entries( timing.summary ).map( ( [ key, v ] ) => ( { case: key, 'render sizes': fixed( v.renderSizes, 3 ), squares: fixed( v.squares, 3 ), ...( 'identical' in v ? { check: v.identical ? 'identical' : `${ v.differing }/${ v.values } differ${ v.worstSteps ? `, worst ${ v.worstSteps } steps` : '' }${ v.nonFinite ? `, ${ v.nonFinite } non-finite` : '' }` } : {} ) } ) ) );

	} else if ( timing?.results.length && options.compare === 'spd' ) {

		// Same-run ratios to 'exact' (average) and 'exactMax' (max).
		const names = timing.strategies;
		console.table( timing.results.map( row => ( {
			size: `${ row.width }x${ row.height }`, format: row.format,
			passes: names.map( name => `${ name } ${ row.passes[ name ] }` ).join( ', ' ),
			...Object.fromEntries( names.map( name => [ name, `${ row.gpuMs[ name ].toFixed( 4 ) } (${ ( row.gpuMs[ name ] / row.gpuMs[ name.endsWith( 'Max' ) ? 'exactMax' : 'exact' ] ).toFixed( 2 ) })` ] ) )
		} ) ) );

	} else if ( timing?.results.length && ( options.compare === 'webgpu-spd' || options.compare === 'render' ) ) {

		// Paired per-round ratios to 'exact' (averages) and 'exactMax' (max); 'render': to render passes.
		console.table( timing.results.map( row => ( {
			size: `${ row.width }x${ row.height }`, format: row.format,
			...Object.fromEntries( Object.keys( row.gpuMs ).map( name => [ name, `${ row.gpuMs[ name ].toFixed( 4 ) } ×${ row.ratios[ name ].toFixed( 2 ) } [${ row.passes[ name ] }]` ] ) )
		} ) ) );

	} else if ( timing?.results.length ) {

		const names = timing.strategies;
		const best = names.filter( name => name.startsWith( 'compute' ) ).pop();
		console.table( timing.results.map( row => ( {
			size: `${ row.width }x${ row.height }x${ row.layers }`, format: row.format,
			passes: names.map( name => row.passes[ name ] ).join( ' / ' ),
			...Object.fromEntries( names.map( name => [ `${ name } ms`, row.gpuMs[ name ].toFixed( 4 ) ] ) ),
			[ `${ best } vs render` ]: `${ ( row.gpuMs.render / row.gpuMs[ best ] ).toFixed( 2 ) }×`,
			[ `${ best } vs compute4` ]: `${ ( row.gpuMs.compute4 / row.gpuMs[ best ] ).toFixed( 2 ) }×`
		} ) ) );

	}

	if ( report.quality ) {

		console.table( report.quality.flatMap( row => row.levels.filter( level => level.width * level.height > 1 || level.level === row.levels.length ).map( level => ( {
			size: `${ row.width }x${ row.height }`, format: row.format, pattern: row.pattern, level: `${ level.level } (${ level.width }x${ level.height })`,
			'spd-exact max': level.spdVsExact.max.toFixed( 4 ), 'spd-exact mean': level.spdVsExact.mean.toFixed( 5 ),
			'exact-ideal mean': level.exactVsIdeal.mean.toFixed( 5 ), 'spd-ideal mean': level.spdVsIdeal.mean.toFixed( 5 ),
			'tail-exact max': level.tailVsExact.max.toFixed( 4 ),
			'max violations e/s/t': `${ level.maxViolations.exact }/${ level.maxViolations.spd }/${ level.maxViolations.tail }`
		} ) ) ) );

	}

	if ( report.webgpuSpd ) {

		const w = report.webgpuSpd;
		console.log( 'webgpu-spd variants:', JSON.stringify( w.variants ) );
		console.table( w.average.flatMap( row => Object.entries( row.variants ).map( ( [ name, v ] ) => ( {
			size: `${ row.width }x${ row.height }`, format: row.format, variant: name, dispatches: v.dispatches, 'unit step': row.firstUnitStep,
			'in-bounds mismatches': `${ v.inBounds.mismatches } / ${ v.inBounds.values }`, exact: v.inBounds.exact, poisoned: v.inBounds.poisoned, 'worst (steps)': v.inBounds.worst.toFixed( 2 ),
			'= ours spd': `${ v.vsOursSpd.identical } / ${ v.vsOursSpd.values } (worst ${ v.vsOursSpd.worst.toFixed( 3 ) })`,
			'unit levels': v.unit.map( u => `L${ u.level } ${ u.size } zero:${ u.zeroMismatches } z1:${ u.zeroAlpha1Mismatches } clamp:${ u.clampMismatches } vs ours ${ u.vsOursSpd?.relative.toFixed( 2 ) }` ).join( '; ' )
		} ) ) ) );
		console.table( w.hiz.flatMap( row => Object.entries( row.reductions ).flatMap( ( [ reduction, entry ] ) => Object.entries( entry ).map( ( [ name, v ] ) => ( {
			size: `${ row.width }x${ row.height }`, reduction, strategy: name, violations: `${ v.violations } / ${ v.texels }`, 'worst level': `${ ( v.worstLevelFraction * 100 ).toFixed( 2 ) } %`,
			oracle: v.oracle ? `${ v.oracle.mismatches } / ${ v.oracle.values } (poisoned ${ v.oracle.poisoned })` : '—',
			unit: v.unit ? v.unit.map( u => `L${ u.level } zero:${ u.zeroMismatches } clamp:${ u.clampMismatches }` ).join( '; ' ) : '—'
		} ) ) ) ) );
		console.table( w.stress.flatMap( row => Object.entries( row.variants ).map( ( [ name, v ] ) => ( { size: `${ row.width }x${ row.height }`, format: row.format, variant: name, levels: v.levels, 'bad frames (one encoder)': `${ v.encoder.badFrames } / ${ v.frames }`, 'bad frames (submit each)': `${ v.submits.badFrames } / ${ v.frames }`, 'isolated runs ≠ majority': `${ v.isolatedDiffer.length } / ${ v.isolatedRuns }` } ) ) ) );
		console.log( 'probes:', JSON.stringify( w.probes, null, 1 ) );
		if ( w.validationError ) console.log( 'GPU validation error during the check:', w.validationError );

	}

	if ( value( 'output' ) ) {

		await mkdir( dirname( value( 'output' ) ), { recursive: true } );
		await writeFile( value( 'output' ), JSON.stringify( { ...report, browserArgs }, null, 2 ) + '\n' );

	}

} finally {

	if ( browser ) await browser.close();
	server.close();

}

/** `--timing`: this run's wall time, shard, listing and per-group timing (test/ci/report.js reads it). */
async function writeTiming( fields ) {

	const timing = { shard: options.shard ?? null, options: { quick: options.quick, compat: options.compat, only: options.only ?? null }, seconds: Math.round( ( Date.now() - startedAt ) / 1000 ), ...fields };
	await mkdir( dirname( timingPath ), { recursive: true } );
	await writeFile( timingPath, JSON.stringify( timing ) + '\n' );

}

/**
 * `--record-durations`: per-group ms and cases into durations.json. An unsharded run of the whole
 * suite replaces the file (dropping groups that no longer exist); a shard or an `--only` part merges.
 */
async function recordDurations( path, validation, adapter ) {

	const whole = ! options.shard && ! options.only;
	const groups = whole ? {} : { ...recorded.groups };
	for ( const [ id, { ms, cases } ] of Object.entries( validation.groups ) ) groups[ id ] = { ms: Math.round( ms ), cases };
	const sorted = Object.fromEntries( Object.keys( groups ).sort().map( id => [ id, groups[ id ] ] ) );
	const flags = [ options.quick && '--quick', value( 'adapter' ) && `--adapter=${ value( 'adapter' ) }`, options.compat && '--compat' ].filter( Boolean ).join( ' ' );
	const file = {
		about: 'Per-case-group wall time (ms) and cases of the GPU suite, which balance CI shards (test/ci/README.md). Regenerate with `node test/run.js --quick --adapter=swiftshader --record-durations`, or merge CI timing artifacts with test/ci/report.js.',
		recorded: whole ? { date: new Date().toISOString().slice( 0, 10 ), run: flags, adapter: adapter.description || adapter.architecture || adapter.vendor || 'unknown' } : { ...recorded.recorded, merged: new Date().toISOString().slice( 0, 10 ) },
		groups: sorted
	};
	await mkdir( dirname( path ), { recursive: true } );
	await writeFile( path, JSON.stringify( file, null, '\t' ) + '\n' );
	console.log( `${ whole ? 'wrote' : 'merged' } ${ Object.keys( validation.groups ).length } group durations into ${ path }` );

}

/** `--list`: the case groups, per shard with `--shard=i/N` or `--shards=N,M`; checks each partition. */
function listShards( ids ) {

	const unknown = ids.filter( id => ! Number.isFinite( durations[ id ] ) );
	console.log( `${ ids.length } case groups; ${ ids.length - unknown.length } with a recorded duration (${ ( ids.reduce( ( sum, id ) => sum + ( durations[ id ] ?? 0 ), 0 ) / 60000 ).toFixed( 1 ) } min recorded)` );
	// In CI, a warning annotation: re-record test/ci/durations.json (test/ci/README.md).
	if ( unknown.length ) console.log( `${ process.env.GITHUB_ACTIONS ? '::warning title=Unrecorded case groups::' : '' }${ unknown.length } case groups have no recorded duration (weighed as the median): ${ unknown.length > 12 ? `${ unknown.slice( 0, 12 ).join( ', ' ) }, …` : unknown.join( ', ' ) }` );
	const stale = Object.keys( durations ).filter( id => ! ids.includes( id ) );
	if ( stale.length && ! options.only ) console.log( `recorded but not in this listing: ${ stale.length }` );
	const counts = value( 'shards' ) ? value( 'shards' ).split( ',' ).map( Number ) : options.shard ? [ options.shard.total ] : [];
	if ( counts.length === 0 ) for ( const id of ids ) console.log( id );

	for ( const total of counts ) {

		const shards = assignShards( ids, total, durations );
		const problems = partitionProblems( ids, shards );
		const loads = shardLoads( shards, durations );
		console.log( `\n${ total } shards: ${ problems.length ? `PARTITION BROKEN\n  ${ problems.join( '\n  ' ) }` : 'disjoint, cover every group' }` );
		shards.forEach( ( shard, i ) => {

			if ( options.shard && options.shard.total === total && options.shard.index !== i + 1 ) return;
			console.log( `  shard ${ i + 1 }/${ total }: ${ shard.length } groups, ~${ ( loads[ i ] / 60000 ).toFixed( 1 ) } min recorded` );
			if ( args.includes( '--verbose' ) || options.shard?.index === i + 1 ) for ( const id of shard ) console.log( `    ${ id }` );

		} );
		if ( problems.length ) process.exitCode = 1;

	}

}
