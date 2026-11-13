// Shards of the GPU suite (test/ci/README.md). Runs in node and in the browser (no imports).
//
// The suite is a list of case groups, each with a stable id (validate.js names them). A run
// enumerates every id first (a pass that runs nothing), then `assignShards` splits them into
// N disjoint shards that cover the list, balanced by the per-group durations recorded in
// test/ci/durations.json. Each shard runs only its groups; every case belongs to one group.

/** '2/5' → { index: 2, total: 5 } (1-based). */
export function parseShard( value ) {

	const match = /^(\d+)\/(\d+)$/.exec( String( value ) );
	const index = match ? Number( match[ 1 ] ) : NaN, total = match ? Number( match[ 2 ] ) : NaN;
	if ( ! ( total >= 1 && index >= 1 && index <= total ) ) throw new Error( `--shard expects i/N with 1 ≤ i ≤ N, got ${ JSON.stringify( value ) }` );
	return { index, total };

}

/**
 * Splits `ids` (in enumeration order) into `total` disjoint shards that cover them.
 * With durations (id → ms): longest first onto the least-loaded shard (LPT), ties by id
 * and then by the lower shard, so the result doesn't depend on the order of `ids`. Ids
 * without a recorded duration count as the median of those with one. Without any
 * recorded duration for these ids: round-robin in enumeration order.
 * Each shard lists its ids in enumeration order.
 */
export function assignShards( ids, total, durations = {} ) {

	const unique = [ ...new Set( ids ) ];
	const order = new Map( unique.map( ( id, i ) => [ id, i ] ) );
	const shards = Array.from( { length: total }, () => [] );
	const known = unique.filter( id => Number.isFinite( durations[ id ] ) );

	if ( known.length === 0 ) {

		unique.forEach( ( id, i ) => shards[ i % total ].push( id ) );
		return shards;

	}

	const sorted = known.map( id => durations[ id ] ).sort( ( a, b ) => a - b );
	const median = sorted[ sorted.length >> 1 ];
	const weight = id => Number.isFinite( durations[ id ] ) ? durations[ id ] : median;
	const loads = new Array( total ).fill( 0 );

	for ( const id of [ ...unique ].sort( ( a, b ) => weight( b ) - weight( a ) || ( a < b ? - 1 : a > b ? 1 : 0 ) ) ) {

		let target = 0;
		for ( let i = 1; i < total; i ++ ) if ( loads[ i ] < loads[ target ] ) target = i;
		shards[ target ].push( id );
		loads[ target ] += weight( id );

	}

	for ( const shard of shards ) shard.sort( ( a, b ) => order.get( a ) - order.get( b ) );
	return shards;

}

/** Estimated ms per shard under `durations` (unknown ids: the median), as assignShards weighs them. */
export function shardLoads( shards, durations = {} ) {

	const known = Object.values( durations ).filter( Number.isFinite ).sort( ( a, b ) => a - b );
	const median = known.length ? known[ known.length >> 1 ] : 0;
	return shards.map( shard => shard.reduce( ( sum, id ) => sum + ( Number.isFinite( durations[ id ] ) ? durations[ id ] : median ), 0 ) );

}

/**
 * Problems with a partition of `ids`: overlap between shards, ids no shard has, ids no listing
 * has. Empty when the shards are disjoint and their union is exactly `ids`.
 */
export function partitionProblems( ids, shards ) {

	const problems = [];
	const owner = new Map();
	shards.forEach( ( shard, i ) => {

		for ( const id of shard ) {

			if ( owner.has( id ) ) problems.push( `${ id } is in shards ${ owner.get( id ) + 1 } and ${ i + 1 }` );
			else owner.set( id, i );

		}

	} );
	const all = new Set( ids );
	for ( const id of all ) if ( ! owner.has( id ) ) problems.push( `${ id } is in no shard` );
	for ( const id of owner.keys() ) if ( ! all.has( id ) ) problems.push( `${ id } is in a shard but not in the listing` );
	return problems;

}

/** A short, order-sensitive digest of an id list (FNV-1a), so shards can tell they saw the same listing. */
export function digest( ids ) {

	let hash = 0x811c9dc5;
	for ( const char of ids.join( '\n' ) ) {

		hash ^= char.codePointAt( 0 );
		hash = Math.imul( hash, 0x01000193 ) >>> 0;

	}

	return hash.toString( 16 ).padStart( 8, '0' );

}

/**
 * Gate and stopwatch for case groups. `has( id )` registers the id (in first-seen order) and
 * says whether this run executes it: never when listing, always without an assignment,
 * else when the shard owns it. `run( id, fn )` runs fn if `has( id )` and records its time
 * and the cases it returned (an array of results). Time between groups (shared setup:
 * textures, inputs, oracles) is charged to the group that follows it, so the recorded
 * durations add up to the run's wall time. With `onError( id, error )` set (--keep-going), a
 * group that throws is reported there and the run moves on to the next group. `enter( id )` and
 * `leave( id )`, when set, bracket each group that runs; `leave` resolves to an error or null.
 */
export class Selection {

	constructor( { assigned = null, list = false } = {} ) {

		this.assigned = assigned === null ? null : new Set( assigned );
		this.list = list;
		this.ids = [];
		this.seen = new Set();
		this.groups = {};
		this.last = performance.now();
		this.onError = null;

	}

	has( id ) {

		if ( ! this.seen.has( id ) ) { this.seen.add( id ); this.ids.push( id ); }
		return ! this.list && ( this.assigned === null || this.assigned.has( id ) );

	}

	async run( id, fn ) {

		if ( ! this.has( id ) ) return [];
		let results = [];

		try {

			// `enter( id )` and `leave( id )` (optional) bracket the group: an error `leave` returns
			// (a GPU validation error scope's) fails the group like a throw.
			this.enter?.( id );
			let failure = null;
			try {

				results = await fn();

			} catch ( error ) {

				failure = error;

			}

			const left = this.leave ? await this.leave( id ) : null;
			// A validation error first: it usually explains the failure that follows it.
			if ( left !== null ) failure = failure === null ? left : new Error( `${ left.message }; then: ${ failure.message }` );
			if ( failure === null ) return results;
			if ( ! this.onError ) throw failure;
			this.onError( id, failure );
			return results;

		} finally {

			const ended = performance.now();
			const group = this.groups[ id ] ??= { ms: 0, cases: 0 };
			group.ms += ended - this.last;
			group.cases += Array.isArray( results ) ? results.length : 0;
			this.last = ended;

		}

	}

}
