// CI shards (test/ci/shard.js): every partition is disjoint, covers the listing exactly and is
// deterministic. The listing here is the committed durations' ids plus ids it doesn't know; the
// real listing comes from the browser (`node test/run.js --list --shards=N` checks it the same way).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { assignShards, digest, parseShard, partitionProblems, Selection, shardLoads } from '../ci/shard.js';

const recorded = JSON.parse( readFileSync( new URL( '../ci/durations.json', import.meta.url ), 'utf8' ) );
const durations = Object.fromEntries( Object.entries( recorded.groups ).map( ( [ id, group ] ) => [ id, group.ms ] ) );
const ids = [ ...Object.keys( durations ).reverse(), 'new/group/a', 'new/group/b' ];

test( 'shards are disjoint and their union is the listing, for 1–24 shards', () => {

	for ( const table of [ durations, {} ] ) for ( let total = 1; total <= 24; total ++ ) {

		const shards = assignShards( ids, total, table );
		assert.equal( shards.length, total );
		assert.deepEqual( partitionProblems( ids, shards ), [], `${ total } shards` );
		assert.equal( shards.reduce( ( sum, shard ) => sum + shard.length, 0 ), ids.length );

	}

} );

test( 'partitionProblems reports overlap, gaps and strangers', () => {

	assert.deepEqual( partitionProblems( [ 'a', 'b', 'c' ], [ [ 'a', 'b' ], [ 'b', 'x' ] ] ), [ 'b is in shards 1 and 2', 'c is in no shard', 'x is in a shard but not in the listing' ] );

} );

test( 'durations balance shards; the listing order does not change them', () => {

	const shards = assignShards( ids, 8, durations );
	const shuffled = assignShards( [ ...ids ].sort(), 8, durations );
	assert.deepEqual( shuffled.map( shard => [ ...shard ].sort() ), shards.map( shard => [ ...shard ].sort() ) );
	// Within a shard, enumeration order (the suite's own order).
	for ( const shard of shards ) assert.deepEqual( shard, [ ...shard ].sort( ( a, b ) => ids.indexOf( a ) - ids.indexOf( b ) ) );
	// LPT: the busiest shard is within the longest group of the mean (or is that group alone).
	const loads = shardLoads( shards, durations );
	const mean = loads.reduce( ( a, b ) => a + b, 0 ) / loads.length;
	const longest = Math.max( ...Object.values( durations ) );
	assert.ok( Math.max( ...loads ) <= Math.max( longest, mean + longest ), `loads ${ loads.map( Math.round ) }` );

} );

test( 'without durations: round-robin in listing order', () => {

	assert.deepEqual( assignShards( [ 'a', 'b', 'c', 'd', 'e' ], 2 ), [ [ 'a', 'c', 'e' ], [ 'b', 'd' ] ] );
	assert.deepEqual( assignShards( [ 'a', 'b' ], 3, { x: 5 } ), [ [ 'a' ], [ 'b' ], [] ] );

} );

test( 'parseShard', () => {

	assert.deepEqual( parseShard( '2/5' ), { index: 2, total: 5 } );
	for ( const bad of [ '0/3', '4/3', '1', 'a/b', '1/0' ] ) assert.throws( () => parseShard( bad ) );

} );

test( 'Selection: lists without running, runs only its shard, counts cases', async () => {

	const listing = new Selection( { list: true } );
	let ran = 0;
	for ( const id of [ 'a', 'b', 'a' ] ) await listing.run( id, async () => { ran ++; return [ 1 ]; } );
	assert.deepEqual( listing.ids, [ 'a', 'b' ] );
	assert.equal( ran, 0 );

	const shard = new Selection( { assigned: [ 'b' ] } );
	assert.deepEqual( await shard.run( 'a', async () => [ 1 ] ), [] );
	assert.deepEqual( await shard.run( 'b', async () => [ 1, 2 ] ), [ 1, 2 ] );
	assert.deepEqual( Object.keys( shard.groups ), [ 'b' ] );
	assert.equal( shard.groups.b.cases, 2 );
	assert.notEqual( digest( [ 'a', 'b' ] ), digest( [ 'b', 'a' ] ) );

} );
