// CI reports for the sharded GPU suite (test/ci/README.md).
//   node test/ci/report.js shard ci-timing.json             one runner's step summary (markdown on stdout)
//   node test/ci/report.js aggregate timing/ [--core=12] [--compat=12] [--write-durations=durations.json]
//       every shard's timing (*.json under timing/), per suite (core, compat): all N shards present and
//       passed, one listing, and the groups they ran disjoint and covering it. Prints a markdown summary;
//       exits 1 on a problem. --write-durations merges the core groups' times into a durations file.
//   NEEDS_JSON='${{ toJSON(needs) }}' node test/ci/report.js needs --require=layout,gpu
//       fails unless every --require job succeeded and no other needed job failed or was cancelled
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { partitionProblems } from './shard.js';

const [ command, target, ...rest ] = process.argv.slice( 2 );
const value = name => [ target, ...rest ].find( arg => arg?.startsWith( `--${ name }=` ) )?.split( '=' )[ 1 ];
const minutes = seconds => `${ Math.floor( seconds / 60 ) }:${ String( Math.round( seconds % 60 ) ).padStart( 2, '0' ) }`;
const label = timing => timing.shard ? `${ timing.shard.index }/${ timing.shard.total }` : 'whole';

function slowest( groups, count = 5 ) {

	return Object.entries( groups ?? {} ).sort( ( a, b ) => b[ 1 ].ms - a[ 1 ].ms ).slice( 0, count ).map( ( [ id, group ] ) => `\`${ id }\` ${ ( group.ms / 1000 ).toFixed( 0 ) } s` ).join( ', ' );

}

function files( dir ) {

	return readdirSync( dir ).flatMap( name => {

		const path = join( dir, name );
		return statSync( path ).isDirectory() ? files( path ) : name.endsWith( '.json' ) ? [ path ] : [];

	} );

}

if ( command === 'shard' ) {

	const timing = JSON.parse( readFileSync( target, 'utf8' ) );
	console.log( `## GPU suite${ timing.options?.compat ? ' (compatibility mode)' : '' }, shard ${ label( timing ) }\n` );
	console.log( timing.ok
		? `${ timing.cases } cases in ${ Object.keys( timing.groups ).length } of ${ timing.listing.groups } case groups, ${ minutes( timing.seconds ) } wall (validation ${ minutes( timing.validationMs / 1000 ) }). Slowest: ${ slowest( timing.groups ) }.`
		: `**Failed** after ${ minutes( timing.seconds ) }: ${ timing.error.split( '\n' )[ 0 ] }` );

} else if ( command === 'aggregate' ) {

	// Two suites may be sharded: core and compat (a compatibility-mode device). Each is checked on its own.
	const timings = existsSync( target ) ? files( target ).map( path => JSON.parse( readFileSync( path, 'utf8' ) ) ) : [];
	const suiteOf = timing => timing.options?.compat ? 'compat' : 'core';
	const problems = [];
	const lines = [ '## GPU suite\n', '| suite | shard | result | cases | groups | wall | slowest groups |', '| --- | --- | --- | ---: | ---: | ---: | --- |' ];
	const totals = [];
	const passedAll = [];

	for ( const suite of [ 'core', 'compat' ] ) {

		const shards = timings.filter( timing => suiteOf( timing ) === suite ).sort( ( a, b ) => ( a.shard?.index ?? 0 ) - ( b.shard?.index ?? 0 ) );
		const total = Number( value( suite ) ?? shards[ 0 ]?.shard?.total ?? 0 );
		if ( total === 0 && shards.length === 0 ) continue;
		if ( total === 0 ) { problems.push( `${ suite }: timing without a shard` ); continue; }

		for ( let index = 1; index <= total; index ++ ) {

			const found = shards.filter( timing => timing.shard?.index === index && timing.shard?.total === total );
			if ( found.length !== 1 ) problems.push( `${ suite } shard ${ index }/${ total }: ${ found.length } timing files` );
			else if ( ! found[ 0 ].ok ) problems.push( `${ suite } shard ${ index }/${ total } failed` );

		}

		const passed = shards.filter( timing => timing.ok && timing.shard?.total === total );
		const digests = new Set( passed.map( timing => timing.listing.digest ) );
		if ( digests.size > 1 ) problems.push( `${ suite }: shards saw different listings (${ [ ...digests ].join( ', ' ) }), so enumeration differs between runners` );
		if ( passed.length === total && digests.size === 1 ) problems.push( ...partitionProblems( passed[ 0 ].listing.ids, passed.map( timing => Object.keys( timing.groups ) ) ).map( problem => `${ suite }: ${ problem }` ) );
		for ( const timing of shards ) lines.push( `| ${ suite } | ${ label( timing ) } | ${ timing.ok ? 'passed' : 'failed' } | ${ timing.cases ?? '' } | ${ timing.ok ? Object.keys( timing.groups ).length : '' } | ${ minutes( timing.seconds ) } | ${ timing.ok ? slowest( timing.groups, 3 ) : timing.error?.split( '\n' )[ 0 ] ?? '' } |` );
		const cases = passed.reduce( ( sum, timing ) => sum + timing.cases, 0 );
		const longest = Math.max( 0, ...shards.map( timing => timing.seconds ) );
		totals.push( `${ suite }: ${ passed.length }/${ total } shards passed, ${ cases } cases in ${ passed[ 0 ]?.listing.groups ?? '?' } case groups (listing ${ [ ...digests ].join( ', ' ) || '?' }), longest shard ${ minutes( longest ) }` );
		if ( suite === 'core' ) passedAll.push( ...passed );

	}

	if ( totals.length === 0 ) problems.push( 'no shard timing found' );
	console.log( [ ...lines, '', ...totals.map( line => `- ${ line }` ) ].join( '\n' ) );
	if ( problems.length ) console.log( `\n**Problems**\n\n${ problems.map( problem => `- ${ problem }` ).join( '\n' ) }` );

	// Durations come from the core suite (the compat listing differs in a few groups; those count as the median).
	const out = value( 'write-durations' );
	if ( out ) {

		const file = existsSync( out ) ? JSON.parse( readFileSync( out, 'utf8' ) ) : { groups: {} };
		for ( const timing of passedAll ) for ( const [ id, { ms, cases } ] of Object.entries( timing.groups ) ) file.groups[ id ] = { ms: Math.round( ms ), cases };
		file.groups = Object.fromEntries( Object.keys( file.groups ).sort().map( id => [ id, file.groups[ id ] ] ) );
		file.recorded = { ...file.recorded, merged: new Date().toISOString().slice( 0, 10 ), from: `${ passedAll.length } CI shards` };
		writeFileSync( out, JSON.stringify( file, null, '\t' ) + '\n' );

	}

	if ( problems.length ) process.exitCode = 1;

} else if ( command === 'needs' ) {

	// Like homefig's assert-needs-results: parse toJSON(needs) rather than match rendered text. Jobs named
	// by --require must succeed; any other may also be skipped (a tier that doesn't run it), never fail.
	const required = ( value( 'require' ) ?? '' ).split( ',' ).filter( Boolean );
	const needs = JSON.parse( process.env.NEEDS_JSON || 'null' );
	const failures = [];
	if ( ! needs || typeof needs !== 'object' || Object.keys( needs ).length === 0 ) failures.push( 'NEEDS_JSON is missing or empty' );
	else {

		for ( const name of required ) if ( ! ( name in needs ) ) failures.push( `${ name } is not a needed job` );
		for ( const [ name, job ] of Object.entries( needs ) ) {

			const result = job?.result ?? 'missing';
			if ( result === 'success' || ( result === 'skipped' && ! required.includes( name ) ) ) continue;
			failures.push( `${ name } was ${ result }${ required.includes( name ) ? ' (it must succeed)' : '' }` );

		}

	}

	if ( failures.length ) {

		console.error( `Required jobs did not pass: ${ failures.join( '; ' ) }` );
		process.exitCode = 1;

	} else console.log( `Every needed job passed or was skipped by its tier: ${ Object.entries( needs ).map( ( [ name, job ] ) => `${ name } ${ job.result }` ).join( ', ' ) }.` );

} else {

	console.error( 'usage: node test/ci/report.js shard <ci-timing.json> | aggregate <dir> [--core=N] [--compat=M] [--write-durations=path] | needs [--require=a,b]' );
	process.exitCode = 2;

}
