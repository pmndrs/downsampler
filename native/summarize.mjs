// Tables for results/windows/native.md: native ms, our (WebGPU) ms and their ratios, per GPU,
// format and size, from the JSON that results/windows/run-benches.sh writes.
//
//   node native/summarize.mjs [results/windows] > results/windows/native-tables.md
//
// Native medians have the `empty` variant (timer overhead per chain) subtracted, as the WebGPU
// numbers have their marker passes subtracted. Ratios are ours ÷ native: above 1, native is faster.
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const folder = process.argv[ 2 ] ?? 'results/windows';
const load = name => {

	const path = join( folder, name );
	return existsSync( path ) ? JSON.parse( readFileSync( path, 'utf8' ) ) : null;

};

const sizes = [ [ 960, 540 ], [ 1129, 635 ], [ 1280, 720 ], [ 1707, 960 ], [ 1920, 1080 ], [ 2560, 1440 ], [ 1024, 1024 ], [ 4096, 4096 ] ];
const ms = value => value === undefined || value === null || Number.isNaN( value ) ? '–' : value.toFixed( 3 );
const ratio = ( a, b ) => a === undefined || b === undefined || ! ( b > 0 ) ? '–' : ( a / b ).toFixed( 2 );

/** Native medians, overhead subtracted: { 'format WxH': { variant: ms } }. */
function nativeTable( json ) {

	const table = {};
	if ( ! json ) return table;
	for ( const row of json.results ) ( table[ `${ row.format } ${ row.width }x${ row.height }` ] ??= {} )[ row.variant ] = row.medianMs;
	for ( const entry of Object.values( table ) ) {

		const overhead = entry.empty ?? 0;
		for ( const name of Object.keys( entry ) ) if ( name !== 'empty' ) entry[ name ] = Math.max( 0, entry[ name ] - overhead );

	}
	return table;

}

/** Our medians: { 'format WxH': { strategy: ms } }. */
function oursTable( json ) {

	const table = {};
	if ( ! json ) return table;
	for ( const row of json.timing.results ) Object.assign( table[ `${ row.format } ${ row.width }x${ row.height }` ] ??= {}, row.gpuMs );
	return table;

}

const lines = [];
const out = text => lines.push( text );

for ( const [ tag, label ] of [ [ 'rtx3060', 'RTX 3060 Laptop' ], [ 'intel-uhd', 'Intel UHD' ] ] ) {

	const d3d12Json = load( `native-d3d12-${ tag }.json` ), vkJson = load( `native-vk-${ tag }.json` );
	const spdJson = load( `${ tag }-spd.json` ), wspdJson = load( `${ tag }-webgpu-spd.json` );
	if ( ! d3d12Json && ! vkJson ) continue;
	const d3d12 = nativeTable( d3d12Json ), vk = nativeTable( vkJson );
	const ours = oursTable( spdJson ), oursHiZ = oursTable( wspdJson );

	out( `## ${ label }` );
	out( '' );
	if ( d3d12Json ) out( `- D3D12: ${ d3d12Json.adapter.name }, driver ${ d3d12Json.adapter.driver }, ${ d3d12Json.date }` );
	if ( vkJson ) out( `- Vulkan: ${ vkJson.adapter.name }, ${ vkJson.adapter.api }, driver ${ vkJson.adapter.driver }, ${ vkJson.date }` );
	if ( spdJson ) out( `- WebGPU: ${ spdJson.adapter.vendor } ${ spdJson.adapter.architecture }, ${ ( spdJson.userAgent.match( /Chrome\/[\d.]+/ ) ?? [ '?' ] )[ 0 ] }, ${ spdJson.date }` );
	out( '' );

	for ( const format of [ 'rgba8unorm', 'rgba16float' ] ) {

		out( `### ${ format }, average: ms per full chain` );
		out( '' );
		out( '| Size | ours `exact` | ours `spd` | ours atomic | ours `render` | AMD SPD (atomic, wave, load) | AMD SPD best | D3D12 render | D3D11 GenerateMips | Vulkan blit | nvpro |' );
		out( '|---|---|---|---|---|---|---|---|---|---|---|' );
		for ( const [ w, h ] of sizes ) {

			const key = `${ format } ${ w }x${ h }`, o = ours[ key ] ?? {}, d = d3d12[ key ] ?? {}, v = vk[ key ] ?? {};
			const spdNames = Object.keys( d ).filter( name => name.startsWith( 'spd-' ) );
			const best = spdNames.sort( ( a, b ) => d[ a ] - d[ b ] )[ 0 ];
			out( `| ${ w }×${ h } | ${ ms( o.exact ) } | ${ ms( o.spd ) } | ${ ms( o.atomic ) } | ${ ms( o.render ) } | ${ ms( d[ 'spd-atomic-wave-load' ] ) } | ${ best ? `${ ms( d[ best ] ) } (${ best.slice( 4 ) })` : '–' } | ${ ms( d.render ) } | ${ ms( d[ 'd3d11-generatemips' ] ) } | ${ ms( v.blit ) } | ${ ms( v[ 'nvpro-pyramid' ] ) } |` );

		}
		out( '' );
		out( `Ratios, ours ÷ native (above 1: native is faster):` );
		out( '' );
		out( '| Size | `spd` ÷ AMD SPD | atomic ÷ AMD SPD | `spd` ÷ SPD split LDS linear | `exact` ÷ nvpro | `exact` ÷ GenerateMips | `exact` ÷ D3D12 render | WebGPU render ÷ D3D12 render | D3D12 render ÷ render-rtonly |' );
		out( '|---|---|---|---|---|---|---|---|---|' );
		for ( const [ w, h ] of sizes ) {

			const key = `${ format } ${ w }x${ h }`, o = ours[ key ] ?? {}, d = d3d12[ key ] ?? {}, v = vk[ key ] ?? {};
			out( `| ${ w }×${ h } | ${ ratio( o.spd, d[ 'spd-atomic-wave-load' ] ) } | ${ ratio( o.atomic, d[ 'spd-atomic-wave-load' ] ) } | ${ ratio( o.spd, d[ 'spd-split-lds-linear' ] ) } | ${ ratio( o.exact, v[ 'nvpro-pyramid' ] ) } | ${ ratio( o.exact, d[ 'd3d11-generatemips' ] ) } | ${ ratio( o.exact, d.render ) } | ${ ratio( o.render, d.render ) } | ${ ratio( d.render, d[ 'render-rtonly' ] ) } |` );

		}
		out( '' );

	}

	out( '### r32float, max (Hi-Z): ms per full chain' );
	out( '' );
	out( '| Size | ours `exact` max | AMD SPD atomic wave | AMD SPD atomic LDS | AMD SPD split wave | AMD SPD split LDS | nvpro max | ours ÷ AMD SPD atomic wave | ours ÷ nvpro |' );
	out( '|---|---|---|---|---|---|---|---|---|' );
	for ( const [ w, h ] of sizes ) {

		const key = `r32float ${ w }x${ h }`, o = oursHiZ[ key ] ?? {}, d = d3d12[ key ] ?? {}, v = vk[ key ] ?? {};
		out( `| ${ w }×${ h } | ${ ms( o.exactMax ) } | ${ ms( d[ 'spd-atomic-wave' ] ) } | ${ ms( d[ 'spd-atomic-lds' ] ) } | ${ ms( d[ 'spd-split-wave' ] ) } | ${ ms( d[ 'spd-split-lds' ] ) } | ${ ms( v[ 'nvpro-pyramid' ] ) } | ${ ratio( o.exactMax, d[ 'spd-atomic-wave' ] ) } | ${ ratio( o.exactMax, v[ 'nvpro-pyramid' ] ) } |` );

	}
	out( '' );

	// Every native variant, for reference.
	out( `<details><summary>All native variants, ${ label }</summary>` );
	out( '' );
	for ( const [ api, table ] of [ [ 'D3D12/D3D11', d3d12 ], [ 'Vulkan', vk ] ] ) {

		const keys = Object.keys( table );
		if ( ! keys.length ) continue;
		const variants = [ ...new Set( keys.flatMap( key => Object.keys( table[ key ] ) ) ) ].filter( name => name !== 'empty' );
		out( `| ${ api } | ${ variants.join( ' | ' ) } |` );
		out( `|---|${ variants.map( () => '---' ).join( '|' ) }|` );
		for ( const key of keys ) out( `| ${ key } | ${ variants.map( name => ms( table[ key ][ name ] ) ).join( ' | ' ) } |` );
		out( '' );

	}
	out( '</details>' );
	out( '' );

}

// Run-to-run spread on the RTX 3060 (second pass of the same jobs).
const first = { ours: oursTable( load( 'rtx3060-spd.json' ) ), d3d12: nativeTable( load( 'native-d3d12-rtx3060.json' ) ) };
const second = { ours: oursTable( load( 'rtx3060-spd-2.json' ) ), d3d12: nativeTable( load( 'native-d3d12-rtx3060-2.json' ) ) };
if ( Object.keys( second.ours ).length && Object.keys( second.d3d12 ).length ) {

	out( '## Run-to-run, RTX 3060 (pass 2 ÷ pass 1)' );
	out( '' );
	out( '| Format, size | ours `exact` | ours `spd` | ours `render` | AMD SPD | D3D12 render | GenerateMips | `spd` ÷ AMD SPD, pass 1 → 2 |' );
	out( '|---|---|---|---|---|---|---|---|' );
	for ( const format of [ 'rgba8unorm', 'rgba16float' ] ) for ( const [ w, h ] of sizes ) {

		const key = `${ format } ${ w }x${ h }`;
		const a = first.ours[ key ] ?? {}, b = second.ours[ key ] ?? {}, c = first.d3d12[ key ] ?? {}, d = second.d3d12[ key ] ?? {};
		out( `| ${ format } ${ w }×${ h } | ${ ratio( b.exact, a.exact ) } | ${ ratio( b.spd, a.spd ) } | ${ ratio( b.render, a.render ) } | ${ ratio( d[ 'spd-atomic-wave-load' ], c[ 'spd-atomic-wave-load' ] ) } | ${ ratio( d.render, c.render ) } | ${ ratio( d[ 'd3d11-generatemips' ], c[ 'd3d11-generatemips' ] ) } | ${ ratio( a.spd, c[ 'spd-atomic-wave-load' ] ) } → ${ ratio( b.spd, d[ 'spd-atomic-wave-load' ] ) } |` );

	}
	out( '' );

}

console.log( lines.join( '\n' ) );
