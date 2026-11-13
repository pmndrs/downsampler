// Shared by the native harnesses (d3d12-mips, vk-mips): the same input data, the same
// decoding, the same statistics and the same dump format, so their results line up with
// each other and with check-dump.mjs.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace harness {

struct Size { uint32_t width, height; };

inline std::vector<Size> parseSizes( const std::string & text ) {

	std::vector<Size> sizes;
	std::stringstream stream( text );
	std::string item;
	while ( std::getline( stream, item, ',' ) ) {

		Size size {};
		if ( sscanf( item.c_str(), "%ux%u", &size.width, &size.height ) == 2 ) sizes.push_back( size );
		else if ( sscanf( item.c_str(), "%u", &size.width ) == 1 ) { size.height = size.width; sizes.push_back( size ); }

	}
	return sizes;

}

inline std::vector<std::string> split( const std::string & text ) {

	std::vector<std::string> parts;
	std::stringstream stream( text );
	std::string item;
	while ( std::getline( stream, item, ',' ) ) if ( ! item.empty() ) parts.push_back( item );
	return parts;

}

/** The formats the native runs cover: average for the two colour formats, max (Hi-Z) for r32float. */
enum class Format { rgba8unorm, rgba16float, r32float };

inline const char * formatName( Format format ) { return format == Format::rgba8unorm ? "rgba8unorm" : format == Format::rgba16float ? "rgba16float" : "r32float"; }
inline uint32_t formatBytes( Format format ) { return format == Format::rgba16float ? 8 : 4; }
inline bool formatIsMax( Format format ) { return format == Format::r32float; }

inline bool parseFormat( const std::string & name, Format & format ) {

	for ( Format candidate : { Format::rgba8unorm, Format::rgba16float, Format::r32float } ) if ( name == formatName( candidate ) ) { format = candidate; return true; }
	return false;

}

inline uint16_t floatToHalf( float value ) {

	// Round to nearest even; inputs here are finite and in [0, 1].
	uint32_t bits;
	memcpy( &bits, &value, 4 );
	const uint32_t sign = ( bits >> 16 ) & 0x8000;
	const int exponent = int( ( bits >> 23 ) & 0xff ) - 127 + 15;
	uint32_t mantissa = bits & 0x7fffff;
	if ( exponent <= 0 ) {

		if ( exponent < -10 ) return uint16_t( sign );
		mantissa |= 0x800000;
		const int shift = 14 - exponent;
		uint32_t half = mantissa >> shift;
		const uint32_t rest = mantissa & ( ( 1u << shift ) - 1 ), midpoint = 1u << ( shift - 1 );
		if ( rest > midpoint || ( rest == midpoint && ( half & 1 ) ) ) half ++;
		return uint16_t( sign | half );

	}
	uint32_t half = ( uint32_t( exponent ) << 10 ) | ( mantissa >> 13 );
	const uint32_t rest = mantissa & 0x1fff;
	if ( rest > 0x1000 || ( rest == 0x1000 && ( half & 1 ) ) ) half ++;
	return uint16_t( sign | half );

}

inline float halfToFloat( uint16_t half ) {

	const uint32_t exponent = ( half >> 10 ) & 0x1f, mantissa = half & 0x3ff;
	float value;
	if ( exponent == 0 ) value = std::ldexp( float( mantissa ), -24 );
	else if ( exponent == 31 ) value = mantissa ? NAN : INFINITY;
	else value = std::ldexp( float( mantissa | 0x400 ), int( exponent ) - 25 );
	return ( half & 0x8000 ) ? -value : value;

}

/** Level 0 in the texture's format: uniform random channels in [0, 1), xorshift32 seeded 1337. */
inline std::vector<uint8_t> makeData( uint32_t width, uint32_t height, Format format ) {

	std::vector<uint8_t> data( size_t( width ) * height * formatBytes( format ) );
	uint32_t state = 1337;
	auto next = [ & ]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return float( state >> 8 ) / 16777216.0f; };
	const size_t texels = size_t( width ) * height;
	if ( format == Format::rgba8unorm ) for ( size_t i = 0; i < texels * 4; i ++ ) data[ i ] = uint8_t( std::min( 255.0f, std::floor( next() * 256.0f ) ) );
	else if ( format == Format::rgba16float ) for ( size_t i = 0; i < texels * 4; i ++ ) { const uint16_t h = floatToHalf( next() ); memcpy( &data[ i * 2 ], &h, 2 ); }
	else for ( size_t i = 0; i < texels; i ++ ) { const float f = next(); memcpy( &data[ i * 4 ], &f, 4 ); }
	return data;

}

/** One row of texels in the texture's format to RGBA float32 (r32float: r, 0, 0, 1). */
inline void decodeRow( const uint8_t * row, uint32_t width, Format format, float * out ) {

	for ( uint32_t x = 0; x < width; x ++ ) {

		float * texel = out + size_t( x ) * 4;
		if ( format == Format::rgba8unorm ) for ( int c = 0; c < 4; c ++ ) texel[ c ] = row[ x * 4 + c ] / 255.0f;
		else if ( format == Format::rgba16float ) for ( int c = 0; c < 4; c ++ ) { uint16_t h; memcpy( &h, row + x * 8 + c * 2, 2 ); texel[ c ] = halfToFloat( h ); }
		else { float f; memcpy( &f, row + x * 4, 4 ); texel[ 0 ] = f; texel[ 1 ] = 0; texel[ 2 ] = 0; texel[ 3 ] = 1; }

	}

}

/** WebGPU's full chain: floor(log2(max(w, h))) + 1 levels. */
inline uint32_t mipCount( uint32_t width, uint32_t height ) {

	uint32_t count = 1, size = std::max( width, height );
	while ( size > 1 ) { size >>= 1; count ++; }
	return count;

}

inline uint32_t mipSize( uint32_t size, uint32_t level ) { return std::max( 1u, size >> level ); }

inline std::vector<char> readFile( const std::filesystem::path & path ) {

	std::ifstream file( path, std::ios::binary );
	if ( ! file ) { fprintf( stderr, "missing %s (build the shaders first)\n", path.string().c_str() ); exit( 1 ); }
	return std::vector<char>( std::istreambuf_iterator<char>( file ), {} );

}

inline double median( std::vector<double> values ) {

	std::sort( values.begin(), values.end() );
	const size_t n = values.size();
	return n % 2 ? values[ n / 2 ] : 0.5 * ( values[ n / 2 - 1 ] + values[ n / 2 ] );

}

inline double percentile( std::vector<double> values, double p ) {

	std::sort( values.begin(), values.end() );
	return values[ std::min( values.size() - 1, size_t( p * ( values.size() - 1 ) + 0.5 ) ) ];

}

inline std::string isoDate() {

	const std::time_t now = std::time( nullptr );
	std::tm utc {};
#ifdef _WIN32
	gmtime_s( &utc, &now );
#else
	gmtime_r( &now, &utc );
#endif
	char text[ 32 ];
	std::strftime( text, sizeof( text ), "%Y-%m-%dT%H:%M:%SZ", &utc );
	return text;

}

/** <folder>/<variant>-<format>-<w>x<h>.bin (float32 RGBA, level 0 first) plus a .json index. */
inline void writeDump( const std::filesystem::path & folder, const std::string & variant, Format format, uint32_t width, uint32_t height, const std::vector<std::vector<float>> & levels ) {

	std::filesystem::create_directories( folder );
	const std::string stem = variant + "-" + formatName( format ) + "-" + std::to_string( width ) + "x" + std::to_string( height );
	std::ofstream binary( folder / ( stem + ".bin" ), std::ios::binary );
	for ( const auto & level : levels ) binary.write( ( const char * ) level.data(), std::streamsize( level.size() * sizeof( float ) ) );
	std::ofstream json( folder / ( stem + ".json" ) );
	json << "{ \"variant\": \"" << variant << "\", \"format\": \"" << formatName( format ) << "\", \"width\": " << width << ", \"height\": " << height << ", \"layout\": \"float32 RGBA, level 0 first, row-major\", \"levels\": [";
	for ( size_t level = 0; level < levels.size(); level ++ ) json << ( level ? ", " : " " ) << "[ " << mipSize( width, uint32_t( level ) ) << ", " << mipSize( height, uint32_t( level ) ) << " ]";
	json << " ] }\n";

}

/** A result row for the JSON output. */
struct Result {
	std::string variant, method, api;
	Format format;
	uint32_t width, height, levels, dispatches;
	std::vector<double> samples;
};

inline void writeResults( const std::string & path, const std::string & tool, const std::string & adapterJson, const std::string & method, int rounds, int repeats, int warmup, const std::vector<Result> & results ) {

	std::filesystem::path output( path );
	if ( output.has_parent_path() ) std::filesystem::create_directories( output.parent_path() );
	std::ofstream json( output );
	json << "{\n  \"tool\": \"" << tool << "\",\n  \"date\": \"" << isoDate() << "\",\n  \"adapter\": " << adapterJson << ",\n  \"method\": \"" << method << "\",\n"
		<< "  \"rounds\": " << rounds << ",\n  \"repeats\": " << repeats << ",\n  \"warmup\": " << warmup << ",\n  \"results\": [";
	for ( size_t r = 0; r < results.size(); r ++ ) {

		const Result & result = results[ r ];
		json << ( r ? ",\n" : "\n" ) << "    { \"variant\": \"" << result.variant << "\", \"method\": \"" << result.method << "\", \"api\": \"" << result.api << "\", \"format\": \"" << formatName( result.format )
			<< "\", \"reduction\": \"" << ( formatIsMax( result.format ) ? "max" : "average" ) << "\", \"width\": " << result.width << ", \"height\": " << result.height << ", \"levels\": " << result.levels
			<< ", \"dispatches\": " << result.dispatches << ", \"medianMs\": " << median( result.samples ) << ", \"p10Ms\": " << percentile( result.samples, 0.1 ) << ", \"p90Ms\": " << percentile( result.samples, 0.9 ) << ", \"samplesMs\": [";
		for ( size_t i = 0; i < result.samples.size(); i ++ ) json << ( i ? ", " : " " ) << result.samples[ i ];
		json << " ] }";

	}
	json << "\n  ]\n}\n";
	printf( "wrote %s\n", path.c_str() );

}

inline void printResult( const Result & result ) {

	printf( "  %-24s %8.4f ms  (p10 %.4f, p90 %.4f, %u %s)\n", result.variant.c_str(), median( result.samples ), percentile( result.samples, 0.1 ), percentile( result.samples, 0.9 ), result.dispatches, result.method == "render" || result.method == "d3d11" || result.method == "blit" ? "passes" : "dispatches" );

}

}  // namespace harness
