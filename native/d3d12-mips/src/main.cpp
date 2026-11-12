// Native D3D12 / D3D11 mip-chain timings on the same GPU as the WebGPU runs.
//
// Variants, for each size and format (see README.md):
//   spd-atomic-<wave|lds>-<load|linear>   AMD FidelityFX SPD, all levels in one dispatch with the
//                                         atomic last-workgroup handoff (AMD's structure)
//   spd-split-<wave|lds>-<load|linear>    the same shader, 6 levels per dispatch (2 dispatches up
//                                         to 4096²): the structure of our oddSizes: 'spd'
//   render                                one render pass per level, linear sampler, on a texture
//                                         that also allows UAV (as WebGPU's test textures do)
//   render-rtonly                         the same on a render-target-only texture
//   d3d11-generatemips                    ID3D11DeviceContext::GenerateMips, the driver's own path
//   empty                                 two timestamps with nothing between (timer overhead)
//
// Timing: GPU timestamps around `repeats` back-to-back chains, one sample per submit, variants in
// rotating order each round, median over `rounds` rounds. Same method as test/bench/timing.js
// (16 repeats per sample), without its marker passes.

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../common/harness.hpp"

using Microsoft::WRL::ComPtr;
using namespace harness;

#define CHECK( x ) do { HRESULT hr_ = ( x ); if ( FAILED( hr_ ) ) { fprintf( stderr, "%s failed: 0x%08lx (%s:%d)\n", #x, ( unsigned long ) hr_, __FILE__, __LINE__ ); exit( 1 ); } } while ( 0 )

// ---------------------------------------------------------------------------------------------
// Options

struct Options {
	std::string power = "high";
	std::vector<Size> sizes = { { 960, 540 }, { 1129, 635 }, { 1280, 720 }, { 1707, 960 }, { 1920, 1080 }, { 2560, 1440 }, { 1024, 1024 }, { 4096, 4096 } };
	std::vector<std::string> formats = { "rgba8unorm", "rgba16float", "r32float" };
	int rounds = 128;
	int repeats = 16;
	int warmup = 8;
	std::string output;
	std::string dump;                                   // folder for level dumps (verification)
	std::vector<Size> dumpSizes = { { 1129, 635 } };
	std::string only;                                   // comma-separated substrings of variant names
	std::string shaders;                                // folder with the .cso files
	bool debug = false;
	bool noTiming = false;
};

// ---------------------------------------------------------------------------------------------
// Formats and data

struct FormatInfo {
	const char * name;
	DXGI_FORMAT dxgi;
	uint32_t bytes;
	bool unorm;
	bool max;        // Hi-Z: max reduction, SPD only
	Format id;
};

static const FormatInfo formatInfos[] = {
	{ "rgba8unorm", DXGI_FORMAT_R8G8B8A8_UNORM, 4, true, false, Format::rgba8unorm },
	{ "rgba16float", DXGI_FORMAT_R16G16B16A16_FLOAT, 8, false, false, Format::rgba16float },
	{ "r32float", DXGI_FORMAT_R32_FLOAT, 4, false, true, Format::r32float },
};

static const FormatInfo * findFormat( const std::string & name ) {

	for ( const auto & info : formatInfos ) if ( name == info.name ) return &info;
	return nullptr;

}

// ---------------------------------------------------------------------------------------------
// D3D12 context

struct Texture {
	ComPtr<ID3D12Resource> resource;
	uint32_t width = 0, height = 0, levels = 0;
	const FormatInfo * format = nullptr;
	std::vector<D3D12_RESOURCE_STATES> state;     // per level, as recorded so far
};

struct Gpu {

	ComPtr<IDXGIAdapter1> adapter;
	ComPtr<ID3D12Device> device;
	ComPtr<ID3D12CommandQueue> queue;
	ComPtr<ID3D12CommandAllocator> allocator;
	ComPtr<ID3D12GraphicsCommandList> list;
	ComPtr<ID3D12Fence> fence;
	uint64_t fenceValue = 0;
	HANDLE event = nullptr;
	ComPtr<ID3D12DescriptorHeap> viewHeap, rtvHeap;
	uint32_t viewSize = 0, viewUsed = 0, viewCapacity = 8192;
	uint32_t rtvSize = 0, rtvUsed = 0, rtvCapacity = 1024;
	ComPtr<ID3D12QueryHeap> queryHeap;
	ComPtr<ID3D12Resource> queryReadback;
	uint64_t frequency = 0;
	ComPtr<ID3D12RootSignature> spdRoot, blitRoot;
	std::vector<D3D12_RESOURCE_BARRIER> pending;

	void begin() {

		CHECK( allocator->Reset() );
		CHECK( list->Reset( allocator.Get(), nullptr ) );
		ID3D12DescriptorHeap * heaps[] = { viewHeap.Get() };
		list->SetDescriptorHeaps( 1, heaps );

	}

	void submitAndWait() {

		CHECK( list->Close() );
		ID3D12CommandList * lists[] = { list.Get() };
		queue->ExecuteCommandLists( 1, lists );
		CHECK( queue->Signal( fence.Get(), ++ fenceValue ) );
		if ( fence->GetCompletedValue() < fenceValue ) {

			CHECK( fence->SetEventOnCompletion( fenceValue, event ) );
			WaitForSingleObject( event, INFINITE );

		}
		CHECK( device->GetDeviceRemovedReason() );
		printDebugMessages();

	}

	ComPtr<ID3D12InfoQueue> infoQueue;   // --debug only
	int debugMessages = 0;

	void printDebugMessages() {

		if ( ! infoQueue ) return;
		const UINT64 count = infoQueue->GetNumStoredMessages();
		for ( UINT64 i = 0; i < count; i ++ ) {

			SIZE_T length = 0;
			infoQueue->GetMessage( i, nullptr, &length );
			std::vector<char> storage( length );
			auto * message = reinterpret_cast<D3D12_MESSAGE *>( storage.data() );
			infoQueue->GetMessage( i, message, &length );
			if ( message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING && debugMessages ++ < 50 ) fprintf( stderr, "D3D12 debug: %.*s\n", int( message->DescriptionByteLength ), message->pDescription );

		}
		infoQueue->ClearStoredMessages();

	}

	/** Queue a transition of one level (recorded on flush()). */
	void require( Texture & texture, uint32_t level, D3D12_RESOURCE_STATES state ) {

		if ( texture.state[ level ] == state ) return;
		D3D12_RESOURCE_BARRIER barrier {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition.pResource = texture.resource.Get();
		barrier.Transition.Subresource = level;
		barrier.Transition.StateBefore = texture.state[ level ];
		barrier.Transition.StateAfter = state;
		pending.push_back( barrier );
		texture.state[ level ] = state;

	}

	void uavBarrier() {

		D3D12_RESOURCE_BARRIER barrier {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		barrier.UAV.pResource = nullptr;
		pending.push_back( barrier );

	}

	void flush() {

		if ( pending.empty() ) return;
		list->ResourceBarrier( UINT( pending.size() ), pending.data() );
		pending.clear();

	}

	uint32_t allocateViews( uint32_t count ) {

		if ( viewUsed + count > viewCapacity ) { fprintf( stderr, "descriptor heap full\n" ); exit( 1 ); }
		const uint32_t first = viewUsed;
		viewUsed += count;
		return first;

	}

	D3D12_CPU_DESCRIPTOR_HANDLE viewCpu( uint32_t index ) { auto handle = viewHeap->GetCPUDescriptorHandleForHeapStart(); handle.ptr += size_t( index ) * viewSize; return handle; }
	D3D12_GPU_DESCRIPTOR_HANDLE viewGpu( uint32_t index ) { auto handle = viewHeap->GetGPUDescriptorHandleForHeapStart(); handle.ptr += uint64_t( index ) * viewSize; return handle; }
	D3D12_CPU_DESCRIPTOR_HANDLE rtvCpu( uint32_t index ) { auto handle = rtvHeap->GetCPUDescriptorHandleForHeapStart(); handle.ptr += size_t( index ) * rtvSize; return handle; }

	void srv( uint32_t index, Texture & texture, uint32_t level ) {

		D3D12_SHADER_RESOURCE_VIEW_DESC desc {};
		desc.Format = texture.format->dxgi;
		desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		desc.Texture2D.MostDetailedMip = level;
		desc.Texture2D.MipLevels = 1;
		device->CreateShaderResourceView( texture.resource.Get(), &desc, viewCpu( index ) );

	}

	void uav( uint32_t index, Texture & texture, int level ) {

		D3D12_UNORDERED_ACCESS_VIEW_DESC desc {};
		desc.Format = texture.format->dxgi;
		desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		desc.Texture2D.MipSlice = level < 0 ? 0 : UINT( level );
		device->CreateUnorderedAccessView( level < 0 ? nullptr : texture.resource.Get(), nullptr, &desc, viewCpu( index ) );

	}

	Texture createTexture( uint32_t width, uint32_t height, const FormatInfo & format, D3D12_RESOURCE_FLAGS flags ) {

		Texture texture;
		texture.width = width; texture.height = height; texture.levels = mipCount( width, height ); texture.format = &format;
		D3D12_RESOURCE_DESC desc {};
		desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		desc.Width = width; desc.Height = height; desc.DepthOrArraySize = 1; desc.MipLevels = UINT16( texture.levels );
		desc.Format = format.dxgi; desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; desc.Flags = flags;
		D3D12_HEAP_PROPERTIES heap { D3D12_HEAP_TYPE_DEFAULT };
		CHECK( device->CreateCommittedResource( &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS( &texture.resource ) ) );
		texture.state.assign( texture.levels, D3D12_RESOURCE_STATE_COPY_DEST );
		return texture;

	}

	ComPtr<ID3D12Resource> createBuffer( uint64_t size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE ) {

		ComPtr<ID3D12Resource> buffer;
		D3D12_RESOURCE_DESC desc {};
		desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		desc.Width = size; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.SampleDesc.Count = 1;
		desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = flags;
		D3D12_HEAP_PROPERTIES heap { type };
		CHECK( device->CreateCommittedResource( &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS( &buffer ) ) );
		return buffer;

	}

	void upload( Texture & texture, const std::vector<uint8_t> & data ) {

		D3D12_RESOURCE_DESC desc = texture.resource->GetDesc();
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout; UINT rows; UINT64 rowBytes, total;
		device->GetCopyableFootprints( &desc, 0, 1, 0, &layout, &rows, &rowBytes, &total );
		auto staging = createBuffer( total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ );
		uint8_t * mapped;
		CHECK( staging->Map( 0, nullptr, ( void ** ) &mapped ) );
		const size_t tight = size_t( texture.width ) * texture.format->bytes;
		for ( UINT y = 0; y < rows; y ++ ) memcpy( mapped + layout.Offset + size_t( y ) * layout.Footprint.RowPitch, data.data() + size_t( y ) * tight, tight );
		staging->Unmap( 0, nullptr );
		begin();
		require( texture, 0, D3D12_RESOURCE_STATE_COPY_DEST );
		flush();
		D3D12_TEXTURE_COPY_LOCATION dst { texture.resource.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX }; dst.SubresourceIndex = 0;
		D3D12_TEXTURE_COPY_LOCATION src { staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT }; src.PlacedFootprint = layout;
		list->CopyTextureRegion( &dst, 0, 0, 0, &src, nullptr );
		submitAndWait();

	}

	/** Every level as RGBA float32, level 0 first. */
	std::vector<std::vector<float>> readLevels( Texture & texture ) {

		D3D12_RESOURCE_DESC desc = texture.resource->GetDesc();
		std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts( texture.levels );
		std::vector<UINT> rows( texture.levels ); std::vector<UINT64> rowBytes( texture.levels ); UINT64 total;
		device->GetCopyableFootprints( &desc, 0, texture.levels, 0, layouts.data(), rows.data(), rowBytes.data(), &total );
		auto readback = createBuffer( total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST );
		begin();
		std::vector<D3D12_RESOURCE_STATES> before = texture.state;
		for ( uint32_t level = 0; level < texture.levels; level ++ ) require( texture, level, D3D12_RESOURCE_STATE_COPY_SOURCE );
		flush();
		for ( uint32_t level = 0; level < texture.levels; level ++ ) {

			D3D12_TEXTURE_COPY_LOCATION src { texture.resource.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX }; src.SubresourceIndex = level;
			D3D12_TEXTURE_COPY_LOCATION dst { readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT }; dst.PlacedFootprint = layouts[ level ];
			list->CopyTextureRegion( &dst, 0, 0, 0, &src, nullptr );

		}
		for ( uint32_t level = 0; level < texture.levels; level ++ ) require( texture, level, before[ level ] );
		flush();
		submitAndWait();
		uint8_t * mapped;
		CHECK( readback->Map( 0, nullptr, ( void ** ) &mapped ) );
		std::vector<std::vector<float>> levels( texture.levels );
		for ( uint32_t level = 0; level < texture.levels; level ++ ) {

			const uint32_t w = mipSize( texture.width, level ), h = mipSize( texture.height, level );
			levels[ level ].resize( size_t( w ) * h * 4 );
			for ( uint32_t y = 0; y < h; y ++ ) decodeRow( mapped + layouts[ level ].Offset + size_t( y ) * layouts[ level ].Footprint.RowPitch, w, texture.format->id, levels[ level ].data() + size_t( y ) * w * 4 );

		}
		readback->Unmap( 0, nullptr );
		return levels;

	}

};

static ComPtr<ID3DBlob> serializeRoot( const D3D12_ROOT_SIGNATURE_DESC & desc ) {

	ComPtr<ID3DBlob> blob, error;
	if ( FAILED( D3D12SerializeRootSignature( &desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error ) ) ) {

		fprintf( stderr, "root signature: %s\n", error ? ( const char * ) error->GetBufferPointer() : "?" );
		exit( 1 );

	}
	return blob;

}

static D3D12_STATIC_SAMPLER_DESC linearClampSampler() {

	D3D12_STATIC_SAMPLER_DESC sampler {};
	sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
	sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
	sampler.MaxLOD = D3D12_FLOAT32_MAX;
	sampler.ShaderRegister = 0;
	sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	return sampler;

}

static void createRoots( Gpu & gpu ) {

	{

		// SPD: 8 root constants (b0); table of t0 + u0..u13; static linear clamp sampler.
		D3D12_DESCRIPTOR_RANGE ranges[ 2 ] {};
		ranges[ 0 ] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0 };
		ranges[ 1 ] = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 14, 0, 0, 1 };
		D3D12_ROOT_PARAMETER params[ 2 ] {};
		params[ 0 ].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		params[ 0 ].Constants = { 0, 0, 8 };
		params[ 0 ].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		params[ 1 ].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		params[ 1 ].DescriptorTable = { 2, ranges };
		params[ 1 ].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		const auto sampler = linearClampSampler();
		D3D12_ROOT_SIGNATURE_DESC desc { 2, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE };
		auto blob = serializeRoot( desc );
		CHECK( gpu.device->CreateRootSignature( 0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS( &gpu.spdRoot ) ) );

	}
	{

		D3D12_DESCRIPTOR_RANGE range { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0 };
		D3D12_ROOT_PARAMETER param {};
		param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		param.DescriptorTable = { 1, &range };
		param.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
		auto sampler = linearClampSampler();
		sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
		D3D12_ROOT_SIGNATURE_DESC desc { 1, &param, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE };
		auto blob = serializeRoot( desc );
		CHECK( gpu.device->CreateRootSignature( 0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS( &gpu.blitRoot ) ) );

	}

}

// ---------------------------------------------------------------------------------------------
// Adapter, devices

static std::string narrow( const wchar_t * text ) {

	char buffer[ 512 ];
	WideCharToMultiByte( CP_UTF8, 0, text, -1, buffer, sizeof( buffer ), nullptr, nullptr );
	return buffer;

}

struct AdapterInfo { std::string name, driver; uint32_t vendor = 0, device = 0; };

static AdapterInfo describe( IDXGIAdapter1 * adapter ) {

	DXGI_ADAPTER_DESC1 desc;
	adapter->GetDesc1( &desc );
	AdapterInfo info;
	info.name = narrow( desc.Description );
	info.vendor = desc.VendorId; info.device = desc.DeviceId;
	LARGE_INTEGER umd;
	if ( SUCCEEDED( adapter->CheckInterfaceSupport( __uuidof( IDXGIDevice ), &umd ) ) ) {

		char text[ 64 ];
		snprintf( text, sizeof( text ), "%u.%u.%u.%u", HIWORD( umd.HighPart ), LOWORD( umd.HighPart ), HIWORD( umd.LowPart ), LOWORD( umd.LowPart ) );
		info.driver = text;

	}
	return info;

}

static Gpu createGpu( const Options & options ) {

	Gpu gpu;
	if ( options.debug ) {

		ComPtr<ID3D12Debug> debug;
		if ( SUCCEEDED( D3D12GetDebugInterface( IID_PPV_ARGS( &debug ) ) ) ) debug->EnableDebugLayer();

	}
	ComPtr<IDXGIFactory6> factory;
	CHECK( CreateDXGIFactory2( 0, IID_PPV_ARGS( &factory ) ) );
	const auto preference = options.power == "low" ? DXGI_GPU_PREFERENCE_MINIMUM_POWER : DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
	CHECK( factory->EnumAdapterByGpuPreference( 0, preference, IID_PPV_ARGS( &gpu.adapter ) ) );
	CHECK( D3D12CreateDevice( gpu.adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS( &gpu.device ) ) );
	if ( options.debug ) {

		if ( SUCCEEDED( gpu.device.As( &gpu.infoQueue ) ) ) printf( "D3D12 debug layer on\n" );
		else printf( "D3D12 debug layer unavailable (install the Graphics Tools optional feature)\n" );

	}

	D3D12_COMMAND_QUEUE_DESC queueDesc { D3D12_COMMAND_LIST_TYPE_DIRECT };
	CHECK( gpu.device->CreateCommandQueue( &queueDesc, IID_PPV_ARGS( &gpu.queue ) ) );
	CHECK( gpu.device->CreateCommandAllocator( D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS( &gpu.allocator ) ) );
	CHECK( gpu.device->CreateCommandList( 0, D3D12_COMMAND_LIST_TYPE_DIRECT, gpu.allocator.Get(), nullptr, IID_PPV_ARGS( &gpu.list ) ) );
	CHECK( gpu.list->Close() );
	CHECK( gpu.device->CreateFence( 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS( &gpu.fence ) ) );
	gpu.event = CreateEvent( nullptr, FALSE, FALSE, nullptr );

	D3D12_DESCRIPTOR_HEAP_DESC viewDesc { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, gpu.viewCapacity, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE };
	CHECK( gpu.device->CreateDescriptorHeap( &viewDesc, IID_PPV_ARGS( &gpu.viewHeap ) ) );
	gpu.viewSize = gpu.device->GetDescriptorHandleIncrementSize( D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV );
	D3D12_DESCRIPTOR_HEAP_DESC rtvDesc { D3D12_DESCRIPTOR_HEAP_TYPE_RTV, gpu.rtvCapacity };
	CHECK( gpu.device->CreateDescriptorHeap( &rtvDesc, IID_PPV_ARGS( &gpu.rtvHeap ) ) );
	gpu.rtvSize = gpu.device->GetDescriptorHandleIncrementSize( D3D12_DESCRIPTOR_HEAP_TYPE_RTV );

	D3D12_QUERY_HEAP_DESC queryDesc { D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2 };
	CHECK( gpu.device->CreateQueryHeap( &queryDesc, IID_PPV_ARGS( &gpu.queryHeap ) ) );
	gpu.queryReadback = gpu.createBuffer( 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST );
	CHECK( gpu.queue->GetTimestampFrequency( &gpu.frequency ) );
	createRoots( gpu );
	return gpu;

}

struct D3D11Context {
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> context;
	ComPtr<ID3D11Query> disjoint, start, end;
};

static D3D11Context createD3D11( IDXGIAdapter1 * adapter ) {

	D3D11Context d3d11;
	D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
	CHECK( D3D11CreateDevice( adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, &level, 1, D3D11_SDK_VERSION, &d3d11.device, nullptr, &d3d11.context ) );
	D3D11_QUERY_DESC disjoint { D3D11_QUERY_TIMESTAMP_DISJOINT }, stamp { D3D11_QUERY_TIMESTAMP };
	CHECK( d3d11.device->CreateQuery( &disjoint, &d3d11.disjoint ) );
	CHECK( d3d11.device->CreateQuery( &stamp, &d3d11.start ) );
	CHECK( d3d11.device->CreateQuery( &stamp, &d3d11.end ) );
	return d3d11;

}

// ---------------------------------------------------------------------------------------------
// Variants

struct Variant {
	std::string name;
	std::string method;       // spd, render, d3d11, empty
	uint32_t dispatches = 0;  // dispatches or passes per chain
	Texture texture;          // D3D12 variants
	std::function<void()> encode;   // one chain, recorded into gpu.list (D3D12) or issued (D3D11)
	bool d3d11 = false;
	ComPtr<ID3D11Texture2D> texture11;
	ComPtr<ID3D11ShaderResourceView> view11;
	std::vector<double> samples;
};

struct Pipelines {
	std::map<std::string, ComPtr<ID3D12PipelineState>> spd;
	std::map<DXGI_FORMAT, ComPtr<ID3D12PipelineState>> blit;
};

static std::string spdShaderName( bool wave, bool linear, const FormatInfo & format ) {

	if ( format.max ) return std::string( "spd-max-" ) + ( wave ? "wave" : "lds" );
	return std::string( "spd-" ) + ( wave ? "wave" : "lds" ) + "-" + ( linear ? "linear" : "load" ) + "-" + ( format.unorm ? "unorm" : "float" );

}

static ID3D12PipelineState * spdPipeline( Gpu & gpu, Pipelines & pipelines, const std::filesystem::path & folder, const std::string & name ) {

	auto & pso = pipelines.spd[ name ];
	if ( ! pso ) {

		const auto code = readFile( folder / ( name + ".cso" ) );
		D3D12_COMPUTE_PIPELINE_STATE_DESC desc {};
		desc.pRootSignature = gpu.spdRoot.Get();
		desc.CS = { code.data(), code.size() };
		CHECK( gpu.device->CreateComputePipelineState( &desc, IID_PPV_ARGS( &pso ) ) );

	}
	return pso.Get();

}

static ID3D12PipelineState * blitPipeline( Gpu & gpu, Pipelines & pipelines, const std::filesystem::path & folder, DXGI_FORMAT format ) {

	auto & pso = pipelines.blit[ format ];
	if ( ! pso ) {

		const auto vs = readFile( folder / "blit-vs.cso" ), ps = readFile( folder / "blit-ps.cso" );
		D3D12_GRAPHICS_PIPELINE_STATE_DESC desc {};
		desc.pRootSignature = gpu.blitRoot.Get();
		desc.VS = { vs.data(), vs.size() };
		desc.PS = { ps.data(), ps.size() };
		desc.BlendState.RenderTarget[ 0 ].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
		desc.SampleMask = UINT_MAX;
		desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
		desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
		desc.RasterizerState.DepthClipEnable = TRUE;
		desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		desc.NumRenderTargets = 1;
		desc.RTVFormats[ 0 ] = format;
		desc.SampleDesc.Count = 1;
		CHECK( gpu.device->CreateGraphicsPipelineState( &desc, IID_PPV_ARGS( &pso ) ) );

	}
	return pso.Get();

}

struct SpdDispatch {
	uint32_t table;          // first descriptor: t0, u0, u1 (level base + 6), u2.. (levels base + 1 …)
	uint32_t constants[ 8 ];
	uint32_t groupsX, groupsY;
	uint32_t base;           // source level
};

/** SPD over `texture`, from `base` for `mips` levels: AMD's ffxSpdSetup for the whole rect. */
static SpdDispatch spdDispatch( Gpu & gpu, Texture & texture, ComPtr<ID3D12Resource> & atomic, uint32_t base, uint32_t mips ) {

	SpdDispatch dispatch {};
	dispatch.base = base;
	const uint32_t width = mipSize( texture.width, base ), height = mipSize( texture.height, base );
	dispatch.groupsX = ( width - 1 ) / 64 + 1;
	dispatch.groupsY = ( height - 1 ) / 64 + 1;
	const float inverse[ 2 ] = { 1.0f / width, 1.0f / height };
	dispatch.constants[ 0 ] = mips;
	dispatch.constants[ 1 ] = dispatch.groupsX * dispatch.groupsY;
	memcpy( &dispatch.constants[ 4 ], inverse, 8 );

	dispatch.table = gpu.allocateViews( 15 );
	gpu.srv( dispatch.table, texture, base );
	D3D12_UNORDERED_ACCESS_VIEW_DESC counter {};
	counter.Format = DXGI_FORMAT_UNKNOWN;
	counter.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	counter.Buffer.NumElements = 1;
	counter.Buffer.StructureByteStride = 4;
	gpu.device->CreateUnorderedAccessView( atomic.Get(), nullptr, &counter, gpu.viewCpu( dispatch.table + 1 ) );
	const auto level = [ & ]( uint32_t l ) { return l <= base + mips && l < texture.levels ? int( l ) : -1; };
	gpu.uav( dispatch.table + 2, texture, level( base + 6 ) );
	for ( uint32_t i = 0; i < 12; i ++ ) gpu.uav( dispatch.table + 3 + i, texture, level( base + 1 + i ) );
	return dispatch;

}

// ---------------------------------------------------------------------------------------------
// Timing

static double timeD3D12( Gpu & gpu, Variant & variant, int repeats ) {

	gpu.begin();
	gpu.list->EndQuery( gpu.queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0 );
	for ( int i = 0; i < repeats; i ++ ) if ( variant.encode ) variant.encode();
	gpu.flush();
	gpu.list->EndQuery( gpu.queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1 );
	gpu.list->ResolveQueryData( gpu.queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, gpu.queryReadback.Get(), 0 );
	gpu.submitAndWait();
	uint64_t * stamps;
	D3D12_RANGE range { 0, 16 };
	CHECK( gpu.queryReadback->Map( 0, &range, ( void ** ) &stamps ) );
	const uint64_t start = stamps[ 0 ], end = stamps[ 1 ];
	gpu.queryReadback->Unmap( 0, nullptr );
	if ( end < start ) { fprintf( stderr, "invalid timestamps %llu %llu\n", start, end ); exit( 1 ); }
	return double( end - start ) / double( gpu.frequency ) * 1000.0 / repeats;

}

static double timeD3D11( D3D11Context & d3d11, Variant & variant, int repeats ) {

	for ( int attempt = 0; attempt < 10; attempt ++ ) {

		d3d11.context->Begin( d3d11.disjoint.Get() );
		d3d11.context->End( d3d11.start.Get() );
		for ( int i = 0; i < repeats; i ++ ) variant.encode();
		d3d11.context->End( d3d11.end.Get() );
		d3d11.context->End( d3d11.disjoint.Get() );
		D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint;
		while ( d3d11.context->GetData( d3d11.disjoint.Get(), &disjoint, sizeof( disjoint ), 0 ) == S_FALSE ) SwitchToThread();
		UINT64 start, end;
		while ( d3d11.context->GetData( d3d11.start.Get(), &start, sizeof( start ), 0 ) == S_FALSE ) SwitchToThread();
		while ( d3d11.context->GetData( d3d11.end.Get(), &end, sizeof( end ), 0 ) == S_FALSE ) SwitchToThread();
		if ( ! disjoint.Disjoint && end >= start ) return double( end - start ) / double( disjoint.Frequency ) * 1000.0 / repeats;

	}
	fprintf( stderr, "D3D11 timestamps stayed disjoint\n" );
	exit( 1 );

}

/** D3D11 levels as RGBA float32 (for the dump). */
static std::vector<std::vector<float>> readLevels11( D3D11Context & d3d11, Variant & variant, uint32_t width, uint32_t height, const FormatInfo & format ) {

	std::vector<std::vector<float>> levels;
	const uint32_t count = mipCount( width, height );
	for ( uint32_t level = 0; level < count; level ++ ) {

		const uint32_t w = mipSize( width, level ), h = mipSize( height, level );
		D3D11_TEXTURE2D_DESC desc {};
		desc.Width = w; desc.Height = h; desc.MipLevels = 1; desc.ArraySize = 1; desc.Format = format.dxgi;
		desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ComPtr<ID3D11Texture2D> staging;
		CHECK( d3d11.device->CreateTexture2D( &desc, nullptr, &staging ) );
		d3d11.context->CopySubresourceRegion( staging.Get(), 0, 0, 0, 0, variant.texture11.Get(), level, nullptr );
		D3D11_MAPPED_SUBRESOURCE mapped;
		CHECK( d3d11.context->Map( staging.Get(), 0, D3D11_MAP_READ, 0, &mapped ) );
		std::vector<float> values( size_t( w ) * h * 4 );
		for ( uint32_t y = 0; y < h; y ++ ) decodeRow( ( const uint8_t * ) mapped.pData + size_t( y ) * mapped.RowPitch, w, format.id, values.data() + size_t( y ) * w * 4 );
		d3d11.context->Unmap( staging.Get(), 0 );
		levels.push_back( std::move( values ) );

	}
	return levels;

}

// ---------------------------------------------------------------------------------------------

static bool selected( const Options & options, const std::string & name ) {

	if ( options.only.empty() || name == "empty" ) return true;
	for ( const auto & part : split( options.only ) ) if ( name.find( part ) != std::string::npos ) return true;
	return false;

}

int main( int argc, char ** argv ) {

	Options options;
	options.shaders = ( std::filesystem::path( argv[ 0 ] ).parent_path() / "shaders" ).string();
	for ( int i = 1; i < argc; i ++ ) {

		const std::string arg = argv[ i ];
		const auto value = [ & ]( const char * key ) -> const char * { const size_t n = strlen( key ); return arg.compare( 0, n, key ) == 0 ? arg.c_str() + n : nullptr; };
		if ( auto v = value( "--power=" ) ) options.power = v;
		else if ( auto v = value( "--sizes=" ) ) options.sizes = parseSizes( v );
		else if ( auto v = value( "--formats=" ) ) options.formats = split( v );
		else if ( auto v = value( "--rounds=" ) ) options.rounds = atoi( v );
		else if ( auto v = value( "--repeats=" ) ) options.repeats = atoi( v );
		else if ( auto v = value( "--warmup=" ) ) options.warmup = atoi( v );
		else if ( auto v = value( "--output=" ) ) options.output = v;
		else if ( auto v = value( "--dump=" ) ) options.dump = v;
		else if ( auto v = value( "--dump-sizes=" ) ) options.dumpSizes = parseSizes( v );
		else if ( auto v = value( "--only=" ) ) options.only = v;
		else if ( auto v = value( "--shaders=" ) ) options.shaders = v;
		else if ( arg == "--debug" ) options.debug = true;
		else if ( arg == "--no-timing" ) options.noTiming = true;
		else { fprintf( stderr, "unknown option %s\n", arg.c_str() ); return 1; }

	}

	Gpu gpu = createGpu( options );
	D3D11Context d3d11 = createD3D11( gpu.adapter.Get() );
	const AdapterInfo adapter = describe( gpu.adapter.Get() );
	printf( "adapter: %s (vendor 0x%04x device 0x%04x), driver %s, timestamp frequency %llu Hz\n", adapter.name.c_str(), adapter.vendor, adapter.device, adapter.driver.c_str(), gpu.frequency );
	{

		D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 {};
		gpu.device->CheckFeatureSupport( D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof( options1 ) );
		printf( "wave lanes: %u-%u\n", options1.WaveLaneCountMin, options1.WaveLaneCountMax );

	}
	fflush( stdout );

	const std::filesystem::path shaderFolder = options.shaders;
	Pipelines pipelines;
	auto atomic = gpu.createBuffer( 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS );
	{

		// Zero the counter explicitly (committed resources are zeroed, but don't rely on it).
		auto zero = gpu.createBuffer( 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ );
		void * mapped; CHECK( zero->Map( 0, nullptr, &mapped ) ); memset( mapped, 0, 256 ); zero->Unmap( 0, nullptr );
		gpu.begin();
		D3D12_RESOURCE_BARRIER barrier {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Transition = { atomic.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST };
		gpu.list->ResourceBarrier( 1, &barrier );
		gpu.list->CopyBufferRegion( atomic.Get(), 0, zero.Get(), 0, 256 );
		std::swap( barrier.Transition.StateBefore, barrier.Transition.StateAfter );
		gpu.list->ResourceBarrier( 1, &barrier );
		gpu.submitAndWait();

	}

	std::vector<Result> results;

	for ( const auto & formatName : options.formats ) {

		const FormatInfo * formatPointer = findFormat( formatName );
		if ( ! formatPointer ) { fprintf( stderr, "unknown format %s\n", formatName.c_str() ); return 1; }
		const FormatInfo & format = *formatPointer;

		for ( const Size size : options.sizes ) {

			gpu.viewUsed = 0;
			gpu.rtvUsed = 0;
			const auto data = makeData( size.width, size.height, format.id );
			const uint32_t levels = mipCount( size.width, size.height ), outputs = levels - 1;
			std::vector<Variant> variants;

			// AMD SPD: atomic single dispatch and 6 levels per dispatch, wave or LDS, load or linear sampler.
			for ( const bool atomicHandoff : { true, false } ) for ( const bool wave : { true, false } ) for ( const bool linear : { false, true } ) {

				if ( format.max && linear ) continue;
				std::string name = std::string( "spd-" ) + ( atomicHandoff ? "atomic" : "split" ) + "-" + ( wave ? "wave" : "lds" ) + ( format.max ? "" : linear ? "-linear" : "-load" );
				if ( ! selected( options, name ) ) continue;
				Variant variant;
				variant.name = name;
				variant.method = "spd";
				variant.texture = gpu.createTexture( size.width, size.height, format, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | ( format.max ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ) );
				gpu.upload( variant.texture, data );
				std::vector<SpdDispatch> dispatches;
				if ( atomicHandoff ) dispatches.push_back( spdDispatch( gpu, variant.texture, atomic, 0, outputs ) );
				else for ( uint32_t base = 0; base < outputs; base += 6 ) dispatches.push_back( spdDispatch( gpu, variant.texture, atomic, base, std::min( 6u, outputs - base ) ) );
				variant.dispatches = uint32_t( dispatches.size() );
				ID3D12PipelineState * pso = spdPipeline( gpu, pipelines, shaderFolder, spdShaderName( wave, linear, format ) );
				variants.push_back( std::move( variant ) );
				// Captures the vector and an index: later push_backs move the variants.
				variants.back().encode = [ &gpu, pso, dispatches, index = variants.size() - 1, &variants ]() {

					Texture & tex = variants[ index ].texture;
					gpu.require( tex, 0, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE );
					for ( uint32_t level = 1; level < tex.levels; level ++ ) gpu.require( tex, level, D3D12_RESOURCE_STATE_UNORDERED_ACCESS );
					gpu.flush();
					gpu.list->SetComputeRootSignature( gpu.spdRoot.Get() );
					gpu.list->SetPipelineState( pso );
					for ( const auto & dispatch : dispatches ) {

						if ( dispatch.base > 0 ) {

							gpu.uavBarrier();
							gpu.require( tex, dispatch.base, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE );
							gpu.flush();

						}
						gpu.list->SetComputeRoot32BitConstants( 0, 8, dispatch.constants, 0 );
						gpu.list->SetComputeRootDescriptorTable( 1, gpu.viewGpu( dispatch.table ) );
						gpu.list->Dispatch( dispatch.groupsX, dispatch.groupsY, 1 );

					}
					gpu.uavBarrier();

				};

			}

			if ( ! format.max ) {

				// Render passes: one per level, linear sampler.
				for ( const bool withUav : { true, false } ) {

					const std::string name = withUav ? "render" : "render-rtonly";
					if ( ! selected( options, name ) ) continue;
					Variant variant;
					variant.name = name;
					variant.method = "render";
					variant.dispatches = outputs;
					variant.texture = gpu.createTexture( size.width, size.height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | ( withUav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE ) );
					gpu.upload( variant.texture, data );
					const uint32_t firstSrv = gpu.allocateViews( levels ), firstRtv = gpu.rtvUsed;
					gpu.rtvUsed += levels;
					for ( uint32_t level = 0; level < levels; level ++ ) {

						gpu.srv( firstSrv + level, variant.texture, level );
						D3D12_RENDER_TARGET_VIEW_DESC rtv {};
						rtv.Format = format.dxgi;
						rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
						rtv.Texture2D.MipSlice = level;
						gpu.device->CreateRenderTargetView( variant.texture.resource.Get(), &rtv, gpu.rtvCpu( firstRtv + level ) );

					}
					ID3D12PipelineState * pso = blitPipeline( gpu, pipelines, shaderFolder, format.dxgi );
					variants.push_back( std::move( variant ) );
					variants.back().encode = [ &gpu, pso, firstSrv, firstRtv, index = variants.size() - 1, &variants ]() {

						Texture & tex = variants[ index ].texture;
						gpu.require( tex, 0, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE );
						for ( uint32_t level = 1; level < tex.levels; level ++ ) gpu.require( tex, level, D3D12_RESOURCE_STATE_RENDER_TARGET );
						gpu.flush();
						gpu.list->SetGraphicsRootSignature( gpu.blitRoot.Get() );
						gpu.list->SetPipelineState( pso );
						gpu.list->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
						for ( uint32_t level = 1; level < tex.levels; level ++ ) {

							if ( level > 1 ) { gpu.require( tex, level - 1, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ); gpu.flush(); }
							const D3D12_CPU_DESCRIPTOR_HANDLE target = gpu.rtvCpu( firstRtv + level );
							gpu.list->OMSetRenderTargets( 1, &target, FALSE, nullptr );
							const D3D12_VIEWPORT viewport { 0, 0, float( mipSize( tex.width, level ) ), float( mipSize( tex.height, level ) ), 0, 1 };
							const D3D12_RECT scissor { 0, 0, LONG( mipSize( tex.width, level ) ), LONG( mipSize( tex.height, level ) ) };
							gpu.list->RSSetViewports( 1, &viewport );
							gpu.list->RSSetScissorRects( 1, &scissor );
							gpu.list->SetGraphicsRootDescriptorTable( 0, gpu.viewGpu( firstSrv + level - 1 ) );
							gpu.list->DrawInstanced( 3, 1, 0, 0 );

						}

					};

				}

				if ( selected( options, "d3d11-generatemips" ) ) {

					Variant variant;
					variant.name = "d3d11-generatemips";
					variant.method = "d3d11";
					variant.d3d11 = true;
					variant.dispatches = outputs;
					D3D11_TEXTURE2D_DESC desc {};
					desc.Width = size.width; desc.Height = size.height; desc.MipLevels = levels; desc.ArraySize = 1; desc.Format = format.dxgi;
					desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
					desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
					CHECK( d3d11.device->CreateTexture2D( &desc, nullptr, &variant.texture11 ) );
					d3d11.context->UpdateSubresource( variant.texture11.Get(), 0, nullptr, data.data(), size.width * format.bytes, 0 );
					CHECK( d3d11.device->CreateShaderResourceView( variant.texture11.Get(), nullptr, &variant.view11 ) );
					variants.push_back( std::move( variant ) );
					variants.back().encode = [ &d3d11, index = variants.size() - 1, &variants ]() { d3d11.context->GenerateMips( variants[ index ].view11.Get() ); };

				}

			}

			{

				Variant variant;
				variant.name = "empty";
				variant.method = "empty";
				variants.push_back( std::move( variant ) );

			}

			// Warm up (also settles every texture into its working states).
			for ( int i = 0; i < options.warmup; i ++ ) for ( auto & variant : variants ) {

				if ( variant.d3d11 ) timeD3D11( d3d11, variant, 1 );
				else timeD3D12( gpu, variant, 1 );

			}

			// Level dumps for the semantics check.
			if ( ! options.dump.empty() ) for ( const Size dumpSize : options.dumpSizes ) if ( dumpSize.width == size.width && dumpSize.height == size.height ) {

				for ( auto & variant : variants ) {

					if ( variant.method == "empty" ) continue;
					auto dumped = variant.d3d11 ? readLevels11( d3d11, variant, size.width, size.height, format ) : gpu.readLevels( variant.texture );
					writeDump( options.dump, variant.name, format.id, size.width, size.height, dumped );

				}
				printf( "dumped %s %ux%u\n", format.name, size.width, size.height );

			}

			if ( options.noTiming ) continue;

			for ( int round = 0; round < options.rounds; round ++ ) for ( size_t k = 0; k < variants.size(); k ++ ) {

				Variant & variant = variants[ ( k + round ) % variants.size() ];
				variant.samples.push_back( variant.d3d11 ? timeD3D11( d3d11, variant, options.repeats ) : timeD3D12( gpu, variant, options.repeats ) );

			}

			printf( "%s %ux%u (%u levels)\n", format.name, size.width, size.height, outputs );
			for ( auto & variant : variants ) {

				results.push_back( { variant.name, variant.method, variant.d3d11 ? "d3d11" : "d3d12", format.id, size.width, size.height, outputs, variant.dispatches, variant.samples } );
				printResult( results.back() );

			}
			fflush( stdout );

		}

	}

	if ( ! options.output.empty() ) {

		std::ostringstream adapterJson;
		adapterJson << "{ \"api\": \"D3D12 (+ D3D11 for GenerateMips)\", \"name\": \"" << adapter.name << "\", \"vendor\": " << adapter.vendor << ", \"device\": " << adapter.device << ", \"driver\": \"" << adapter.driver << "\", \"timestampFrequency\": " << gpu.frequency << " }";
		const std::string method = "GPU timestamps around " + std::to_string( options.repeats ) + " back-to-back chains, one submit per sample, variants rotated each round, median over " + std::to_string( options.rounds ) + " rounds; ms per chain";
		writeResults( options.output, "native/d3d12-mips", adapterJson.str(), method, options.rounds, options.repeats, options.warmup, results );

	}
	return 0;

}
