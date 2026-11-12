// Native Vulkan mip-chain timings on the same GPU as the WebGPU runs.
//
// Variants, for each size and format (see README.md):
//   nvpro-pyramid    NVIDIA's nvpro_pyramid (vk_compute_mipmaps), fast + general pipelines as its
//                    default dispatcher picks them; area-weighted odd sizes (our planned 'box')
//   nvpro-general    the general pipeline only (2 levels per dispatch), for reference
//   blit             vkCmdBlitImage per level with a linear filter: Vulkan's driver path
//   empty            two timestamps with nothing between (timer overhead)
//
// Timing as native/d3d12-mips: GPU timestamps around `repeats` back-to-back chains, one submit per
// sample, variants in rotating order each round, median over `rounds` rounds.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "../../common/harness.hpp"
#include "nvpro_pyramid_dispatch.hpp"

using namespace harness;

#define VK_CHECK( x ) do { VkResult r_ = ( x ); if ( r_ != VK_SUCCESS ) { fprintf( stderr, "%s failed: %d (%s:%d)\n", #x, int( r_ ), __FILE__, __LINE__ ); exit( 1 ); } } while ( 0 )

struct Options {
	std::string power = "high";
	std::string gpu;                                    // part of the device name, overrides --power
	std::vector<Size> sizes = { { 960, 540 }, { 1129, 635 }, { 1280, 720 }, { 1707, 960 }, { 1920, 1080 }, { 2560, 1440 }, { 1024, 1024 }, { 4096, 4096 } };
	std::vector<std::string> formats = { "rgba8unorm", "rgba16float", "r32float" };
	int rounds = 128;
	int repeats = 16;
	int warmup = 8;
	std::string output;
	std::string dump;
	std::vector<Size> dumpSizes = { { 1129, 635 } };
	std::string only;
	std::string shaders;
	bool validation = false;
	bool noTiming = false;
};

static VkFormat vkFormat( Format format ) {

	return format == Format::rgba8unorm ? VK_FORMAT_R8G8B8A8_UNORM : format == Format::rgba16float ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32_SFLOAT;

}

static const char * shaderFormat( Format format ) { return format == Format::rgba8unorm ? "rgba8" : format == Format::rgba16float ? "rgba16f" : "r32f-max"; }

// ---------------------------------------------------------------------------------------------

struct Image {
	VkImage image = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkImageView sampledView = VK_NULL_HANDLE;       // all levels
	std::vector<VkImageView> levelViews;            // one per level
	VkDescriptorSet sampledSet = VK_NULL_HANDLE, storageSet = VK_NULL_HANDLE;
	uint32_t width = 0, height = 0, levels = 0;
	Format format = Format::rgba8unorm;
	std::vector<VkImageLayout> layout;              // per level, as recorded so far
};

struct Vk {

	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	uint32_t family = 0;
	VkQueue queue = VK_NULL_HANDLE;
	VkCommandPool pool = VK_NULL_HANDLE;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VkFence fence = VK_NULL_HANDLE;
	VkQueryPool queries = VK_NULL_HANDLE;
	double timestampPeriod = 1;   // ns per tick
	VkPhysicalDeviceMemoryProperties memoryProperties {};
	VkSampler sampler = VK_NULL_HANDLE;
	VkDescriptorSetLayout sampledLayout = VK_NULL_HANDLE, storageLayout = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	bool fastSupported = false;
	uint32_t subgroupSize = 0;
	std::string deviceName, driverName, driverInfo, apiVersion;
	uint32_t vendor = 0, deviceId = 0, driverVersion = 0;
	std::vector<VkImageMemoryBarrier> pending;
	VkPipelineStageFlags pendingSrc = 0, pendingDst = 0;

	uint32_t memoryType( uint32_t bits, VkMemoryPropertyFlags flags ) {

		for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; i ++ ) if ( ( bits & ( 1u << i ) ) && ( memoryProperties.memoryTypes[ i ].propertyFlags & flags ) == flags ) return i;
		fprintf( stderr, "no memory type\n" );
		exit( 1 );

	}

	void begin() {

		VK_CHECK( vkResetCommandBuffer( cmd, 0 ) );
		VkCommandBufferBeginInfo info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VK_CHECK( vkBeginCommandBuffer( cmd, &info ) );

	}

	void submitAndWait() {

		VK_CHECK( vkEndCommandBuffer( cmd ) );
		VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &cmd;
		VK_CHECK( vkResetFences( device, 1, &fence ) );
		VK_CHECK( vkQueueSubmit( queue, 1, &submit, fence ) );
		VK_CHECK( vkWaitForFences( device, 1, &fence, VK_TRUE, UINT64_MAX ) );

	}

	/** Queue a layout change of one level (recorded on flush()); also orders memory. */
	void require( Image & image, uint32_t level, VkImageLayout layout, VkPipelineStageFlags src, VkAccessFlags srcAccess, VkPipelineStageFlags dst, VkAccessFlags dstAccess ) {

		if ( image.layout[ level ] == layout ) return;
		VkImageMemoryBarrier barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		barrier.srcAccessMask = srcAccess;
		barrier.dstAccessMask = dstAccess;
		barrier.oldLayout = image.layout[ level ];
		barrier.newLayout = layout;
		barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image.image;
		barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1 };
		pending.push_back( barrier );
		pendingSrc |= src; pendingDst |= dst;
		image.layout[ level ] = layout;

	}

	void flush() {

		if ( pending.empty() ) return;
		vkCmdPipelineBarrier( cmd, pendingSrc, pendingDst, 0, 0, nullptr, 0, nullptr, uint32_t( pending.size() ), pending.data() );
		pending.clear();
		pendingSrc = pendingDst = 0;

	}

	void memoryBarrier( VkPipelineStageFlags src, VkAccessFlags srcAccess, VkPipelineStageFlags dst, VkAccessFlags dstAccess ) {

		VkMemoryBarrier barrier { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		barrier.srcAccessMask = srcAccess;
		barrier.dstAccessMask = dstAccess;
		vkCmdPipelineBarrier( cmd, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr );

	}

	struct Buffer { VkBuffer buffer; VkDeviceMemory memory; void * mapped; };

	Buffer hostBuffer( VkDeviceSize size, VkBufferUsageFlags usage ) {

		Buffer buffer {};
		VkBufferCreateInfo info { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		info.size = size;
		info.usage = usage;
		VK_CHECK( vkCreateBuffer( device, &info, nullptr, &buffer.buffer ) );
		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements( device, buffer.buffer, &requirements );
		VkMemoryAllocateInfo allocate { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
		allocate.allocationSize = requirements.size;
		allocate.memoryTypeIndex = memoryType( requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );
		VK_CHECK( vkAllocateMemory( device, &allocate, nullptr, &buffer.memory ) );
		VK_CHECK( vkBindBufferMemory( device, buffer.buffer, buffer.memory, 0 ) );
		VK_CHECK( vkMapMemory( device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped ) );
		return buffer;

	}

	void destroy( Buffer & buffer ) { vkDestroyBuffer( device, buffer.buffer, nullptr ); vkFreeMemory( device, buffer.memory, nullptr ); }

	Image createImage( uint32_t width, uint32_t height, Format format, VkImageUsageFlags usage ) {

		Image image;
		image.width = width; image.height = height; image.levels = mipCount( width, height ); image.format = format;
		VkImageCreateInfo info { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
		info.imageType = VK_IMAGE_TYPE_2D;
		info.format = vkFormat( format );
		info.extent = { width, height, 1 };
		info.mipLevels = image.levels;
		info.arrayLayers = 1;
		info.samples = VK_SAMPLE_COUNT_1_BIT;
		info.tiling = VK_IMAGE_TILING_OPTIMAL;
		info.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		VK_CHECK( vkCreateImage( device, &info, nullptr, &image.image ) );
		VkMemoryRequirements requirements;
		vkGetImageMemoryRequirements( device, image.image, &requirements );
		VkMemoryAllocateInfo allocate { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
		allocate.allocationSize = requirements.size;
		allocate.memoryTypeIndex = memoryType( requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
		VK_CHECK( vkAllocateMemory( device, &allocate, nullptr, &image.memory ) );
		VK_CHECK( vkBindImageMemory( device, image.image, image.memory, 0 ) );
		image.layout.assign( image.levels, VK_IMAGE_LAYOUT_UNDEFINED );

		if ( usage & ( VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT ) ) {

			VkImageViewCreateInfo view { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
			view.image = image.image;
			view.viewType = VK_IMAGE_VIEW_TYPE_2D;
			view.format = info.format;
			view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, image.levels, 0, 1 };
			VK_CHECK( vkCreateImageView( device, &view, nullptr, &image.sampledView ) );
			for ( uint32_t level = 0; level < image.levels; level ++ ) {

				view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1 };
				VkImageView levelView;
				VK_CHECK( vkCreateImageView( device, &view, nullptr, &levelView ) );
				image.levelViews.push_back( levelView );

			}

		}
		return image;

	}

	void destroy( Image & image ) {

		for ( auto view : image.levelViews ) vkDestroyImageView( device, view, nullptr );
		if ( image.sampledView ) vkDestroyImageView( device, image.sampledView, nullptr );
		vkDestroyImage( device, image.image, nullptr );
		vkFreeMemory( device, image.memory, nullptr );

	}

	void upload( Image & image, const std::vector<uint8_t> & data ) {

		Buffer staging = hostBuffer( data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT );
		memcpy( staging.mapped, data.data(), data.size() );
		begin();
		require( image, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT );
		flush();
		VkBufferImageCopy copy {};
		copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		copy.imageExtent = { image.width, image.height, 1 };
		vkCmdCopyBufferToImage( cmd, staging.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy );
		submitAndWait();
		destroy( staging );

	}

	/** Every level as RGBA float32, level 0 first. */
	std::vector<std::vector<float>> readLevels( Image & image ) {

		const uint32_t bytes = formatBytes( image.format );
		std::vector<VkDeviceSize> offsets;
		VkDeviceSize total = 0;
		for ( uint32_t level = 0; level < image.levels; level ++ ) { offsets.push_back( total ); total += VkDeviceSize( mipSize( image.width, level ) ) * mipSize( image.height, level ) * bytes; }
		Buffer readback = hostBuffer( total, VK_BUFFER_USAGE_TRANSFER_DST_BIT );
		begin();
		std::vector<VkImageLayout> before = image.layout;
		for ( uint32_t level = 0; level < image.levels; level ++ ) require( image, level, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT );
		flush();
		for ( uint32_t level = 0; level < image.levels; level ++ ) {

			VkBufferImageCopy copy {};
			copy.bufferOffset = offsets[ level ];
			copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 };
			copy.imageExtent = { mipSize( image.width, level ), mipSize( image.height, level ), 1 };
			vkCmdCopyImageToBuffer( cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &copy );

		}
		for ( uint32_t level = 0; level < image.levels; level ++ ) require( image, level, before[ level ], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT );
		flush();
		submitAndWait();
		std::vector<std::vector<float>> levels( image.levels );
		for ( uint32_t level = 0; level < image.levels; level ++ ) {

			const uint32_t w = mipSize( image.width, level ), h = mipSize( image.height, level );
			levels[ level ].resize( size_t( w ) * h * 4 );
			for ( uint32_t y = 0; y < h; y ++ ) decodeRow( ( const uint8_t * ) readback.mapped + offsets[ level ] + size_t( y ) * w * bytes, w, image.format, levels[ level ].data() + size_t( y ) * w * 4 );

		}
		destroy( readback );
		return levels;

	}

	/** Descriptor sets for nvpro_pyramid: the sampled image (all levels) and 16 level views. */
	void bindPyramid( Image & image ) {

		VkDescriptorSetLayout layouts[] = { sampledLayout, storageLayout };
		VkDescriptorSetAllocateInfo allocate { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		allocate.descriptorPool = descriptorPool;
		allocate.descriptorSetCount = 2;
		allocate.pSetLayouts = layouts;
		VkDescriptorSet sets[ 2 ];
		VK_CHECK( vkAllocateDescriptorSets( device, &allocate, sets ) );
		image.sampledSet = sets[ 0 ]; image.storageSet = sets[ 1 ];

		VkDescriptorImageInfo sampled { sampler, image.sampledView, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo storage[ 16 ];
		// Unused entries repeat the last level; the shader never touches them.
		for ( uint32_t i = 0; i < 16; i ++ ) storage[ i ] = { VK_NULL_HANDLE, image.levelViews[ std::min( i, image.levels - 1 ) ], VK_IMAGE_LAYOUT_GENERAL };
		VkWriteDescriptorSet writes[ 2 ] { { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }, { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET } };
		writes[ 0 ].dstSet = image.sampledSet; writes[ 0 ].descriptorCount = 1; writes[ 0 ].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[ 0 ].pImageInfo = &sampled;
		writes[ 1 ].dstSet = image.storageSet; writes[ 1 ].descriptorCount = 16; writes[ 1 ].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[ 1 ].pImageInfo = storage;
		vkUpdateDescriptorSets( device, 2, writes, 0, nullptr );

	}

};

static Vk createVk( const Options & options ) {

	Vk vk;
	VkApplicationInfo app { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	app.pApplicationName = "vk-mips";
	app.apiVersion = VK_API_VERSION_1_2;
	VkInstanceCreateInfo instanceInfo { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	instanceInfo.pApplicationInfo = &app;
	const char * validationLayer = "VK_LAYER_KHRONOS_validation";
	if ( options.validation ) { instanceInfo.enabledLayerCount = 1; instanceInfo.ppEnabledLayerNames = &validationLayer; }
	VK_CHECK( vkCreateInstance( &instanceInfo, nullptr, &vk.instance ) );

	uint32_t count = 0;
	vkEnumeratePhysicalDevices( vk.instance, &count, nullptr );
	std::vector<VkPhysicalDevice> devices( count );
	vkEnumeratePhysicalDevices( vk.instance, &count, devices.data() );
	const VkPhysicalDeviceType wanted = options.power == "low" ? VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU : VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
	for ( auto device : devices ) {

		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties( device, &properties );
		std::string name = properties.deviceName, lower = name, part = options.gpu;
		std::transform( lower.begin(), lower.end(), lower.begin(), ::tolower );
		std::transform( part.begin(), part.end(), part.begin(), ::tolower );
		if ( ! options.gpu.empty() ? lower.find( part ) != std::string::npos : properties.deviceType == wanted ) { vk.physical = device; break; }

	}
	if ( ! vk.physical ) { fprintf( stderr, "no matching Vulkan device\n" ); exit( 1 ); }

	VkPhysicalDeviceDriverProperties driver { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
	VkPhysicalDeviceSubgroupProperties subgroup { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
	subgroup.pNext = &driver;
	VkPhysicalDeviceProperties2 properties { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
	properties.pNext = &subgroup;
	vkGetPhysicalDeviceProperties2( vk.physical, &properties );
	vk.deviceName = properties.properties.deviceName;
	vk.vendor = properties.properties.vendorID;
	vk.deviceId = properties.properties.deviceID;
	vk.driverVersion = properties.properties.driverVersion;
	vk.driverName = driver.driverName;
	vk.driverInfo = driver.driverInfo;
	{

		const uint32_t v = properties.properties.apiVersion;
		vk.apiVersion = std::to_string( VK_API_VERSION_MAJOR( v ) ) + "." + std::to_string( VK_API_VERSION_MINOR( v ) ) + "." + std::to_string( VK_API_VERSION_PATCH( v ) );

	}
	vk.timestampPeriod = properties.properties.limits.timestampPeriod;
	vk.subgroupSize = subgroup.subgroupSize;
	vk.fastSupported = subgroup.subgroupSize >= 16 && ( subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT ) && ( subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT );
	vkGetPhysicalDeviceMemoryProperties( vk.physical, &vk.memoryProperties );

	vkGetPhysicalDeviceQueueFamilyProperties( vk.physical, &count, nullptr );
	std::vector<VkQueueFamilyProperties> families( count );
	vkGetPhysicalDeviceQueueFamilyProperties( vk.physical, &count, families.data() );
	for ( uint32_t i = 0; i < count; i ++ ) if ( ( families[ i ].queueFlags & ( VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT ) ) == ( VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT ) && families[ i ].timestampValidBits ) { vk.family = i; break; }

	const float priority = 1;
	VkDeviceQueueCreateInfo queueInfo { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	queueInfo.queueFamilyIndex = vk.family;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = &priority;
	VkDeviceCreateInfo deviceInfo { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	deviceInfo.queueCreateInfoCount = 1;
	deviceInfo.pQueueCreateInfos = &queueInfo;
	VK_CHECK( vkCreateDevice( vk.physical, &deviceInfo, nullptr, &vk.device ) );
	vkGetDeviceQueue( vk.device, vk.family, 0, &vk.queue );

	VkCommandPoolCreateInfo poolInfo { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = vk.family;
	VK_CHECK( vkCreateCommandPool( vk.device, &poolInfo, nullptr, &vk.pool ) );
	VkCommandBufferAllocateInfo cmdInfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cmdInfo.commandPool = vk.pool;
	cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdInfo.commandBufferCount = 1;
	VK_CHECK( vkAllocateCommandBuffers( vk.device, &cmdInfo, &vk.cmd ) );
	VkFenceCreateInfo fenceInfo { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VK_CHECK( vkCreateFence( vk.device, &fenceInfo, nullptr, &vk.fence ) );
	VkQueryPoolCreateInfo queryInfo { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
	queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
	queryInfo.queryCount = 2;
	VK_CHECK( vkCreateQueryPool( vk.device, &queryInfo, nullptr, &vk.queries ) );

	// nvpro's sampler: linear, clamp to edge, levels picked explicitly by textureLod.
	VkSamplerCreateInfo samplerInfo { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
	VK_CHECK( vkCreateSampler( vk.device, &samplerInfo, nullptr, &vk.sampler ) );

	VkDescriptorSetLayoutBinding sampledBinding { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorSetLayoutBinding storageBinding { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16, VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorSetLayoutCreateInfo layoutInfo { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &sampledBinding;
	VK_CHECK( vkCreateDescriptorSetLayout( vk.device, &layoutInfo, nullptr, &vk.sampledLayout ) );
	layoutInfo.pBindings = &storageBinding;
	VK_CHECK( vkCreateDescriptorSetLayout( vk.device, &layoutInfo, nullptr, &vk.storageLayout ) );
	VkDescriptorSetLayout setLayouts[] = { vk.sampledLayout, vk.storageLayout };
	VkPushConstantRange push { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 };
	VkPipelineLayoutCreateInfo pipelineLayoutInfo { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipelineLayoutInfo.setLayoutCount = 2;
	pipelineLayoutInfo.pSetLayouts = setLayouts;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &push;
	VK_CHECK( vkCreatePipelineLayout( vk.device, &pipelineLayoutInfo, nullptr, &vk.pipelineLayout ) );

	VkDescriptorPoolSize sizes[] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64 * 16 } };
	VkDescriptorPoolCreateInfo descriptorPoolInfo { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	descriptorPoolInfo.maxSets = 128;
	descriptorPoolInfo.poolSizeCount = 2;
	descriptorPoolInfo.pPoolSizes = sizes;
	VK_CHECK( vkCreateDescriptorPool( vk.device, &descriptorPoolInfo, nullptr, &vk.descriptorPool ) );
	return vk;

}

static VkPipeline computePipeline( Vk & vk, const std::filesystem::path & file ) {

	const auto code = readFile( file );
	VkShaderModuleCreateInfo moduleInfo { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	moduleInfo.codeSize = code.size();
	moduleInfo.pCode = reinterpret_cast<const uint32_t *>( code.data() );
	VkShaderModule module;
	VK_CHECK( vkCreateShaderModule( vk.device, &moduleInfo, nullptr, &module ) );
	VkComputePipelineCreateInfo info { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	info.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	info.stage.module = module;
	info.stage.pName = "main";
	info.layout = vk.pipelineLayout;
	VkPipeline pipeline;
	VK_CHECK( vkCreateComputePipelines( vk.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline ) );
	vkDestroyShaderModule( vk.device, module, nullptr );
	return pipeline;

}

// ---------------------------------------------------------------------------------------------

struct Variant {
	std::string name, method;
	uint32_t dispatches = 0;
	Image image;
	bool hasImage = false;
	std::function<void()> encode;
	std::vector<double> samples;
};

static double timeVk( Vk & vk, Variant & variant, int repeats ) {

	vk.begin();
	vkCmdResetQueryPool( vk.cmd, vk.queries, 0, 2 );
	vkCmdWriteTimestamp( vk.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk.queries, 0 );
	for ( int i = 0; i < repeats; i ++ ) if ( variant.encode ) variant.encode();
	vk.flush();
	vkCmdWriteTimestamp( vk.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk.queries, 1 );
	vk.submitAndWait();
	uint64_t stamps[ 2 ];
	VK_CHECK( vkGetQueryPoolResults( vk.device, vk.queries, 0, 2, sizeof( stamps ), stamps, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT ) );
	if ( stamps[ 1 ] < stamps[ 0 ] ) { fprintf( stderr, "invalid timestamps\n" ); exit( 1 ); }
	return double( stamps[ 1 ] - stamps[ 0 ] ) * vk.timestampPeriod / 1e6 / repeats;

}

static bool selected( const Options & options, const std::string & name ) {

	if ( options.only.empty() || name == "empty" ) return true;
	for ( const auto & part : split( options.only ) ) if ( name.find( part ) != std::string::npos ) return true;
	return false;

}

/** Count the dispatches nvproCmdPyramidDispatch records, by replaying its schedule. */
static uint32_t countNvproDispatches( uint32_t width, uint32_t height, bool fast ) {

	uint32_t x = width, y = height, remaining = mipCount( width, height ) - 1, count = 0;
	while ( remaining ) {

		uint32_t levels = 0;
		if ( fast && x % 4 == 0 && y % 4 == 0 ) {

			uint32_t a = x, b = y;
			while ( a % 2 == 0 && b % 2 == 0 && levels < remaining && levels < 6 ) { a /= 2; b /= 2; levels ++; }

		}
		if ( ! levels ) levels = std::min( 2u, remaining );
		remaining -= levels;
		x = std::max( 1u, x >> levels ); y = std::max( 1u, y >> levels );
		count ++;

	}
	return count;

}

int main( int argc, char ** argv ) {

	Options options;
	options.shaders = ( std::filesystem::path( argv[ 0 ] ).parent_path() / "shaders" ).string();
	for ( int i = 1; i < argc; i ++ ) {

		const std::string arg = argv[ i ];
		const auto value = [ & ]( const char * key ) -> const char * { const size_t n = strlen( key ); return arg.compare( 0, n, key ) == 0 ? arg.c_str() + n : nullptr; };
		if ( auto v = value( "--power=" ) ) options.power = v;
		else if ( auto v = value( "--gpu=" ) ) options.gpu = v;
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
		else if ( arg == "--validation" ) options.validation = true;
		else if ( arg == "--no-timing" ) options.noTiming = true;
		else { fprintf( stderr, "unknown option %s\n", arg.c_str() ); return 1; }

	}

	Vk vk = createVk( options );
	printf( "device: %s (vendor 0x%04x device 0x%04x), Vulkan %s, driver %s %s, subgroup %u, nvpro fast pipeline %s, timestamp period %.3f ns\n", vk.deviceName.c_str(), vk.vendor, vk.deviceId, vk.apiVersion.c_str(), vk.driverName.c_str(), vk.driverInfo.c_str(), vk.subgroupSize, vk.fastSupported ? "on" : "off", vk.timestampPeriod );
	fflush( stdout );

	const std::filesystem::path shaderFolder = options.shaders;
	std::vector<Result> results;

	for ( const auto & formatText : options.formats ) {

		Format format;
		if ( ! parseFormat( formatText, format ) ) { fprintf( stderr, "unknown format %s\n", formatText.c_str() ); return 1; }
		const std::string tag = shaderFormat( format );
		NvproPyramidPipelines pipelines;
		pipelines.layout = vk.pipelineLayout;
		pipelines.generalPipeline = computePipeline( vk, shaderFolder / ( "nvpro-general-" + tag + ".spv" ) );
		pipelines.fastPipeline = vk.fastSupported ? computePipeline( vk, shaderFolder / ( "nvpro-fast-" + tag + ".spv" ) ) : VK_NULL_HANDLE;

		for ( const Size size : options.sizes ) {

			VK_CHECK( vkResetDescriptorPool( vk.device, vk.descriptorPool, 0 ) );
			const auto data = makeData( size.width, size.height, format );
			const uint32_t levels = mipCount( size.width, size.height ), outputs = levels - 1;
			std::vector<Variant> variants;
			variants.reserve( 8 );

			for ( const bool generalOnly : { false, true } ) {

				const std::string name = generalOnly ? "nvpro-general" : "nvpro-pyramid";
				if ( ! selected( options, name ) || ( generalOnly && ! pipelines.fastPipeline ) ) continue;
				Variant variant;
				variant.name = name;
				variant.method = "nvpro";
				variant.dispatches = countNvproDispatches( size.width, size.height, ! generalOnly && pipelines.fastPipeline );
				variant.image = vk.createImage( size.width, size.height, format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT );
				variant.hasImage = true;
				vk.upload( variant.image, data );
				vk.bindPyramid( variant.image );
				variants.push_back( std::move( variant ) );
				NvproPyramidPipelines used = pipelines;
				if ( generalOnly ) used.fastPipeline = VK_NULL_HANDLE;
				variants.back().encode = [ &vk, used, index = variants.size() - 1, &variants ]() {

					Image & image = variants[ index ].image;
					for ( uint32_t level = 0; level < image.levels; level ++ ) vk.require( image, level, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT );
					vk.flush();
					VkDescriptorSet sets[] = { image.sampledSet, image.storageSet };
					vkCmdBindDescriptorSets( vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipelineLayout, 0, 2, sets, 0, nullptr );
					nvproCmdPyramidDispatch( vk.cmd, used, image.width, image.height, image.levels );
					// nvpro's benchmark: a write-after-write barrier between repeats.
					vk.memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT );

				};

			}

			if ( ! formatIsMax( format ) && selected( options, "blit" ) ) {

				// The common Vulkan pattern: per level, previous level to TRANSFER_SRC, blit with a linear filter.
				Variant variant;
				variant.name = "blit";
				variant.method = "blit";
				variant.dispatches = outputs;
				variant.image = vk.createImage( size.width, size.height, format, VK_IMAGE_USAGE_SAMPLED_BIT );
				variant.hasImage = true;
				vk.upload( variant.image, data );
				variants.push_back( std::move( variant ) );
				variants.back().encode = [ &vk, index = variants.size() - 1, &variants ]() {

					Image & image = variants[ index ].image;
					vk.require( image, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT );
					for ( uint32_t level = 1; level < image.levels; level ++ ) vk.require( image, level, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT );
					vk.flush();
					for ( uint32_t level = 1; level < image.levels; level ++ ) {

						if ( level > 1 ) { vk.require( image, level - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT ); vk.flush(); }
						VkImageBlit blit {};
						blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 };
						blit.srcOffsets[ 1 ] = { int32_t( mipSize( image.width, level - 1 ) ), int32_t( mipSize( image.height, level - 1 ) ), 1 };
						blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 };
						blit.dstOffsets[ 1 ] = { int32_t( mipSize( image.width, level ) ), int32_t( mipSize( image.height, level ) ), 1 };
						vkCmdBlitImage( vk.cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR );

					}

				};

			}

			{

				Variant variant;
				variant.name = "empty";
				variant.method = "empty";
				variants.push_back( std::move( variant ) );

			}

			for ( int i = 0; i < options.warmup; i ++ ) for ( auto & variant : variants ) timeVk( vk, variant, 1 );

			if ( ! options.dump.empty() ) for ( const Size dumpSize : options.dumpSizes ) if ( dumpSize.width == size.width && dumpSize.height == size.height ) {

				for ( auto & variant : variants ) if ( variant.hasImage ) writeDump( options.dump, variant.name, format, size.width, size.height, vk.readLevels( variant.image ) );
				printf( "dumped %s %ux%u\n", formatName( format ), size.width, size.height );

			}

			if ( ! options.noTiming ) {

				for ( int round = 0; round < options.rounds; round ++ ) for ( size_t k = 0; k < variants.size(); k ++ ) {

					Variant & variant = variants[ ( k + round ) % variants.size() ];
					variant.samples.push_back( timeVk( vk, variant, options.repeats ) );

				}
				printf( "%s %ux%u (%u levels)\n", formatName( format ), size.width, size.height, outputs );
				for ( auto & variant : variants ) {

					results.push_back( { variant.name, variant.method, "vulkan", format, size.width, size.height, outputs, variant.dispatches, variant.samples } );
					printResult( results.back() );

				}
				fflush( stdout );

			}

			VK_CHECK( vkDeviceWaitIdle( vk.device ) );
			for ( auto & variant : variants ) if ( variant.hasImage ) vk.destroy( variant.image );

		}

		vkDestroyPipeline( vk.device, pipelines.generalPipeline, nullptr );
		if ( pipelines.fastPipeline ) vkDestroyPipeline( vk.device, pipelines.fastPipeline, nullptr );

	}

	if ( ! options.output.empty() ) {

		std::ostringstream adapterJson;
		adapterJson << "{ \"api\": \"Vulkan " << vk.apiVersion << "\", \"name\": \"" << vk.deviceName << "\", \"vendor\": " << vk.vendor << ", \"device\": " << vk.deviceId << ", \"driver\": \"" << vk.driverName << " " << vk.driverInfo
			<< "\", \"driverVersionRaw\": " << vk.driverVersion << ", \"subgroupSize\": " << vk.subgroupSize << ", \"nvproFastPipeline\": " << ( vk.fastSupported ? "true" : "false" ) << ", \"timestampPeriodNs\": " << vk.timestampPeriod << " }";
		const std::string method = "GPU timestamps (bottom of pipe) around " + std::to_string( options.repeats ) + " back-to-back chains, one submit per sample, variants rotated each round, median over " + std::to_string( options.rounds ) + " rounds; ms per chain";
		writeResults( options.output, "native/vk-mips", adapterJson.str(), method, options.rounds, options.repeats, options.warmup, results );

	}
	return 0;

}
