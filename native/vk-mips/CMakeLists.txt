cmake_minimum_required( VERSION 3.20 )
project( vk_mips CXX )

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

find_package( Vulkan REQUIRED )

# nvpro_pyramid (vk_compute_mipmaps 6c248f5), fetched by ../fetch-deps.ps1.
set( NVPRO_PYRAMID "${CMAKE_CURRENT_SOURCE_DIR}/../.deps/vk_compute_mipmaps/nvpro_pyramid" CACHE PATH "nvpro_pyramid folder" )
if( NOT EXISTS "${NVPRO_PYRAMID}/nvpro_pyramid.glsl" )
	message( FATAL_ERROR "nvpro_pyramid not found under ${NVPRO_PYRAMID}: run native/fetch-deps.ps1 first" )
endif()

find_program( GLSLANG glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED )
message( STATUS "glslangValidator: ${GLSLANG}" )

add_executable( vk-mips src/main.cpp )
target_include_directories( vk-mips PRIVATE "${NVPRO_PYRAMID}" )
target_link_libraries( vk-mips PRIVATE Vulkan::Vulkan )
target_compile_definitions( vk-mips PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX )
if( MSVC )
	target_compile_options( vk-mips PRIVATE /W3 /utf-8 )
endif()

set( COMP "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pyramid.comp" )
set( PREAMBLE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/preamble.glsl" )
set( SHADER_OUTPUTS "" )
foreach( pipeline fast general )
	if( pipeline STREQUAL "fast" )
		set( fastValue 1 )
	else()
		set( fastValue 0 )
	endif()
	foreach( tag rgba8 rgba16f r32f-max )
		if( tag STREQUAL "r32f-max" )
			set( defines -DIMAGE_FORMAT=r32f -DMAX=1 )
		else()
			set( defines -DIMAGE_FORMAT=${tag} )
		endif()
		set( output "${CMAKE_CURRENT_BINARY_DIR}/shaders/nvpro-${pipeline}-${tag}.spv" )
		add_custom_command( OUTPUT "${output}"
			COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
			COMMAND "${GLSLANG}" -V --target-env vulkan1.1 -S comp -DFAST=${fastValue} ${defines} "-I${NVPRO_PYRAMID}" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders" -o "${output}" "${COMP}"
			DEPENDS "${COMP}" "${PREAMBLE}" "${NVPRO_PYRAMID}/nvpro_pyramid.glsl"
			COMMENT "glslang nvpro-${pipeline}-${tag}" VERBATIM )
		list( APPEND SHADER_OUTPUTS "${output}" )
	endforeach()
endforeach()

add_custom_target( shaders DEPENDS ${SHADER_OUTPUTS} )
add_dependencies( vk-mips shaders )
add_custom_command( TARGET vk-mips POST_BUILD
	COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders" "$<TARGET_FILE_DIR:vk-mips>/shaders" VERBATIM )
