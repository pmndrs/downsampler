cmake_minimum_required( VERSION 3.20 )
project( d3d12_mips CXX )

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

# AMD's SPD headers (FidelityFX SDK 7539ae8), fetched by ../fetch-deps.ps1.
set( FFX_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/../.deps/ffx-sdk/Kits/Cauldron2/framework/shaders/fidelityfx" CACHE PATH "FidelityFX shader headers (ffx_core.h, spd/ffx_spd.h)" )
if( NOT EXISTS "${FFX_SHADERS}/spd/ffx_spd.h" )
	message( FATAL_ERROR "ffx_spd.h not found under ${FFX_SHADERS}: run native/fetch-deps.ps1 first" )
endif()

# The Windows SDK's DXC ships dxil.dll, so its output is signed (the Vulkan SDK's is not).
file( GLOB SDK_DXC "C:/Program Files (x86)/Windows Kits/10/bin/10.*/x64/dxc.exe" )
list( SORT SDK_DXC ORDER DESCENDING )
if( NOT DXC )
	if( NOT SDK_DXC )
		message( FATAL_ERROR "dxc.exe not found in the Windows SDK: pass -DDXC=<path to a dxc.exe with dxil.dll beside it>" )
	endif()
	list( GET SDK_DXC 0 DXC )
endif()
message( STATUS "dxc: ${DXC}" )

add_executable( d3d12-mips src/main.cpp )
target_link_libraries( d3d12-mips PRIVATE d3d12 d3d11 dxgi )
target_compile_definitions( d3d12-mips PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS )
if( MSVC )
	target_compile_options( d3d12-mips PRIVATE /W3 /utf-8 )
endif()

set( SHADER_OUT "$<TARGET_FILE_DIR:d3d12-mips>/shaders" )
set( SPD_HLSL "${CMAKE_CURRENT_SOURCE_DIR}/shaders/spd.hlsl" )
set( BLIT_HLSL "${CMAKE_CURRENT_SOURCE_DIR}/shaders/blit.hlsl" )
set( SHADER_OUTPUTS "" )

function( spd_shader name )
	set( output "${CMAKE_CURRENT_BINARY_DIR}/shaders/${name}.cso" )
	add_custom_command( OUTPUT "${output}"
		COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
		COMMAND "${DXC}" -nologo -T cs_6_0 -E main -O3 -I "${FFX_SHADERS}" ${ARGN} -Fo "${output}" "${SPD_HLSL}"
		DEPENDS "${SPD_HLSL}" "${FFX_SHADERS}/spd/ffx_spd.h"
		COMMENT "dxc ${name}" VERBATIM )
	set( SHADER_OUTPUTS ${SHADER_OUTPUTS} "${output}" PARENT_SCOPE )
endfunction()

foreach( wave wave lds )
	if( wave STREQUAL "lds" )
		set( waveDefine -D FFX_SPD_NO_WAVE_OPERATIONS )
	else()
		set( waveDefine "" )
	endif()
	foreach( source load linear )
		if( source STREQUAL "linear" )
			set( sourceDefine -D SPD_LINEAR_SAMPLER )
		else()
			set( sourceDefine "" )
		endif()
		spd_shader( spd-${wave}-${source}-float ${waveDefine} ${sourceDefine} )
		spd_shader( spd-${wave}-${source}-unorm ${waveDefine} ${sourceDefine} -D SPD_UNORM )
	endforeach()
	spd_shader( spd-max-${wave} ${waveDefine} -D SPD_MAX -D SPD_CHANNELS=1 )
endforeach()

foreach( stage vs ps )
	set( output "${CMAKE_CURRENT_BINARY_DIR}/shaders/blit-${stage}.cso" )
	add_custom_command( OUTPUT "${output}"
		COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
		COMMAND "${DXC}" -nologo -T ${stage}_6_0 -E ${stage} -O3 -Fo "${output}" "${BLIT_HLSL}"
		DEPENDS "${BLIT_HLSL}" COMMENT "dxc blit-${stage}" VERBATIM )
	list( APPEND SHADER_OUTPUTS "${output}" )
endforeach()

add_custom_target( shaders DEPENDS ${SHADER_OUTPUTS} )
add_dependencies( d3d12-mips shaders )
add_custom_command( TARGET d3d12-mips POST_BUILD
	COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders" "${SHADER_OUT}" VERBATIM )
