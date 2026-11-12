# Fetch the upstream sources the native harnesses build against, at pinned commits, into
# native/.deps (gitignored). Nothing is vendored into this repository.
#
#   pwsh native/fetch-deps.ps1
#
#   FidelityFX SDK 7539ae8 (AMD FSR SDK 2.4.0; FSR 3.1.6): sparse checkout of the shader headers.
#     d3d12-mips compiles Kits/Cauldron2/framework/shaders/fidelityfx/spd/ffx_spd.h, which is
#     byte-identical to Kits/FidelityFX/upscalers/fsr3/include/gpu/spd/ffx_spd.h (the copy
#     test/tools/spd.js ports).
#   vk_compute_mipmaps 6c248f5: nvpro_pyramid (vk-mips uses only the nvpro_pyramid folder).

$ErrorActionPreference = 'Stop'
$deps = Join-Path $PSScriptRoot '.deps'
New-Item -ItemType Directory -Force $deps | Out-Null

function Fetch( $name, $url, $commit, $sparse ) {

	$dir = Join-Path $deps $name
	if ( -not ( Test-Path $dir ) ) {

		if ( $sparse ) {

			git clone --filter=blob:none --no-checkout $url $dir
			git -C $dir sparse-checkout init --cone
			git -C $dir sparse-checkout set @sparse

		} else {

			git clone --no-checkout $url $dir

		}

	}
	git -C $dir fetch --quiet origin $commit 2>$null
	git -C $dir checkout --quiet $commit
	Write-Host "$name at $( git -C $dir rev-parse --short HEAD )"

}

Fetch 'ffx-sdk' 'https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git' '7539ae8b6568ad29ed78a753404351daf3b5cca7' @( 'Kits/Cauldron2/framework/shaders', 'Kits/FidelityFX/upscalers/fsr3/include', 'Kits/FidelityFX/api/include', 'Kits/FidelityFX/api/internal/gpu' )
Fetch 'vk_compute_mipmaps' 'https://github.com/nvpro-samples/vk_compute_mipmaps.git' '6c248f554e91329875e9860f2258019a32b44ee9' $null
