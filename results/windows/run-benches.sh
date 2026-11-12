#!/bin/sh
# Every timed run of docs/WINDOWS-HANDOFF.md §2–3, in one session, so the WebGPU and native
# clocks are comparable. Each job runs under the shared GPU lock from webgpu-ddgi
# (tools/reference/bench-lock.ts): one GPU job at a time, a wait until the RTX 3060 is under
# 70 C, SM/memory clocks sampled every 200 ms and a VALID / INVALID throttling verdict, all in
# logs/<job>.log. Raw results go next to this file. Every job is killed after 25 min (owner: no run
# over 30 min).
#
# Usage (Git Bash, from anywhere): sh results/windows/run-benches.sh [job-name-filter]
# Build first: npm run build, and the native harnesses (native/README.md).
set -u
REPO=$( cd "$( dirname "$0" )/../.." && pwd )
LOCK=${BENCH_LOCK:-/c/Users/denni/Documents/dev/webgpu-ddgi/tools/reference/bench-lock.ts}
OUT=results/windows
LOGS=$REPO/$OUT/logs
FILTER=${1:-}
mkdir -p "$LOGS"

# The native sizes (handoff §3) plus 2048² and 256², for the webgpu-spd comparison.
SPD_SIZES=960x540,1129x635,1280x720,1707x960,1920x1080,2560x1440,1024x1024,2048x2048,4096x4096,256x256

job() {

	name=$1; shift
	case "$name" in *"$FILTER"*) ;; *) return ;; esac
	echo "=== $name start $( date -Iseconds )"
	timeout 1500 node "$LOCK" run --what "downsampler: $name" --clock-file "$LOGS/$name.clock.json" -- sh -c "cd '$REPO' && $*" > "$LOGS/$name.log" 2>&1
	echo "=== $name exit $? $( date -Iseconds )"
	grep -E "^(timing verdict|VALID|INVALID|NO VERDICT)|verdict" "$LOGS/$name.log" | tail -2

}

D3D12=native/d3d12-mips/build/Release/d3d12-mips.exe
VK=native/vk-mips/build/Release/vk-mips.exe

for gpu in rtx3060:high intel-uhd:low; do

	tag=${gpu%%:*}; power=${gpu#*:}
	job "$tag-spd" npm run bench -- --compare=spd --skip-validation --power=$power --output=$OUT/$tag-spd.json
	job "native-d3d12-$tag" $D3D12 --power=$power --output=$OUT/native-d3d12-$tag.json
	job "native-vk-$tag" $VK --power=$power --output=$OUT/native-vk-$tag.json
	job "$tag-webgpu-spd" npm run bench -- --compare=webgpu-spd --formats=rgba8unorm,rgba16float,r32float --sizes=$SPD_SIZES --skip-validation --power=$power --output=$OUT/$tag-webgpu-spd.json
	job "$tag-render-exact32" npm run bench -- --compare=render --skip-validation --power=$power --output=$OUT/$tag-render-exact32.json
	job "$tag-render-unsafe" npm run bench -- --compare=render --skip-validation --power=$power --dawn-features=disable_robustness,skip_validation --output=$OUT/$tag-render-unsafe.json
	job "$tag-bench" npm run bench -- --skip-validation --power=$power --output=$OUT/$tag-bench.json
	job "$tag-tools" npm run tools -- --power=$power --output=$OUT/$tag-tools.json
	job "$tag-webgpu-spd-check" node test/run.js --webgpu-spd-check --skip-validation --frames=256 --power=$power --output=$OUT/$tag-webgpu-spd-check.json

done

# Second pass on the RTX 3060: run-to-run spread for the ratios the owner decides on.
job "rtx3060-spd-2" npm run bench -- --compare=spd --skip-validation --power=high --output=$OUT/rtx3060-spd-2.json
job "native-d3d12-rtx3060-2" $D3D12 --power=high --output=$OUT/native-d3d12-rtx3060-2.json
job "native-vk-rtx3060-2" $VK --power=high --output=$OUT/native-vk-rtx3060-2.json
