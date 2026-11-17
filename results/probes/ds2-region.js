const WIDTH = 1920;
const HEIGHT = 1080;
const RECT_SIZE = 32;
const UPDATES = 64;
const ROUNDS = 5;
const output = document.querySelector('#result');

function assert(condition, message) {
	if (!condition) throw new Error(message);
}

function percentile(values, p) {
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)];
}

function freshCounters() {
	return { createBuffer: 0, bufferBytes: 0, createBindGroup: 0, createView: 0, createComputePipeline: 0 };
}

function rectangles(texture) {
	const left = Math.floor(texture.width / 3);
	const top = Math.floor(texture.height / 3);
	return Array.from({ length: UPDATES }, (_, index) => ({ x: left + index, y: top + index, width: RECT_SIZE, height: RECT_SIZE }));
}

function instrumented(device, counters) {
	const counted = new Set(['createBuffer', 'createBindGroup', 'createComputePipeline', 'createComputePipelineAsync']);
	const measuredDevice = new Proxy(device, {
		get(target, property) {
			const value = Reflect.get(target, property, target);
			if (counted.has(property)) return (...args) => {
				if (property === 'createBuffer') {
					counters.createBuffer++;
					counters.bufferBytes += args[0].size;
				} else if (property === 'createBindGroup') counters.createBindGroup++;
				else counters.createComputePipeline++;
				return target[property](...args);
			};
			return typeof value === 'function' ? value.bind(target) : value;
		},
	});
	return { device: measuredDevice };
}

function wrapTexture(texture, counters) {
	return new Proxy(texture, {
		get(target, property) {
			if (property === 'createView') return (...args) => {
				counters.createView++;
				return target.createView(...args);
			};
			const value = Reflect.get(target, property, target);
			return typeof value === 'function' ? value.bind(target) : value;
		},
	});
}

async function settle(device) {
	await device.queue.onSubmittedWorkDone();
	await new Promise(resolve => setTimeout(resolve, 0));
}

async function warm(ds, device, texture, fixedRect, movingRects) {
	record(ds, device, texture, fixedRect);
	for (const rect of movingRects) record(ds, device, texture, rect);
	// Moving regions can evict the fixed plan from the 16-entry cache. Touch fixed last.
	record(ds, device, texture, fixedRect);
	await settle(device);
}

function record(ds, device, texture, rect) {
	const encoder = device.createCommandEncoder();
	ds.encode(encoder, texture, { rect });
	encoder.finish();
}

function encodeCpu(ds, device, texture, rect) {
	const start = performance.now();
	const encoder = device.createCommandEncoder();
	ds.encode(encoder, texture, { rect });
	encoder.finish();
	return performance.now() - start;
}

async function allocationRun(Downsampler, device, texture, fixedRect, movingRects, which) {
	const counters = freshCounters();
	const wrapped = instrumented(device, counters);
	const ds = new Downsampler(wrapped.device);
	const fixedTexture = wrapTexture(texture, counters);
	await warm(ds, device, fixedTexture, fixedRect, movingRects);
	Object.assign(counters, freshCounters());
	for (let round = 0; round < ROUNDS; round++) {
		for (let index = 0; index < UPDATES; index++) record(ds, wrapped.device, fixedTexture, fixedRect);
	}
	await settle(device);
	const fixed = { ...counters, updates: ROUNDS * UPDATES };
	Object.assign(counters, freshCounters());
	for (const rect of movingRects) record(ds, wrapped.device, fixedTexture, rect);
	await settle(device);
	const movingFirst64 = { ...counters, updates: UPDATES };
	const moving = { ...counters };
	Object.assign(counters, freshCounters());
	for (let round = 1; round < ROUNDS; round++) {
		for (const rect of movingRects) record(ds, wrapped.device, fixedTexture, rect);
		await settle(device);
		for (const key of Object.keys(counters)) moving[key] += counters[key];
		Object.assign(counters, freshCounters());
	}
	moving.updates = ROUNDS * UPDATES;
	ds.release(fixedTexture);
	return { build: which, fixed, movingFirst64, moving };
}

async function cpuRound(ds, device, texture, rects) {
	const times = [];
	for (const rect of rects) times.push(encodeCpu(ds, device, texture, rect));
	return times;
}

async function timeGpuRound(device, ds, texture, rects) {
	const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
	const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
	const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
	for (const rect of rects) ds.encodePass(pass, texture, { rect });
	pass.end();
	encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
	encoder.copyBufferToBuffer(resolve, 0, readback, 0, 16);
	device.queue.submit([encoder.finish()]);
	await readback.mapAsync(GPUMapMode.READ);
	const [start, end] = new BigUint64Array(readback.getMappedRange());
	readback.unmap();
	querySet.destroy(); resolve.destroy(); readback.destroy();
	assert(end > start, `invalid timestamp range ${start}..${end}`);
	return Number(end - start) / 1e6 / UPDATES;
}

async function main() {
	const parameters = new URLSearchParams(location.search);
	const baselineUrl = parameters.get('baseline');
	const candidateUrl = parameters.get('candidate');
	const builds = baselineUrl && candidateUrl
		? [
			{ name: 'baseline', url: baselineUrl },
			{ name: 'candidate', url: candidateUrl },
		]
		: [{ name: 'target', url: parameters.get('target') ?? 'target' }];
	for (const build of builds) {
		const module = await import(`/dist/${build.url}/webgpu/index.js`);
		build.Downsampler = module.Downsampler;
	}
	const adapter = await navigator.gpu?.requestAdapter();
	assert(adapter, 'WebGPU adapter unavailable');
	const timestampQuery = adapter.features.has('timestamp-query');
	const device = await adapter.requestDevice({ requiredFeatures: timestampQuery ? ['timestamp-query'] : [] });
	device.pushErrorScope('validation');
	assert(crossOriginIsolated, 'COOP/COEP headers are required for the high-resolution timer');
	const texture = device.createTexture({
		size: [WIDTH, HEIGHT, 1],
		mipLevelCount: Math.floor(Math.log2(Math.max(WIDTH, HEIGHT))) + 1,
		format: 'rgba8unorm',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
	});
	const fixedRect = { x: Math.floor(WIDTH / 3), y: Math.floor(HEIGHT / 3), width: RECT_SIZE, height: RECT_SIZE };
	const movingRects = rectangles(texture);
	const fixedRects = Array.from({ length: UPDATES }, () => fixedRect);
	const raw = new Map();
	const allocations = {};
	for (const build of builds) {
		const ds = new build.Downsampler(device);
		raw.set(build.name, ds);
		await warm(ds, device, texture, fixedRect, movingRects);
		allocations[build.name] = await allocationRun(build.Downsampler, device, texture, fixedRect, movingRects, build.name);
	}

	const cpuTimes = Object.fromEntries(builds.map(build => [build.name, { fixed: [], moving: [] }]));
	for (let round = 0; round < ROUNDS; round++) {
		const order = round % 2 === 0 ? builds : [...builds].reverse();
		for (const build of order) {
			const ds = raw.get(build.name);
			const cases = round % 2 === 0 ? [['fixed', fixedRects], ['moving', movingRects]] : [['moving', movingRects], ['fixed', fixedRects]];
			for (const [label, rects] of cases) {
				cpuTimes[build.name][label].push(...await cpuRound(ds, device, texture, rects));
				await settle(device);
			}
		}
	}
	const cpu = Object.fromEntries(builds.map(build => [build.name, Object.fromEntries(['fixed', 'moving'].map(label => {
		const values = cpuTimes[build.name][label];
		return [label, { updates: values.length, encodeMs: { p50: percentile(values, 0.5), p95: percentile(values, 0.95) } }];
	}))]));

	const gpuSamples = Object.fromEntries(builds.map(build => [build.name, { fixed: [], moving: [] }]));
	if (timestampQuery) {
		// Warm pipelines with submitted work; readback and queue waits are outside timestamps.
		for (const build of builds) {
			await timeGpuRound(device, raw.get(build.name), texture, fixedRects);
			await timeGpuRound(device, raw.get(build.name), texture, movingRects);
		}
		for (let round = 0; round < ROUNDS; round++) {
			const order = round % 2 === 0 ? builds : [...builds].reverse();
			for (const build of order) {
				const cases = round % 2 === 0 ? [['fixed', fixedRects], ['moving', movingRects]] : [['moving', movingRects], ['fixed', fixedRects]];
				for (const [label, rects] of cases) gpuSamples[build.name][label].push(await timeGpuRound(device, raw.get(build.name), texture, rects));
			}
		}
	}
	const gpu = timestampQuery ? Object.fromEntries(builds.map(build => [build.name, Object.fromEntries(['fixed', 'moving'].map(label => {
		const samples = gpuSamples[build.name][label];
		return [label, { msPerUpdate: { p50: percentile(samples, 0.5), p95: percentile(samples, 0.95) }, samples }];
	}))])) : { available: false, reason: 'adapter lacks timestamp-query' };
	await device.queue.onSubmittedWorkDone();
	const validationError = await device.popErrorScope();
	assert(validationError === null, `WebGPU validation error: ${validationError?.message}`);
	let resolution = Infinity;
	let previous = performance.now();
	for (let index = 0; index < 10000; index++) {
		const now = performance.now();
		if (now > previous) { resolution = Math.min(resolution, now - previous); previous = now; }
	}
	const report = {
		adapter: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, device: adapter.info?.device, description: adapter.info?.description },
		timer: { crossOriginIsolated, observedResolutionMs: resolution },
		fixture: { width: WIDTH, height: HEIGHT, format: 'rgba8unorm', rect: `${RECT_SIZE}x${RECT_SIZE}`, updatesPerRound: UPDATES, rounds: ROUNDS },
		allocations,
		cpu,
		gpu,
		runSequence: Array.from({ length: ROUNDS }, (_, round) => ({
			round: round + 1,
			buildOrder: round % 2 === 0 ? builds.map(build => build.name) : [...builds].reverse().map(build => build.name),
			caseOrder: round % 2 === 0 ? ['fixed', 'moving'] : ['moving', 'fixed'],
		})),
	};
	output.textContent = JSON.stringify(report, null, 2);
	window.__probeResult = report;
	for (const ds of raw.values()) ds.release(texture);
	texture.destroy();
	device.destroy();
}

main().catch(error => {
	output.textContent = `ERROR: ${error.stack ?? error}`;
	window.__probeError = String(error.stack ?? error);
});
