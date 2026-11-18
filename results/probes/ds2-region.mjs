import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { gpuChoiceArgs } from '../../test/gpu-choice.js';

const probeDir = fileURLToPath(new URL('.', import.meta.url));
const arg = name => process.argv.find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3);
const baselinePath = arg('baseline');
const candidatePath = arg('candidate');
if (Boolean(baselinePath) !== Boolean(candidatePath)) throw new Error('paired mode requires both --baseline and --candidate');
const distPaths = baselinePath && candidatePath
	? { baseline: resolve(baselinePath), candidate: resolve(candidatePath) }
	: { target: resolve(arg('dist') ?? 'dist') };
const power = arg('power');
if (power && power !== 'high' && power !== 'low') throw new Error('--power must be high or low');
for (const path of Object.values(distPaths)) await stat(resolve(path, 'webgpu', 'index.js'));

const mime = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json' };
const server = createServer(async (request, response) => {
	const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
	let file;
	if (pathname === '/' || pathname === '/index.html') file = resolve(probeDir, 'ds2-region.html');
	else if (pathname === '/probe.js') file = resolve(probeDir, 'ds2-region.js');
	else if (pathname === '/favicon.ico') return response.writeHead(204).end();
	else {
		const match = /^\/dist\/(target|baseline|candidate)\/(.+)$/.exec(pathname);
		if (!match || !distPaths[match[1]]) return response.writeHead(404).end('not found');
		const root = resolve(distPaths[match[1]]);
		file = resolve(root, match[2]);
		if (!file.startsWith(`${root}${sep}`)) return response.writeHead(403).end('forbidden');
	}
	try {
		const body = await readFile(file);
		response.writeHead(200, {
			'Content-Type': mime[extname(file)] ?? 'application/octet-stream',
			'Cache-Control': 'no-store',
			'Cross-Origin-Opener-Policy': 'same-origin',
			'Cross-Origin-Embedder-Policy': 'require-corp',
			'Cross-Origin-Resource-Policy': 'same-origin',
		}).end(body);
	} catch { response.writeHead(404).end('not found'); }
});

await new Promise(resolveListen => server.listen(0, '127.0.0.1', resolveListen));
const args = ['--enable-unsafe-webgpu', ...gpuChoiceArgs(power)];
const browser = await puppeteer.launch({ headless: true, args, timeout: 120000, protocolTimeout: 120000 });
try {
	const page = await browser.newPage();
	page.on('console', message => { if (message.type() === 'error') console.error(message.text()); });
	page.on('pageerror', error => console.error(error.stack ?? error));
	const query = baselinePath
		? `?baseline=baseline&candidate=candidate`
		: '?target=target';
	await page.goto(`http://127.0.0.1:${server.address().port}/${query}`);
	await page.waitForFunction(() => window.__probeResult || window.__probeError, { timeout: 180000 });
	const error = await page.evaluate(() => window.__probeError);
	if (error) throw new Error(error);
	console.log(JSON.stringify({
		dist: distPaths,
		power: power ?? 'default',
		...await page.evaluate(() => window.__probeResult),
	}, null, 2));
} finally {
	await browser.close();
	server.close();
}
