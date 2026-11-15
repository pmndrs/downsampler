// Chrome flags that pick the GPU for --power=high|low on machines with two.
//
// macOS / Linux: --use-webgpu-power-preference=force-high-performance | force-low-power.
// Windows: those flags make headless Chrome's requestAdapter() return null (RTX 3060 laptop +
// Intel UHD, 2026-11-12; webgpu-ddgi saw the same), so the GPU is pinned by adapter LUID
// instead: --use-adapter-luid=<high>,<low>, after which Dawn sees only that adapter. LUIDs change
// on every boot, so they are read from the Windows device properties (DEVPKEY_Gpu_Luid) each run.
// 'low' is the Intel GPU (PCI vendor 8086); 'high' is the first other display adapter.
// The PowerShell query takes seconds and timed out under load (six suite shards at once,
// 2026-11-14), so its output is kept in DS_GPU_ADAPTERS: the first call in a process sets it,
// and child processes (test/parallel.js's shards) inherit it instead of querying again.
import { execFileSync } from 'node:child_process';

const listLuids = `
Get-PnpDevice -Class Display -PresentOnly | ForEach-Object {
	$luid = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName '{60B193CB-5276-4D0F-96FC-F173ABAD3EC6} 2' -ErrorAction SilentlyContinue
	if ($luid) { '{0}|{1}' -f $_.InstanceId, $luid.Data }
}`;

/** Browser arguments for `power` ('high' | 'low' | undefined). */
export function gpuChoiceArgs( power ) {

	if ( ! power ) return [];
	if ( process.platform !== 'win32' ) return [ `--use-webgpu-power-preference=${ power === 'low' ? 'force-low-power' : 'force-high-performance' }` ];

	const output = process.env.DS_GPU_ADAPTERS ?? execFileSync( 'powershell.exe', [ '-NoProfile', '-NonInteractive', '-Command', listLuids ], { encoding: 'utf8', timeout: 120000, windowsHide: true } );
	process.env.DS_GPU_ADAPTERS = output;
	const adapters = output.split( /\r?\n/ ).map( line => line.trim().split( '|' ) ).filter( ( [ id, luid ] ) => id && /^\d+$/.test( luid ?? '' ) )
		.map( ( [ id, luid ] ) => ( { id, intel: id.toUpperCase().includes( 'VEN_8086' ), luid: BigInt( luid ) } ) );
	const adapter = adapters.find( candidate => candidate.intel === ( power === 'low' ) );
	if ( ! adapter ) throw new Error( `--power=${ power }: no matching display adapter among ${ adapters.map( candidate => candidate.id ).join( ', ' ) || 'none' }` );
	return [ `--use-adapter-luid=${ Number( adapter.luid >> 32n ) },${ Number( adapter.luid & 0xffffffffn ) }` ];

}
