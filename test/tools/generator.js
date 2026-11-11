// Plug-in points for downsampler-generator versions of the upscaler's pyramid
// passes: the third column of the comparison (upscaler-current vs
// FSR-reference vs downsampler-generator).
//
// Each entry's `create( device, context )` returns a Fixture (see harness.js)
// or null while the generator API it needs does not exist. `context` holds:
//   scene     the synthetic frame (scene.js)
//   formats   storage formats for the FSR outputs (gpu.js outputFormats)
//   tools     factories for the FSR tool halves, already in SPEC §14.1 shape:
//               tools.luma( { variant } )      → { wgsl, layout, group, reset, check, destroy }
//               tools.shading( { levels } )    → { wgsl, layout, group, texture, check, destroy }
//             `wgsl` defines load / store (/ finish, spdLoad) for @group(1);
//             bind `group` at index 1 and the fixture's check() compares
//             against the same oracle as the FSR fixture.
//
// A generator fixture builds its kernel from tool.wgsl (the generator owns
// @group(0)), records its dispatches in encodePass( pass ), and returns
// { name, pass, dispatches, encodePass, reset: tool.reset, check: () => tool.check(), destroy }.
// It appears in the report next to `compareWith`, and the ratio
// generator ÷ compareWith is printed.

export const generatorPasses = [
	{
		// M2T.2 reduceToScalar + M2T.1 hooks: every texel, 'zero-weight' edges,
		// level [1] stored by the hook, finish() for FrameInfo. 2 dispatches.
		name: 'generator/luma-pyramid',
		pass: 'exposure',
		compareWith: 'fsr/luma-pyramid-correct',
		milestone: 'M2T.2',
		create: null // async ( device, { scene, formats, tools } ) => { const tool = tools.luma( { variant: 'correct' } ); … }
	},
	{
		// M2T.1 with edges: 'clamp' and levels [1, 6, mips]: reproduces AMD's output (parity).
		name: 'generator/luma-pyramid-clamp',
		pass: 'exposure',
		compareWith: 'fsr/luma-pyramid',
		milestone: 'M2T.1',
		create: null // tools.luma( { variant: 'amd' } )
	},
	{
		// M2T.1: custom load, per-channel average, levels [1, 2, 3], edges: 'clamp'. 1 dispatch.
		name: 'generator/shading-change-pyramid',
		pass: 'shading',
		compareWith: 'fsr/shading-change-pyramid',
		milestone: 'M2T.1',
		create: null // tools.shading( { levels: 3 } )
	}
];
