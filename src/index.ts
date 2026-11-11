// Backend-neutral core. Backends live in `downsampler/webgpu` (and later others);
// the three.js adapter lives in `downsampler/three`.
export { planBatches, mipSize, mipLevelCountFor, tileSize, workgroupSide } from './core/plan.js';
export type { Batch, PlanOptions, OddSizePolicy } from './core/plan.js';
export { formats, integerDomains, isIntegerDomain, isSupportedFormat, quantize, roundHalfEven, sampledChannels, toHalf, fromHalf } from './core/formats.js';
export type { FormatInfo, IntegerQuantization, Quantization, SupportedFormat } from './core/formats.js';
export { copyReference, reduceReference, tolerance } from './core/reference.js';
export type { ReferenceLevel, ReferenceOptions } from './core/reference.js';
export { footprint, foldChannel, isBuiltinReduction, isChannelOps, needsTileEdges, reductionKey, resolveReduction } from './core/reduction.js';
export type { BuiltinReduction, ChannelOp, ChannelOps, CustomReduction, EdgePolicy, PyramidReferenceReduction, Reduction, ReferenceReduction, ResolvedReduction, TileEdgePolicy } from './core/reduction.js';
export { checkLevels, levelSize, maxTileLevels, pyramidTolerance, reducePyramid, tileDispatch } from './core/pyramid.js';
export type { PyramidLevel, PyramidReferenceOptions, TileDispatch } from './core/pyramid.js';
export { isSumOp, scalarBlock, scalarCombineDepth, scalarCombineReference, scalarDepth, scalarInvocations, scalarOps, scalarReference, scalarSide, scalarWeightChannel, scalarWorkgroups } from './core/scalar.js';
export type { ScalarCombineReferenceOptions, ScalarOp, ScalarReduce, ScalarReferenceOptions, ScalarReferenceResult } from './core/scalar.js';
