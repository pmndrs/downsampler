// Backend-neutral core. Backends live in `downsampler/webgpu` (and later others);
// the three.js adapter lives in `downsampler/three`.
export { outputBufferLayout, planBatches, mipSize, mipLevelCountFor, tailTile, tailTileFor, tileSize, workgroupSide } from './core/plan.js';
export type { Batch, BatchKernel, BufferLevel, PlanOptions, OddSizePolicy } from './core/plan.js';
export { bufferWritable, formats, fromHalf, fromUfloat, integerDomains, isIntegerDomain, isPackedQuantization, isSupportedFormat, packedStep, packRg11b10, packRgb9e5, packTexel, quantize, quantizeTexel, rg11b10Max, rgb9e5Max, roundHalfEven, sampledChannels, storageWritable, toHalf, toUfloat, unpackRg11b10, unpackRgb9e5, unpackTexel } from './core/formats.js';
export type { FormatInfo, IntegerQuantization, PackedQuantization, Quantization, SupportedFormat } from './core/formats.js';
export { copyReference, reduceReference, tolerance } from './core/reference.js';
export type { ReferenceLevel, ReferenceOptions, ToleranceTexel } from './core/reference.js';
export { checkChannels, footprint, foldChannel, isBuiltinReduction, isChannelOps, needsTileEdges, payloadGroups, payloadWins, reductionKey, resolveReduction } from './core/reduction.js';
export type { BuiltinReduction, ChannelOp, ChannelOps, Channels, CustomReduction, PayloadGroup, EdgePolicy, PyramidReferenceReduction, Reduction, ReferenceReduction, ResolvedReduction, TileEdgePolicy } from './core/reduction.js';
export { maxViolations, spdChainReference, spdReference } from './core/spd.js';
export { checkLevels, levelSize, maxTileLevels, pyramidTolerance, reducePyramid, tileDispatch } from './core/pyramid.js';
export type { PyramidLevel, PyramidReferenceOptions, TileDispatch } from './core/pyramid.js';
export { isSumOp, scalarBlock, scalarCombineDepth, scalarCombineReference, scalarDepth, scalarInvocations, scalarOps, scalarReference, scalarSide, scalarWeightChannel, scalarWorkgroups } from './core/scalar.js';
export type { ScalarCombineReferenceOptions, ScalarOp, ScalarReduce, ScalarReferenceOptions, ScalarReferenceResult } from './core/scalar.js';
