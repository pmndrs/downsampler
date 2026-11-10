// Backend-neutral core. Backends live in `downsampler/webgpu` (and later others);
// the three.js adapter lives in `downsampler/three`.
export { planBatches, mipSize, mipLevelCountFor, tileSize, workgroupSide } from './core/plan.js';
export type { Batch, PlanOptions, OddSizePolicy } from './core/plan.js';
export { formats, integerDomains, isIntegerDomain, isSupportedFormat, quantize, roundHalfEven, sampledChannels, toHalf, fromHalf } from './core/formats.js';
export type { FormatInfo, IntegerQuantization, Quantization, SupportedFormat } from './core/formats.js';
export { copyReference, reduceReference, tolerance } from './core/reference.js';
export type { ReferenceLevel, ReferenceOptions } from './core/reference.js';
export { footprint, foldChannel, isBuiltinReduction, reductionKey } from './core/reduction.js';
export type { BuiltinReduction, CustomReduction, Reduction, ReferenceReduction } from './core/reduction.js';
