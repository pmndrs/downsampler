import { test } from 'node:test';
import assert from 'node:assert/strict';
import { optionalFeatures, requiredFeatures, requiredLimits } from '../../dist/webgpu/limits.js';

const adapter = ( features, limits = {} ) => ( { features: new Set( features ), limits: { maxStorageTexturesPerShaderStage: 8, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, ...limits } } );

test( 'requiredFeatures asks only for the optional features the adapter has', () => {

	assert.deepEqual( requiredFeatures( adapter( [] ) ), [] );
	assert.deepEqual( requiredFeatures( adapter( [ 'float32-filterable', 'timestamp-query', 'shader-f16' ] ) ), [ 'float32-filterable' ] );
	assert.deepEqual( requiredFeatures( adapter( [ ...optionalFeatures, 'subgroups' ] ) ), [ ...optionalFeatures ] );
	assert.deepEqual( [ ...optionalFeatures ].sort(), [ 'bgra8unorm-storage', 'float32-filterable', 'texture-formats-tier1' ] );

} );

test( 'requiredLimits: six storage textures at most, 256-invocation workgroups where offered', () => {

	assert.deepEqual( requiredLimits( adapter( [] ) ), { maxStorageTexturesPerShaderStage: 6, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } );
	assert.deepEqual( requiredLimits( adapter( [], { maxStorageTexturesPerShaderStage: 4, maxComputeInvocationsPerWorkgroup: 128, maxComputeWorkgroupSizeX: 128 } ) ), { maxStorageTexturesPerShaderStage: 4 } );

} );
