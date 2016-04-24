#ifndef __CUDA_SCENE
#define __CUDA_SCENE

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>

struct cudaScene
{
	// Geometry
	int N;
	float *position; // float3 * N
	float *rotation; // float4 * N

	// Material
	int N_material;
	float *diffuse; // float3 * N
	float *data;

	// Camera
	float cam_position[3];
	float cam_orientation[4];
	float fov;
	float aspect;
	float near_clip;
	float far_clip;

	int width;
	int height;
	curandState *curand;
};

#endif

