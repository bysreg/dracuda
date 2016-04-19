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
	float *scale; // float3 * N
	int *type; // int * N
	int *material; // int * N
	float *radius; // float * N
	float *vertex0;
	float *vertex1;
	float *vertex2;

	// Material
	int N_material;
	float *ambient; // float3 * N
	float *diffuse; // float3 * N
	float *specular; // float3 * N

	// Light
	int N_light;
	float *light_pos;
	float *light_col;
	float *light_radius;
	float ambient_light_col[3];

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

