#ifndef __CUDA_SCENE
#define __CUDA_SCENE

struct cudaScene
{
	// Geometry
	float *position; // float3 * N
	float *rotation; // float4 * N
	float *scale; // float3 * N
	int *type; // int * N
	int *material; // int * N
	float *radius; // float * N

	// Material
	float *ambient; // float3 * N
	float *diffuse; // float3 * N
	float *specular; // float3 * N
	// Camera
	float cam_position[3];
	float cam_orientation[4];
	float fov;
	float aspect;
	float near_clip;
	float far_clip;
	int width;
	int height;
	int N;
};

#endif

