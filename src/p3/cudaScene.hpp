#ifndef __CUDA_SCENE
#define __CUDA_SCENE

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

	// Material
	int N_material;
	float *ambient; // float3 * N
	float *diffuse; // float3 * N
	float *specular; // float3 * N

	// Light
	int N_light;
	float *light_pos;
	float *light_col;

	// Camera
	float cam_position[3];
	float cam_orientation[4];
	float fov;
	float aspect;
	float near_clip;
	float far_clip;

	int width;
	int height;
};

#endif

