#ifndef __CUDA_SCENE
#define __CUDA_SCENE

#include "constants.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>

struct CudaScene
{
	// Sphere position & orientation
	float data[SPHERES * 7];

	// Camera
	float cam_position[3];
	float cam_orientation[4];
	float fov;
	float aspect;
	float near_clip;
	float far_clip;
	
	int y0; // render offset
	int render_height;
};

#endif

