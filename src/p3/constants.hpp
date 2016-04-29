#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define WIDTH 64
#define HEIGHT 48

#define PLANES 1
#define SPHERES 4
#define SOLIDS 2

#define TABLE_WIDTH 20.5
#define TABLE_HEIGHT 41.0
#define TABLE_EDGE 2



struct PoolConstants
{
	int plane_axes[PLANES];

	float3 sphere_colors[SPHERES];
	float3 plane_colors[PLANES];
	float3 normals[PLANES];
	float3 lower_bounds[PLANES];
	float3 upper_bounds[PLANES];
	float positions[PLANES];

	curandState *curand;
};

extern PoolConstants poolConstants;

extern void initialize_constants();

#endif
