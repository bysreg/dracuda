#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cuda.h>
#include <cuda_runtime.h>

#define PLANES 1
#define SPHERES 4



struct PoolConstants
{
	int plane_axes[PLANES];

	float3 sphere_colors[SPHERES];
	float3 plane_colors[PLANES];
	float3 normals[PLANES];
	float3 lower_bounds[PLANES];
	float3 upper_bounds[PLANES];
	float positions[PLANES];
};

extern PoolConstants poolConstants;

extern void initialize_constants();

#endif
