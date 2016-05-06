#include "constants.hpp"

PoolConstants poolConstants;

#define INFINITY 100000.0

void initialize_constants()
{
	poolConstants.sphere_colors[0] = {0.29, 0.27, 0.25};
	poolConstants.sphere_colors[1] = {0.20, 0.20, 0.00};
	poolConstants.sphere_colors[2] = {0.00, 0.10, 0.20};
	poolConstants.sphere_colors[3] = {0.20, 0.00, 0.00};
	poolConstants.sphere_colors[4] = {0.05, 0.00, 0.10};
	poolConstants.sphere_colors[5] = {0.30, 0.10, 0.02};
	poolConstants.sphere_colors[6] = {0.00, 0.20, 0.02};
	poolConstants.sphere_colors[7] = {0.15, 0.03, 0.01};
	poolConstants.sphere_colors[8] = {0.0, 0.00, 0.00};
	for (int i = 9; i < SPHERES; i++) {
		poolConstants.sphere_colors[i] = poolConstants.sphere_colors[i - 8];
	}
	poolConstants.plane_colors[0] = {0.0, 0.4, 0.0};
	poolConstants.normals[0] = {0.0, 1.0, 0.0};
	float w = TABLE_WIDTH;
	float h = TABLE_HEIGHT;

	poolConstants.lower_bounds[0] = {-w, -INFINITY, -h};
	poolConstants.upper_bounds[0] = {w, INFINITY, h};

	poolConstants.positions[0] = 0.0;
	poolConstants.plane_axes[0] = 1;
	
}
