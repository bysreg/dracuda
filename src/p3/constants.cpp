#include "constants.hpp"

PoolConstants poolConstants;

#define INFINITY 100000.0

void initialize_constants()
{
	poolConstants.sphere_colors[0] = {0.20, 0.00, 0.00};
	poolConstants.sphere_colors[1] = {0.20, 0.20, 0.00};
	poolConstants.sphere_colors[2] = {0.00, 0.10, 0.20};
	poolConstants.sphere_colors[3] = {0.30, 0.25, 0.20};
	poolConstants.plane_colors[0] = {0.0, 0.4, 0.0};
	poolConstants.normals[0] = {0.0, 1.0, 0.0};
	poolConstants.lower_bounds[0] = {-10, -INFINITY, -10};
	poolConstants.upper_bounds[0] = {10, INFINITY, 10};
	poolConstants.positions[0] = 0.0;
	poolConstants.plane_axes[0] = 1;
	
}
