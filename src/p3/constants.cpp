#include "constants.hpp"

PoolConstants poolConstants;

#define INFINITY 100000.0

void initialize_constants()
{
	float3 mat_color = {0.0, 0.4, 0.0};
	float3 edge_color = {0.13, 0.03, 0.02};
	float3 leng_color = {0.01, 0.09, 0.01};
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
	poolConstants.plane_colors[0] = mat_color;
	poolConstants.plane_colors[1] = edge_color;
	poolConstants.plane_colors[2] = edge_color;
	poolConstants.plane_colors[3] = edge_color;
	poolConstants.plane_colors[4] = edge_color;
	poolConstants.plane_colors[5] = leng_color;
	poolConstants.plane_colors[6] = leng_color;
	poolConstants.plane_colors[7] = leng_color;
	poolConstants.plane_colors[8] = leng_color;
	poolConstants.normals[0] = {0.0, 1.0, 0.0};
	poolConstants.normals[1] = {0.0, 1.0, 0.0};
	poolConstants.normals[2] = {0.0, 1.0, 0.0};
	poolConstants.normals[3] = {0.0, 1.0, 0.0};
	poolConstants.normals[4] = {0.0, 1.0, 0.0};
	poolConstants.normals[5] = {1.0, 0.0, 0.0};
	poolConstants.normals[6] = {-1.0, 0.0, 0.0};
	poolConstants.normals[7] = {0.0, 0.0, -1.0};
	poolConstants.normals[8] = {0.0, 0.0, 1.0};
	float w = TABLE_WIDTH;
	float h = TABLE_HEIGHT;
	float e = TABLE_EDGE;
	float q = 1.0;

	poolConstants.lower_bounds[0] = {-w, -INFINITY, -h};
	poolConstants.upper_bounds[0] = {w, INFINITY, h};
	poolConstants.positions[0] = 0.0;
	poolConstants.lower_bounds[1] = {-w - e, -INFINITY, -h - e};
	poolConstants.upper_bounds[1] = {-w, INFINITY, h + e};
	poolConstants.positions[1] = q;
	poolConstants.lower_bounds[2] = {w, -INFINITY, -h - e};
	poolConstants.upper_bounds[2] = {w + e, INFINITY, h + e};
	poolConstants.positions[2] = q;
	poolConstants.lower_bounds[3] = {-w, -INFINITY, h};
	poolConstants.upper_bounds[3] = {w, INFINITY, h + e};
	poolConstants.positions[3] = q;
	poolConstants.lower_bounds[4] = {-w, -INFINITY, -h - e};
	poolConstants.upper_bounds[4] = {w, INFINITY, -h};
	poolConstants.positions[4] = q;
	// X
	poolConstants.lower_bounds[5] = {-INFINITY, 0, -h};
	poolConstants.upper_bounds[5] = {INFINITY, q, h};
	poolConstants.positions[5] = -w;

	poolConstants.lower_bounds[6] = {-INFINITY, 0, -h};
	poolConstants.upper_bounds[6] = {INFINITY, q, h};
	poolConstants.positions[6] = w;

	// Z
	poolConstants.lower_bounds[7] = {-w, 0, -INFINITY};
	poolConstants.upper_bounds[7] = {w, q, INFINITY};
	poolConstants.positions[7] = -h;
	
	poolConstants.lower_bounds[8] = {-w, 0, -INFINITY};
	poolConstants.upper_bounds[8] = {w, q, INFINITY};
	poolConstants.positions[8] = h;
}
