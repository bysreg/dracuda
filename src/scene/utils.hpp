#ifndef _462_SCENE_UTILS_HPP_
#define _462_SCENE_UTILS_HPP_
#include "math/math.hpp"

#include "math/vector.hpp"
#include "scene/ray.hpp"

namespace _462 {
	bool solve_quadratic(real_t *x1,real_t *x2, real_t a, real_t b, real_t c);
	real_t solve_time(real_t a,real_t b,real_t c);
	bool TriangleIntersection (Ray &ray, Vector3 v0, Vector3 v1, Vector3 v2, real_t &beta, real_t &gamma, real_t &time);
	bool solve_time2(real_t a, real_t b, real_t c, real_t &x1, real_t &x2);
}

#endif
