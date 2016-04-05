#include "scene/envmap.hpp"

namespace _462 {

EnvMap::EnvMap()
{
}
void EnvMap::initialize()
{
	if (enabled) {
		posx.filename = prefix + "posx.png";
		posy.filename = prefix + "posy.png";
		posz.filename = prefix + "posz.png";
		negx.filename = prefix + "negx.png";
		negy.filename = prefix + "negy.png";
		negz.filename = prefix + "negz.png";
		posx.load();
		posy.load();
		posz.load();
		negx.load();
		negy.load();
		negz.load();
	}
}

bool EnvMap::intersect(Ray &ray, Color3 &color)
{
	if(!enabled)
		return false;
	double x, y, z, t;
	double tex_x, tex_y, tex_z;
	double min_t = INFINITY;
	if (ray.d.x != 0) {
		// Pos X
		t = (size - ray.e.x) / ray.d.x;
		y = ray.e.y + t * ray.d.y;
		z = ray.e.z + t * ray.d.z;
		if (y < size && y > -size && z < size && z > -size && t > 0 && t < min_t) {
			min_t = t;
			tex_y = 0.5 + y / size / 2;
			tex_z = 0.5 - z / size / 2;
			color = posx.sample(Vector2(tex_z, tex_y));
		}
		// Neg X
		t = (-size - ray.e.x) / ray.d.x;
		y = ray.e.y + t * ray.d.y;
		z = ray.e.z + t * ray.d.z;
		if (y < size && y > -size && z < size && z > -size && t > 0 && t < min_t) {
			min_t = t;
			tex_y = 0.5 + y / size / 2;
			tex_z = 0.5 + z / size / 2;
			color = negx.sample(Vector2(tex_z, tex_y));
		}
	}
	if (ray.d.y != 0) {
		// Pos Y
		t = (size - ray.e.y) / ray.d.y;
		x = ray.e.x + t * ray.d.x;
		z = ray.e.z + t * ray.d.z;
		if (x < size && x > -size && z < size && z > -size && t > 0 && t < min_t) {
			min_t = t;
			tex_x = 0.5 + x / size / 2;
			tex_z = 0.5 - z / size / 2;
			color = posy.sample(Vector2(tex_x, tex_z));
		}
		// Neg Y
		t = (-size - ray.e.y) / ray.d.y;
		x = ray.e.x + t * ray.d.x;
		z = ray.e.z + t * ray.d.z;
		if (x < size && x > -size && z < size && z > -size && t > 0 && t < min_t) {
			min_t = t;
			tex_x = 0.5 + x / size / 2;
			tex_z = 0.5 + z / size / 2;
			color = negy.sample(Vector2(tex_x, tex_z));
		}
	}

	if (ray.d.z != 0) {
		// Pos Z
		t = (size - ray.e.z) / ray.d.z;
		x = ray.e.x + t * ray.d.x;
		y = ray.e.y + t * ray.d.y;
		if (x < size && x > -size && y < size && y > -size && t > 0 && t < min_t) {
			min_t = t;
			tex_x = 0.5 + x / size / 2;
			tex_y = 0.5 + y / size / 2;
			color = posz.sample(Vector2(tex_x, tex_y));
		}
		// Neg Z
		t = (-size - ray.e.z) / ray.d.z;
		x = ray.e.x + t * ray.d.x;
		y = ray.e.y + t * ray.d.y;
		if (x < size && x > -size && y < size && y > -size && t > 0 && t < min_t) {
			min_t = t;
			tex_x = 0.5 - x / size / 2;
			tex_y = 0.5 + y / size / 2;
			color = negz.sample(Vector2(tex_x, tex_y));
		}
	}
	return (min_t != INFINITY);
}

}
