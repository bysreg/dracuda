#include "scene/csg.hpp"

namespace _462 {

CSG::CSG()
{
}


void CSG::set_is_negative(CSGNode &n, bool is_negative)
{
	if (n.left == -1) {
		n.is_negative = is_negative;
	} else {
		set_is_negative(nodes[n.left], is_negative);
		if (n.op == DIFFERENCE_OP)
			set_is_negative(nodes[n.right], !is_negative);
		else
			set_is_negative(nodes[n.right], is_negative);
	}
}

CSG::~CSG()
{
	for (unsigned int i = 0; i < geometries.size(); i++)
		delete geometries[i];
}

void CSG::render() const
{
}

bool CSG::initialize()
{
	set_is_negative(nodes[0], false);
	bool ret = Geometry::initialize();
	for (unsigned int i = 0; i < geometries.size(); i++)
		ret &= geometries[i]->initialize();
	return ret;
}

bool CSG::post_initialize()
{
	bool ret = true;
	for (unsigned int i = 0; i < geometries.size(); i++)
		ret &= geometries[i]->post_initialize();
	return ret;
}

int CSG::new_node()
{
	CSGNode node;
	node.left = -1;
	node.right = -1;
	node.is_negative = false;
	nodes.push_back(node);
	return nodes.size() - 1;
}

void CSG::bump_normal(Intersection &intersection) const
{
	nodes[intersection.hit_geometry].g->bump_normal(intersection);
}

real_t CSG::intersect_ray(Ray &ray, Intersection &intersection) const
{
	CSGIntervals *intervals = new CSGIntervals[nodes.size()]();
	
	for (int i = nodes.size() - 1; i >= 0; i--)
		calculate_interval(i, ray, intervals);

	if (intervals[0].size() == 0) {
		delete[] intervals;
		return -1;
	}

	Interval hit_interval = intervals[0][0];
	for (unsigned int i = 0; i < nodes.size(); i++)
		intervals[i].clear();
	delete[] intervals;
	real_t time = hit_interval.min;
	bool is_negative = nodes[hit_interval.min_geometry].is_negative;
	Geometry *g = nodes[hit_interval.min_geometry].g;

	Vector4 ray_d = Vector4(ray.d.x, ray.d.y, ray.d.z, 0);
	Vector4 ray_e = Vector4(ray.e.x, ray.e.y, ray.e.z, 1);
	Ray transformed_ray;
	transformed_ray.d = (g->invMat * ray_d).xyz();
	transformed_ray.e = (g->invMat * ray_e).xyz();
	real_t t = g->intersect_ray(transformed_ray, intersection);

	if (is_negative) {
		transformed_ray.e += transformed_ray.d * t;
		t = g->intersect_ray(transformed_ray, intersection);
		intersection.normal = -intersection.normal;
	}

	intersection.hit_geometry = hit_interval.min_geometry;
	return time;
}

void CSG::intersect_ray_interval(Ray &ray, CSGIntervals &interval) const
{
}

void CSG::calculate_interval(int node, Ray &ray, CSGIntervals *intervals) const
{
	CSGIntervals &interval = intervals[node];
	const CSGNode &csg_node = nodes[node];
	Geometry *g = csg_node.g;

	Vector4 ray_d = Vector4(ray.d.x, ray.d.y, ray.d.z, 0);
	Vector4 ray_e = Vector4(ray.e.x, ray.e.y, ray.e.z, 1);
	Ray transformed_ray;
	if (csg_node.left == -1) {
		transformed_ray.d = (g->invMat * ray_d).xyz();
		transformed_ray.e = (g->invMat * ray_e).xyz();
		g->intersect_ray_interval(transformed_ray, interval);
		for (int i = 0; i < interval.size(); i++) {
			interval[i].min_geometry = node;
			interval[i].max_geometry = node;
		}
	} else {
		switch(csg_node.op) {
			case UNION:
				intervals_union (intervals[csg_node.left], intervals[csg_node.right], interval);
				break;
			case DIFFERENCE_OP:
				difference(intervals[csg_node.left], intervals[csg_node.right], interval);
				break;
			case INTERSECTION:
				intersection(intervals[csg_node.left], intervals[csg_node.right], interval);
				break;
		}
	}
}

void CSG::interpolate_material (Intersection &intersection, InterpolatedMaterial &i_material) const {
	nodes[intersection.hit_geometry].g->interpolate_material(intersection, i_material);
}
}

