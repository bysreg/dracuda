#ifndef _462_SCENE_CSG_HPP_
#define _462_SCENE_CSG_HPP_

#include "scene/geometry.hpp"
#include "scene/csgintervals.hpp"
#include "scene/ray.hpp"

namespace _462 {

enum CSGOperation {
	UNION,
	DIFFERENCE,
	INTERSECTION
};

struct CSGNode
{
	int left;
	int right;
	Geometry *g;
	CSGOperation op;
	bool is_negative;
};

class CSG : public Geometry
{
public:
	CSG();

	typedef std::vector<Geometry *> GeometryList;
	typedef std::vector<CSGNode> CSGNodeList;
	
	GeometryList geometries;
	bool initialize();
	bool post_initialize();
	CSGNodeList nodes;

	real_t intersect_ray(Ray &ray, Intersection &intersection) const;
	void interpolate_material (Intersection &intersection, InterpolatedMaterial &i_material) const;
	void intersect_ray_interval(Ray &ray, CSGIntervals &interval) const;
	void bump_normal(Intersection &intersection) const;

	void calculate_interval(int node, Ray &ray, CSGIntervals *intervals) const;
	int new_node();
	void set_is_negative(CSGNode &n, bool is_negative);

	virtual ~CSG();
	virtual void render() const;
};

}
#endif
