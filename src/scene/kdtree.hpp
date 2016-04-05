#ifndef _462_SCENE_KDTREE_HPP_
#define _462_SCENE_KDTREE_HPP_

#include "scene/mesh.hpp"
#include "scene/bound.hpp"
#include "scene/geometry.hpp"

namespace _462 {

struct KDTreeNode
{
	unsigned int left;
	unsigned int right;
	unsigned int axis;
	double split_point;
	Bound bound;
	unsigned int *triangles;
	unsigned int triangle_num;
	bool leaf;
};

class KDTree
{
public:
	KDTree(const Mesh *mesh, const Bound &bound);
	~KDTree();
	const Mesh *mesh;
	void split(unsigned int node);
	real_t intersect(unsigned int node, Ray &ray, Intersection &intersection);
	std::vector<KDTreeNode> nodes;
	Bound * bounds;
};

}
#endif
