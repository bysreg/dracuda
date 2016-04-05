#include "scene/kdtree.hpp"

#include <iostream>
#include "scene/utils.hpp"

using namespace std;
namespace _462 {

static const double resolution = 0.0001;

KDTree::KDTree (const Mesh *mesh, const Bound &bound)
{
	this->mesh = mesh;
	bounds = new Bound[mesh->num_triangles()];
	const MeshVertex *vertices = mesh->get_vertices();
	const MeshTriangle *triangles = mesh->get_triangles();
	double lx, ly, lz, ux, uy, uz;
	for (unsigned int i = 0; i < mesh->num_triangles(); i++)
	{
		lx = INFINITY;
		ly = INFINITY;
		lz = INFINITY;
		ux = -INFINITY;
		uy = -INFINITY;
		uz = -INFINITY;
		for (unsigned int j = 0; j < 3; j++) {
			Vector3 v = vertices[triangles[i].vertices[j]].position;
			if (v.x < lx)
				lx = v.x;
			if (v.y < ly)
				ly = v.y;
			if (v.z < lz)
				lz = v.z;
			if (v.x > ux)
				ux = v.x;
			if (v.y > uy)
				uy = v.y;
			if (v.z > uz)
				uz = v.z;
		}
		bounds[i].lower.x = lx;
		bounds[i].lower.y = ly;
		bounds[i].lower.z = lz;
		bounds[i].upper.x = ux;
		bounds[i].upper.y = uy;
		bounds[i].upper.z = uz;
	}
	// Build root
	KDTreeNode node;
	node.triangle_num = mesh->num_triangles();
	node.triangles = (unsigned int *)malloc(sizeof(unsigned int) * node.triangle_num);
	for (unsigned int i = 0; i < node.triangle_num; i++)
		node.triangles[i] = i;
	node.bound = bound;
	nodes.push_back(node);
	split(0);
}

KDTree::~KDTree()
{
	delete[] bounds;
	for (unsigned int i = 0; i < nodes.size(); i++) {
		if (nodes[i].leaf)
			free(nodes[i].triangles);
	}
}

real_t KDTree::intersect(unsigned int node, Ray &ray, Intersection &intersection)
{
	KDTreeNode &n = nodes[node];
	if (n.leaf) {
		real_t min_time = INFINITY;
		bool intersected = false;
		for (unsigned int i = 0; i < n.triangle_num; i++) {
			real_t beta, gamma, time;
			MeshVertex v0, v1, v2;
			const MeshTriangle *triangle = mesh->get_triangles() + n.triangles[i];
			v0 = mesh->get_vertices()[triangle->vertices[0]];
			v1 = mesh->get_vertices()[triangle->vertices[1]];
			v2 = mesh->get_vertices()[triangle->vertices[2]];
			bool ret = TriangleIntersection(ray, v0.position, v1.position, v2.position, beta, gamma, time);
			if (ret && (!intersected || time < min_time)) {
				Vector3 hit_point = ray.e + ray.d * time;
				if (n.bound.within(hit_point)) {
					min_time = time;
					intersection.hit_triangle = n.triangles[i];
					intersection.beta = beta;
					intersection.gamma = gamma;
					intersection.alpha = 1 - beta - gamma;
					intersected = true;
				}
			}
		}
		if (intersected) {
			return min_time;
		} else {
			return -1;
		}
	} else {
		KDTreeNode &leftnode = nodes[nodes[node].left];
		KDTreeNode &rightnode = nodes[nodes[node].right];
		bool left_intersect = leftnode.bound.intersects(ray);
		bool right_intersect = rightnode.bound.intersects(ray);
		real_t time;
		if (left_intersect && right_intersect) {
			if (ray.d[n.axis] > 0) {
				time = intersect(nodes[node].left, ray, intersection);
				if (time < EPS)
					return intersect(nodes[node].right, ray, intersection);
				else
					return time;
			} else {
				time = intersect(nodes[node].right, ray, intersection);
				if (time < EPS)
					return intersect(nodes[node].left, ray, intersection);
				else
					return time;
			}
		}
		if (left_intersect)
			return intersect(nodes[node].left, ray, intersection);
		if (right_intersect)
			return intersect(nodes[node].right, ray, intersection);
		return -1;
	}
	return -1;
}

void KDTree::split(unsigned int node)
{
	unsigned int left_num, right_num, shared_num;
	unsigned int best_left_size = 0, best_right_size = 0;
	unsigned int best_fom = ~0;
	double split_point = 0;
	unsigned int size = nodes[node].triangle_num;
	unsigned int split_axis = 0;
	for (unsigned int axis = 0; axis < 3; axis ++) {
		double left = nodes[node].bound.lower[axis];
		double right = nodes[node].bound.upper[axis];
		while (1) {
			left_num = 0;
			right_num = 0;
			shared_num = 0;
			double mid = (left + right) / 2;
			for (unsigned int i = 0; i < size; i++)
			{
				unsigned int t = nodes[node].triangles[i];
				if (bounds[t].upper[axis] < mid)
					left_num ++;
				else if (bounds[t].lower[axis] > mid)
					right_num ++;
				else
					shared_num ++;
			}
			unsigned int fom;
			if (right_num > left_num)
				fom = right_num - left_num + shared_num;
			else 
				fom = left_num - right_num + shared_num;
			if (fom < best_fom) {
				best_fom = fom;
				best_left_size = left_num + shared_num;
				best_right_size = right_num + shared_num;
				split_point = mid;
				split_axis = axis;
			}
			if (shared_num == size)
				break;
			if (right - left < resolution)
				break;
			if (right_num > left_num)
				left = mid;
			else if (right_num < left_num)
				right = mid;
			else
				break;
		}
	}
	nodes[node].axis = split_axis;
	if (best_fom != size) {
		KDTreeNode leftnode, rightnode;
		unsigned int current_nodeID = nodes.size();
		nodes[node].left = current_nodeID;
		nodes[node].right = current_nodeID + 1;
		leftnode.triangle_num = 0;
		rightnode.triangle_num = 0;
		leftnode.triangles = (unsigned int *)malloc(sizeof(unsigned int) * best_left_size);
		rightnode.triangles = (unsigned int *)malloc(sizeof(unsigned int) * best_right_size);
		for (unsigned int i = 0; i < size; i++) {
			unsigned int t = nodes[node].triangles[i];
			if (bounds[t].lower[split_axis] <= split_point)
				leftnode.triangles[leftnode.triangle_num++] = t;
			if (bounds[t].upper[split_axis] >= split_point)
				rightnode.triangles[rightnode.triangle_num++] = t;
		}

		free(nodes[node].triangles);
		nodes[node].triangles = 0;
		leftnode.bound = nodes[node].bound;
		rightnode.bound = nodes[node].bound;
		leftnode.bound.upper[split_axis] = split_point;
		rightnode.bound.lower[split_axis] = split_point;
		nodes.push_back(leftnode);
		nodes.push_back(rightnode);
		split(current_nodeID);
		split(current_nodeID + 1);

		nodes[node].leaf = false;
	} else {
		nodes[node].leaf = true;
	}
}

}
