/**
 * @file model.cpp
 * @brief Model class
 *
 * @author Eric Butler (edbutler)
 * @author Zeyang Li (zeyangl)
 */

#include "scene/model.hpp"
#include "scene/material.hpp"
#include "application/opengl.hpp"
#include "scene/triangle.hpp"
#include "scene/utils.hpp"
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>

#include "scene/meshtree.hpp"

namespace _462 {

Model::Model() : mesh( 0 ), material( 0 ) { }
Model::~Model() { 
	delete kdtree;
}

void Model::render() const
{
    if ( !mesh )
        return;
    if ( material )
        material->set_gl_state();
    mesh->render();
    if ( material )
        material->reset_gl_state();
}
bool Model::initialize()
{
    Geometry::initialize();
    return true;
}

bool Model::post_initialize()
{
	calculate_bound();
	kdtree = new KDTree(mesh, bound);

	// create tree
	std::cout << "Start creating Tree" << std::endl;
	tree = new MeshTree(mesh, mat);
	std::cout << "Done creating Tree" << std::endl;

	return true;
}

real_t Model::intersect_ray(Ray &ray, Intersection &intersection) const
{
	if (!bound.intersects(ray))
		return -1;

	MeshVertex v0, v1, v2;
	real_t time = kdtree->intersect(0, ray, intersection);
	if (time > EPS) {
		const MeshTriangle *triangle = mesh->get_triangles() + intersection.hit_triangle;
		v0 = mesh->get_vertices()[triangle->vertices[0]];
		v1 = mesh->get_vertices()[triangle->vertices[1]];
		v2 = mesh->get_vertices()[triangle->vertices[2]];
		real_t beta = intersection.beta;
		real_t gamma = intersection.gamma;
		intersection.normal = normalize(normMat * (v0.normal * (1 - beta - gamma) + v1.normal * beta + v2.normal * gamma));
		intersection.tex_coord = v0.tex_coord * (1 - beta - gamma) + v1.tex_coord * beta + v2.tex_coord * gamma;
	}
	return time;
}

void Model::bump_normal(Intersection &intersection) const
{
	if (material->bump.data) {
		MeshTriangle t = mesh->get_triangles()[intersection.hit_triangle];
		const MeshVertex *v = mesh->get_vertices();
		
		Vector2 t10 = v[t.vertices[1]].tex_coord - v[t.vertices[0]].tex_coord;
		Vector2 t20 = v[t.vertices[2]].tex_coord - v[t.vertices[0]].tex_coord;
		Vector3 a1 = v[t.vertices[1]].position - v[t.vertices[0]].position;
		Vector3 a2 = v[t.vertices[2]].position - v[t.vertices[0]].position;
		Vector3 T, B;
		for (int i = 0; i < 3; i++) {
			T[i] = (a1[i] * t20.y - a2[i] * t10.y) / (t10.x * t20.y - t20.x * t10.y);
			B[i] = (a1[i] * t20.x - a2[i] * t10.x) / (t10.y * t20.x - t20.y * t10.x);
		}
		real_t bump_u = material->bump.sample_bump_u(intersection.tex_coord);
		real_t bump_v = material->bump.sample_bump_v(intersection.tex_coord);
		intersection.normal = normalize(intersection.normal - T * bump_u - B * bump_v);
	}
}

void Model::intersect_ray_interval(Ray &ray, CSGIntervals &interval) const
{
	if (!bound.intersects(ray))
		return;
	std::vector <double> times;
	
	for (unsigned int i = 0; i < mesh->num_triangles(); i++) {
		real_t beta, gamma, time;
		MeshVertex v0, v1, v2;
		const MeshTriangle *triangle = mesh->get_triangles() + i;
		v0 = mesh->get_vertices()[triangle->vertices[0]];
		v1 = mesh->get_vertices()[triangle->vertices[1]];
		v2 = mesh->get_vertices()[triangle->vertices[2]];
		bool ret = TriangleIntersection(ray, v0.position, v1.position, v2.position, beta, gamma, time);
		if (ret)
			times.push_back(time);
	}
	std::sort(times.begin(), times.end());
	for (unsigned int i = 0; i < times.size(); i+=2) {
		Interval interv;
		interv.min = times[i];
		interv.max = times[i+1];
		interval.add(interv);
	}
}
void Model::interpolate_material (Intersection &intersection, InterpolatedMaterial &i_material) const
{
	i_material.ambient = material->ambient;
	i_material.diffuse = material->diffuse;
	i_material.specular = material->specular;
	i_material.tex_color = material->texture.sample(intersection.tex_coord);
	i_material.refractive_index = material->refractive_index;
	i_material.shininess = material->shininess;
}

void Model::calculate_bound()
{
	const MeshVertex *vertices = mesh->get_vertices();
	double lx, ly, lz, ux, uy, uz;
	lx = INFINITY;
	ly = INFINITY;
	lz = INFINITY;
	ux = -INFINITY;
	uy = -INFINITY;
	uz = -INFINITY;
	for (unsigned int i = 0; i < mesh->num_vertices(); i++) {
		Vector3 v = vertices[i].position;
//		std::cout << v << std::endl;
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
	bound.lower.x = lx;
	bound.lower.y = ly;
	bound.lower.z = lz;
	
	bound.upper.x = ux;
	bound.upper.y = uy;
	bound.upper.z = uz;
}

} /* _462 */
	/*
	real_t min_time = -1;
	bool intersected = false;
	for (unsigned int i = 0; i < mesh->num_triangles(); i++) {
		real_t beta, gamma, time;
		MeshVertex v0, v1, v2;
		const MeshTriangle *triangle = mesh->get_triangles() + i;
		v0 = mesh->get_vertices()[triangle->vertices[0]];
		v1 = mesh->get_vertices()[triangle->vertices[1]];
		v2 = mesh->get_vertices()[triangle->vertices[2]];
		bool ret = TriangleIntersection(ray, v0.position, v1.position, v2.position, beta, gamma, time);
		if (ret && (!intersected || time < min_time)) {
			min_time = time;
			intersection.normal = v0.normal * (1 - beta - gamma) + v1.normal * beta + v2.normal * gamma;
			intersection.normal = normalize(normMat * intersection.normal);
			intersection.tex_coord = v0.tex_coord * (1 - beta - gamma) + v1.tex_coord * beta + v2.tex_coord * gamma;
			intersection.beta = beta;
			intersection.gamma = gamma;
			intersection.alpha = 1 - beta - gamma;
			intersected = true;
		}
	}
	if (intersected) {
		return min_time;
	} else {
		return -1;
	}
	*/
