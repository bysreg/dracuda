/**
 * @file model.hpp
 * @brief Model class
 *
 * @author Eric Butler (edbutler)
 */

#ifndef _462_SCENE_MODEL_HPP_
#define _462_SCENE_MODEL_HPP_

#include "scene/geometry.hpp"
#include "scene/mesh.hpp"
#include "scene/kdtree.hpp"

namespace _462 {

/**
 * A mesh of triangles.
 */
class Model : public Geometry
{
public:

    const Mesh* mesh;
    const Material* material;
	Bound bound;
	KDTree *kdtree;

    Model();
    virtual ~Model();

    virtual void render() const;
    virtual bool initialize();
	bool post_initialize();
	void calculate_bound();
	real_t intersect_ray(Ray &ray, Intersection &intersection) const;
	void intersect_ray_interval(Ray &ray, CSGIntervals &interval) const;
	void bump_normal(Intersection &intersection) const;
	void interpolate_material (Intersection &intersection, InterpolatedMaterial &material) const;
};


} /* _462 */

#endif /* _462_SCENE_MODEL_HPP_ */

