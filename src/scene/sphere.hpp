/**
 * @file sphere.hpp
 * @brief Class defnition for Sphere.
 *
 * @author Kristin Siu (kasiu)
 * @author Eric Butler (edbutler)
 */

#ifndef _462_SCENE_SPHERE_HPP_
#define _462_SCENE_SPHERE_HPP_

#include "scene/geometry.hpp"
#include "scene/csgintervals.hpp"

namespace _462 {

/**
 * A sphere, centered on its position with a certain radius.
 */

class Sphere : public Geometry
{
public:

    real_t radius;
    const Material* material;

    Sphere();
    virtual ~Sphere();
	bool post_initialize();
	real_t intersect_ray (Ray &ray, Intersection &intersection) const;
	void intersect_ray_interval(Ray &ray, CSGIntervals &interval) const;
	void interpolate_material (Intersection &intersection, InterpolatedMaterial &material) const;
	void bump_normal(Intersection &intersection) const;
    virtual void render() const;
};

} /* _462 */

#endif /* _462_SCENE_SPHERE_HPP_ */

