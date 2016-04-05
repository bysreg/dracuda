/**
 * @file triangle.hpp
 * @brief Class definition for Triangle.
 *
 * @author Eric Butler (edbutler)
 */

#ifndef _462_SCENE_TRIANGLE_HPP_
#define _462_SCENE_TRIANGLE_HPP_

#include "scene/geometry.hpp"

namespace _462 {

/**
 * a triangle geometry.
 * Triangles consist of 3 vertices. Each vertex has its own position, normal,
 * texture coordinate, and material. These should all be interpolated to get
 * values in the middle of the triangle.
 * These values are all in local space, so it must still be transformed by
 * the Geometry's position, orientation, and scale.
 */
class Triangle : public Geometry
{
public:

    struct Vertex
    {
        // note that position and normal are in local space
        Vector3 position;
        Vector3 normal;
        Vector2 tex_coord;
        const Material* material;
    };

    // the triangle's vertices, in CCW order
	bool post_initialize();
    Vertex vertices[3];
	real_t intersect_ray(Ray &ray, Intersection &intersection) const;
	void intersect_ray_interval(Ray &ray, CSGIntervals &interval) const;
	void interpolate_material (Intersection &intersection, InterpolatedMaterial &material) const;
	void bump_normal(Intersection &intersection) const;

    Triangle();
    virtual ~Triangle();
    virtual void render() const;
};


} /* _462 */

#endif /* _462_SCENE_TRIANGLE_HPP_ */
