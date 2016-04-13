#ifndef _462_SCENE_GEOMETRY_HPP_
#define _462_SCENE_GEOMETRY_HPP_

#include "math/vector.hpp"
#include "math/color.hpp"
#include "math/quaternion.hpp"
#include "scene/ray.hpp"
#include "scene/csgintervals.hpp"

namespace _462 {
//represents an intersection between a ray and a geometry
struct Intersection{
	Vector3 hit_point;
	Vector3 normal;
	real_t beta;
	real_t gamma;
	real_t alpha;
	Vector2 tex_coord;
	int hit_triangle;
	int hit_geometry; // For CSG only
};

struct InterpolatedMaterial {
	Color3 ambient;
	Color3 diffuse;
	Color3 specular;
	Color3 tex_color;
	real_t refractive_index;
	real_t shininess;
};


class Geometry
{
public:
    Geometry();
    virtual ~Geometry();
    /*
       World transformation are applied in the following order:
       1. Scale
       2. Orientation
       3. Position
    */

    // The world position of the object.
    Vector3 position;

    // The world orientation of the object.
    // Use Quaternion::to_matrix to get the rotation matrix.
    Quaternion orientation;

    // The world scale of the object.
    Vector3 scale;

    // Forward transformation matrix
    Matrix4 mat;

    // Inverse transformation matrix
    Matrix4 invMat;
    // Normal transformation matrix
    Matrix3 normMat;
    bool isBig;
    /**
     * Renders this geometry using OpenGL in the local coordinate space.
     */
    virtual void render() const = 0;
	virtual real_t intersect_ray(Ray &ray, Intersection &intersection) const = 0;
	virtual void interpolate_material (Intersection &intersection, InterpolatedMaterial &material) const = 0;
	virtual void intersect_ray_interval(Ray &ray, CSGIntervals &interval) const = 0;
	virtual void bump_normal(Intersection &intersection) const = 0;

    virtual bool initialize();
    virtual bool post_initialize();
};
}
#endif

