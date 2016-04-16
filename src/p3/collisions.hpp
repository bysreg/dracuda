#ifndef _462_COLLISIONS_HPP_
#define _462_COLLISIONS_HPP_

#include "scene/sphere.hpp"
#include "spherebody.hpp"
#include "trianglebody.hpp"
#include "planebody.hpp"
#include "modelbody.hpp"

namespace _462 {

class MeshTreeNode;

bool collides( SphereBody& body1, SphereBody& body2, real_t collision_damping );
bool collides( SphereBody& body1, TriangleBody& body2, real_t collision_damping );
bool collides( SphereBody& body1, PlaneBody& body2, real_t collision_damping );
bool collides( SphereBody& body1, ModelBody& body2, real_t collision_damping );

// additional functions
real_t relative_velocity(SphereBody& body1, Vector3 velocity, Vector3 position);
Vector3 damping(Vector3 velocity, real_t collision_damping);
bool is_within_barycentric(const Vector3 point, const Vector3 vertices[3]);
bool is_hit_triangle(SphereBody& body1, const Vector3 vertices[3]);
real_t minimum_distance(Vector3 p, Vector3 v, Vector3 w);
void collides_node(SphereBody& body1, ModelBody& body2, real_t collision_damping, MeshTreeNode* node, bool &is_collide);
}

#endif
