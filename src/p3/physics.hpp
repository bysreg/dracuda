#ifndef _462_PHYSICS_PHYSICS_HPP_
#define _462_PHYSICS_PHYSICS_HPP_

#include "math/math.hpp"
#include "math/quaternion.hpp"
#include "math/vector.hpp"

#include "p3/body.hpp"
#include "p3/spherebody.hpp"
#include "p3/trianglebody.hpp"
#include "p3/planebody.hpp"
#include "p3/modelbody.hpp"
#include "p3/spring.hpp"
#include "p3/collisions.hpp"

#include <vector>

namespace _462 {

class Physics
{
public:
    Vector3 gravity;
	real_t collision_damping;

    Physics();
    ~Physics();

    void step( real_t dt );
    void add_sphere( SphereBody* s );
    size_t num_spheres() const;
    void add_plane( PlaneBody* p );
    size_t num_planes() const;
    void add_triangle( TriangleBody* t );
    size_t num_triangles() const;
    void add_model( ModelBody* m );
    size_t num_models() const;
    void add_spring( Spring* s );
    size_t num_springs() const;

    void reset();

	// additional functions
	void check_collision();
	void update_geometries(real_t dt);

	// RK4
	void integrate(real_t dt);
	Derivative evaluate(SphereBody* body, const State &initial);
	Derivative evaluate(SphereBody* body, const State &initial, float dt, const Derivative &d);
	void apply_forces(SphereBody* body, const State &initial);
	Quaternion add_quaternion(Quaternion first_orientation, Vector3 delta_orientation);

    typedef std::vector< Spring* > SpringList;
    typedef std::vector< SphereBody* > SphereList;
    typedef std::vector< PlaneBody* > PlaneList;
    typedef std::vector< TriangleBody* > TriangleList;
    typedef std::vector< ModelBody* > ModelList;

    SpringList springs;
    SphereList spheres;
    PlaneList planes;
    TriangleList triangles;
    ModelList models;

private:
};

}

#endif

