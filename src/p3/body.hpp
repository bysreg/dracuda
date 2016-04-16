#ifndef _462_PHYSICS_BODY_HPP_
#define _462_PHYSICS_BODY_HPP_

#include "math/vector.hpp"
#include "math/quaternion.hpp"
#include "scene/bound.hpp"
#include <exception>
#include <iostream>

namespace _462 {

struct State
{
	Vector3 pos;
	Vector3 vel;
	Quaternion rot;
	Vector3 ang_vel;
};

struct Derivative
{
	Vector3 d_pos;
	Vector3 d_vel;
	Quaternion d_rot;
	Vector3 d_ang_vel;
};

class Body
{
public:
	int id;
    int type;
    Vector3 position;
    Quaternion orientation;
    Vector3 velocity;
    Vector3 angular_velocity;

	Bound bound;

    virtual ~Body() { }
    virtual Vector3 step_position( real_t dt, real_t motion_damping ) = 0;
    virtual Vector3 step_orientation( real_t dt, real_t motion_damping ) = 0;
    virtual void apply_force( const Vector3& f, const Vector3& offset ) = 0;
};

}

#endif
