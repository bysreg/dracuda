#ifndef _462_PHYSICS_MODEL_HPP_
#define _462_PHYSICS_MODEL_HPP_

#include "body.hpp"
#include "math/vector.hpp"
#include "math/quaternion.hpp"
#include "scene/model.hpp"

namespace _462 {

class ModelBody : public Body
{
public:
    Model* model;

    ModelBody( Model* geom );
    virtual ~ModelBody() { }
    virtual Vector3 step_position( real_t dt, real_t motion_damping );
    virtual Vector3 step_orientation( real_t dt, real_t motion_damping );
    virtual void apply_force( const Vector3& f, const Vector3& offset );
};

}

#endif


