#ifndef _462_SCENE_ANIMATOR_HPP_
#define _462_SCENE_ANIMATOR_HPP_

#include "scene/geometry.hpp"
#include "math/vector.hpp"

namespace _462 {

class Animator
{
	public:
	Geometry *g;
	virtual void update(real_t time) = 0;
};

class BounceAnimator : public Animator
{
	public:
	Vector3 original_position;
	Vector3 acceleration;
	real_t interval;
	real_t phase;
	void update(real_t time);
};

}

#endif
