#include "scene/animator.hpp"

namespace _462 {

void BounceAnimator::update(real_t time)
{
	time += phase;
	int cycles = time / interval;
	time -= cycles * interval;
	if (time > interval / 2) {
		real_t rtime = interval - time;
		g->position = original_position + rtime * rtime * acceleration / 2;
	} else {
		g->position = original_position + time * time * acceleration / 2;
	}
}

}
