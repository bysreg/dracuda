/**
 * @file pool_controller.cpp
 * @brief Pool Controller class
 *
 * @author Mahardiansyah Kartika (mkartika)
 */

#ifndef _462_APPLICATION_POOLCONTROLLER_HPP_
#define _462_APPLICATION_POOLCONTROLLER_HPP_

#include "application/application.hpp"
#include "p3/spherebody.hpp"
#include "scene/triangle.hpp"

namespace _462 {

class Scene;

class PoolController
{
public:
    PoolController();
    ~PoolController();

	void handle_event(const SDL_Event& event, bool is_pool);
	void update(real_t dt, bool is_pool);

	void initialize(Scene *scene);
	real_t convert_to_yaw(Quaternion q);

private:
	SphereBody* cue_ball;
	Triangle* arrow;

	// power controller
	bool is_key_down = false;
	bool is_k_key_down = false;
	bool is_l_key_down = false;
	bool is_can_hit = true;
	real_t elapsed_time = 0;
};

} /* _462 */

#endif /* _462_APPLICATION_POOLCONTROLLER_HPP_ */

