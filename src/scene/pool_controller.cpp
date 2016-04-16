/**
 * @file pool_controller.cpp
 * @brief PoolController class
 *
 * @author Mahardiansyah Kartika (mkartika)
 * @author Eric Butler (edbutler)
 */

#include "scene/pool_controller.hpp"
#include "scene/scene.hpp"

namespace _462 {

PoolController::PoolController()
{
}

PoolController::~PoolController()
{
}

void PoolController::initialize(Scene *scene) {
	// get cue ball
	for (size_t i = 0; i < scene->get_physics()->num_spheres(); ++i) {
		SphereBody* ball = scene->get_physics()->spheres[i];

		// get cue ball
		if (ball->id == -1) { // cue ball id = -1
			cue_ball = ball;
			break;
		}
	}

	// get arrow
	for (size_t i = 0; i < scene->get_physics()->num_triangles(); ++i){
		TriangleBody* triangle_body = scene->get_physics()->triangles[i];

		if (triangle_body->id == -2) { // arrow triangle id = -2
			arrow = triangle_body->triangle;
			// remove from list
			scene->get_physics()->triangles.erase(scene->get_physics()->triangles.begin() + i);
			break;
		}
	}
}

void PoolController::handle_event(const SDL_Event& event, bool is_pool)
{
	if (is_pool) {
		switch (event.type)
		{
		case SDL_KEYDOWN:
			switch (event.key.keysym.sym)
			{
			case SDLK_o: // shoot ball
				if (is_can_hit) {
					is_key_down = true;
					elapsed_time = 0;
				}
				break;
			case SDLK_k: // rotate left 
				is_k_key_down = true;
				break;
			case SDLK_l: // rotate right
				is_l_key_down = true;
				break;
			}
			break;
		case SDL_KEYUP:
			real_t rad;

			switch (event.key.keysym.sym)
			{
			case SDLK_o:
				if (is_key_down) {
					is_key_down = false;
					
					rad = convert_to_yaw(arrow->orientation);
					cue_ball->velocity = Vector3(sin(rad), 0, cos(rad)) * (elapsed_time * 10);					
				}
				break;
			case SDLK_k: // rotate left 
				is_k_key_down = false;
				break;
			case SDLK_l: // rotate right
				is_l_key_down = false;
				break;
			}
			break;
		case SDL_MOUSEBUTTONDOWN:
			break;
		case SDL_MOUSEBUTTONUP:
			break;
		case SDL_MOUSEMOTION:
			break;
		default:
			break;
		}
	}
}

void PoolController::update(real_t dt, bool is_pool)
{
	if (is_pool) {
		if (is_key_down) {
			elapsed_time += dt;
			
			// set max
			if (elapsed_time > 1) elapsed_time = 1;
			
			// set arrow
			arrow->scale = Vector3(1, 1, 1 + elapsed_time*3);
		}

		// rotation
		real_t delta_rotation = 0.005;
		if (is_k_key_down) { // left
			arrow->orientation = normalize(Quaternion(arrow->orientation * Vector3::UnitY(), delta_rotation) * arrow->orientation);
		}
		if (is_l_key_down) { // right
			arrow->orientation = normalize(Quaternion(arrow->orientation * Vector3::UnitY(), -delta_rotation) * arrow->orientation);
		}

		// is can hit
		if (length(cue_ball->velocity) < 0.01) {
			if (!is_can_hit) { // from moving to static ball
				// show arrow
				arrow->scale = Vector3(1, 1, 1);
				arrow->position = cue_ball->position;
			}
			is_can_hit = true;
		} 
		else {
			if (is_can_hit) { // from static to moving ball
				// hide arrow
				arrow->position = Vector3(0, -1, 0);
			}
			is_can_hit = false;
		}
	}
}

real_t PoolController::convert_to_yaw(Quaternion q) {
	return atan2(2 * q.y*q.w - 2 * q.x*q.z, 1 - 2 * pow(q.y, 2) - 2 * pow(q.z, 2));
}
}
