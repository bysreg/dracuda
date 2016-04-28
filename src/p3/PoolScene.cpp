#include "PoolScene.hpp"
#include "cudaScene.hpp"
#include "constants.hpp"

Vector3 ball_initial_positions[SPHERES] = {
	Vector3(1.0, 1.0, 1.0),
	Vector3(-4.0, 1.0, 0.0),
	Vector3(0.0, 1.0, 3.0),
	Vector3(5.0, 1.0, -2.0)
};

Vector3 velocity_acc[SPHERES];
int times[SPHERES];


void PoolScene::initialize()
{
	for (size_t i = 0; i < SPHERES; i++) {
		balls[i].position = ball_initial_positions[i];
		balls[i].orientation = Quaternion::Identity();
	}
	balls[2].velocity = Vector3(0.9, 0, -0.80);
	camera.fov = 0.785;
	camera.aspect = (WIDTH + 0.0) / (HEIGHT + 0.0);
	camera.near_clip = 0.01;
	camera.far_clip = 200.0;
}

void PoolScene::toDataBuffer(float *buffer)
{
	for (int i = 0; i < SPHERES; i++) {
		balls[i].position.to_array(buffer + 4 * SPHERES + 3 * i);
		balls[i].orientation.to_array(buffer + 4 * i);
	}
}

void PoolScene::toCudaScene(CudaScene &scene)
{
	camera.position.to_array(scene.cam_position); 
	camera.orientation.to_array(scene.cam_orientation);
}

void PoolScene::update (float delta_time)
{
	time += delta_time;
	float time_scale = 0.5;
	for (int i = 0; i < SPHERES; i++) {
		velocity_acc[i] = Vector3(0, 0, 0);
		times[i] = 0;
	}

	// Collision between spheres
	for (int i = 0; i < SPHERES; i++) {
		for (int j = i + 1; j < SPHERES; j++) {
			Vector3 dist = -balls[i].position + balls[j].position;
			Vector3 vel1 = balls[i].velocity - balls[j].velocity;
			if (length(dist) < 2) {
				Vector3 vel2 = normalize(dist) * dot(normalize(dist), vel1);
				Vector3 u2 = balls[j].velocity + vel2;
				times[i] ++;
				times[j] ++;
				velocity_acc[i] += balls[i].velocity + balls[j].velocity - u2;
				velocity_acc[j] += u2;
			}
		}
	}

	float width = 5.0, height = 5.0;
	// Collision with walls
	for (int i = 0; i < SPHERES; i++) {
		if (balls[i].position.x < -width) {
			velocity_acc[i] += Vector3(fabs(balls[i].velocity.x), 0, balls[i].velocity.z);
			times[i] ++;
		}
		if (balls[i].position.x > width) {
			velocity_acc[i] += Vector3(-fabs(balls[i].velocity.x), 0, balls[i].velocity.z);
			times[i] ++;
		}
		if (balls[i].position.z < -height) {
			velocity_acc[i] += Vector3(balls[i].velocity.x, 0, fabs(balls[i].velocity.z));
			times[i] ++;
		}
		if (balls[i].position.z > height) {
			velocity_acc[i] += Vector3(balls[i].velocity.x, 0, -fabs(balls[i].velocity.z));
			times[i] ++;
		}
	}

	for (int i = 0; i < SPHERES; i++) {
		if (times[i] > 0) {
			balls[i].velocity = velocity_acc[i] / times[i];
		}
	}
	// Update position & orientation;
	for (int i = 0; i < SPHERES; i++) {
		Vector3 distance = balls[i].velocity * delta_time;
		balls[i].position += distance;
		Vector3 axis = normalize(Vector3(distance.z, 0, -distance.x));
		Quaternion rotation = Quaternion(axis, length(distance));
		if (length(distance) <= 0) {
			rotation = Quaternion::Identity();
		}
		balls[i].orientation = balls[i].orientation * rotation;
	}

}


	/*
	Vector3 pos(8 * sin(time * time_scale), 4, 8 * cos(time * time_scale));
	Vector3 dir = -normalize(pos);
	Vector3 look(0, 0, -1);
	Vector3 up(0, 1, 0);
	Vector3 v = dir + up * -dot(up, dir);
	Quaternion q = FromToRotation(look, v);
	Quaternion ret = FromToRotation(v, dir) * q;
	pos = Vector3(0, 10, 0);
	ret = Quaternion(-0.707, 0.707, 0, 0);
	pos.to_array(cscene.cam_position);
	ret.to_array(cscene.cam_orientation);
	*/
