#ifndef POOL_SCENE_HPP
#define POOL_SCENE_HPP

#include "math/vector.hpp"
#include "math/quaternion.hpp"
#include "math/camera.hpp"
#include "cudaScene.hpp"
#include "constants.hpp"

#include "PoolScene.hpp"

struct Ball {
	Vector3 position;
	Quaternion orientation;
	Vector3 acceleration;
	Vector3 velocity;
};

struct PoolScene {
	Ball balls[SPHERES];
	Camera camera;
	float time;
	void initialize();
	void update(float delta_time);
	void toDataBuffer(float *buffer);
	void toCudaScene(CudaScene &scene);
};

#endif
