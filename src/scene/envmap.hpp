#ifndef _462_SCENE_ENVMAP_HPP_
#define _462_SCENE_ENVMAP_HPP_

#include "scene/ray.hpp"
#include "scene/texture.hpp"

namespace _462 {

class EnvMap
{
public:
	EnvMap();
	std::string prefix;
	real_t size;
	bool enabled = false;
	void initialize();
	bool intersect(Ray &ray, Color3 &color);
	Texture posx;
	Texture posy;
	Texture posz;
	Texture negx;
	Texture negy;
	Texture negz;
	
};

}
#endif
