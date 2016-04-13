/**
 * @file scene.cpp
 * @brief Function definitions for scenes.
 *
 * @author Eric Butler (edbutler)
 * @author Kristin Siu (kasiu)
 */

#include "scene/scene.hpp"
#include "math/random462.hpp"

#include "scene/csg.hpp"
#include "scene/sphere.hpp"
namespace _462 {


Geometry::Geometry():
    position(Vector3::Zero()),
    orientation(Quaternion::Identity()),
    scale(Vector3::Ones())
{

}

Geometry::~Geometry() { }


bool Geometry::initialize()
{
    make_inverse_transformation_matrix(&invMat, position, orientation, scale);
    make_transformation_matrix(&mat, position, orientation, scale);
    make_normal_matrix(&normMat, mat);
    return true;
}

bool Geometry::post_initialize()
{
	make_inverse_transformation_matrix(&invMat, position, orientation, scale);
	make_transformation_matrix(&mat, position, orientation, scale);
	make_normal_matrix(&normMat, mat);
	return true;
}

SphereLight::SphereLight():
    position(Vector3::Zero()),
    color(Color3::White()),
    radius(real_t(0))
{
    attenuation.constant = 1;
    attenuation.linear = 0;
    attenuation.quadratic = 0;
}

real_t SphereLight::get_intensity(real_t time) const
{
	return 1.0 / (attenuation.constant
				+ attenuation.linear * time 
				+ attenuation.quadratic * time * time);
}

Vector3 SphereLight::sample() const
{
	Vector3 v = Vector3(random_gaussian(), random_gaussian(), random_gaussian());
	normalize(v);
	return v * radius + position;
}


Scene::Scene() : is_pool(false)
{
    reset();
}

Scene::~Scene()
{
    reset();
}

bool Scene::initialize()
{
    bool res = true;
    for (unsigned int i = 0; i < num_geometries(); i++)
        res &= geometries[i]->initialize();
	envmap.initialize();

    return res;
}

bool Scene::post_initialize()
{
	if (is_pool){
		pool_controller.initialize(this);
	}

	return true;
}

Geometry* const* Scene::get_geometries() const
{
    return geometries.empty() ? NULL : &geometries[0];
}

size_t Scene::num_geometries() const
{
    return geometries.size();
}

const SphereLight* Scene::get_lights() const
{
    return point_lights.empty() ? NULL : &point_lights[0];
}

size_t Scene::num_lights() const
{
    return point_lights.size();
}

Material* const* Scene::get_materials() const
{
    return materials.empty() ? NULL : &materials[0];
}

size_t Scene::num_materials() const
{
    return materials.size();
}

Mesh* const* Scene::get_meshes() const
{
    return meshes.empty() ? NULL : &meshes[0];
}

size_t Scene::num_meshes() const
{
    return meshes.size();
}

Physics* Scene::get_physics()
{
	return &phys;
}

Animator* const* Scene::get_animators() const
{
	return animators.empty() ? NULL : &animators[0];
}

size_t Scene::num_animators() const
{
	return animators.size();
}

void Scene::reset()
{
    for ( GeometryList::iterator i = geometries.begin(); i != geometries.end(); ++i ) {
        delete *i;
    }
    for ( MaterialList::iterator i = materials.begin(); i != materials.end(); ++i ) {
        delete *i;
    }
    for ( MeshList::iterator i = meshes.begin(); i != meshes.end(); ++i ) {
        delete *i;
    }

    geometries.clear();
    materials.clear();
    meshes.clear();
    point_lights.clear();

    camera = Camera();

	phys = Physics();
	pool_controller = PoolController();

    background_color = Color3::Black();
    ambient_light = Color3::Black();
    refractive_index = 1.0;
}

void Scene::add_geometry( Geometry* g )
{
    geometries.push_back( g );
}

void Scene::add_material( Material* m )
{
    materials.push_back( m );
}

void Scene::add_mesh( Mesh* m )
{
    meshes.push_back( m );
}

void Scene::add_light( const SphereLight& l )
{
    point_lights.push_back( l );
}

void Scene::add_animator( Animator *animator)
{
	animators.push_back(animator);
}

void Scene::update_animation(real_t time)
{
	Animator * const *animators = get_animators();
	for (unsigned int i = 0; i < num_animators(); i++)
		animators[i]->update(time);
}

void Scene::update(real_t dt)
{
	phys.step(dt);
	pool_controller.update(dt, is_pool);
}

void Scene::handle_event(const SDL_Event& event)
{
	pool_controller.handle_event(event, is_pool);
}


} /* _462 */

