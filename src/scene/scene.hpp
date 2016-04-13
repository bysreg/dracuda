/**
 * @file scene.hpp
 * @brief Class definitions for scenes.
 *
 */

#ifndef _462_SCENE_SCENE_HPP_
#define _462_SCENE_SCENE_HPP_

#include "math/vector.hpp"
#include "math/matrix.hpp"
#include "math/camera.hpp"
#include "scene/material.hpp"
#include "scene/mesh.hpp"
#include "scene/envmap.hpp"
#include "scene/animator.hpp"
#include "scene/ray.hpp"
#include <string>
#include <vector>
#include <cfloat>
#include "scene/bound.hpp"
#include "scene/csgintervals.hpp"
#include "scene/pool_controller.hpp"

#include "p3/physics.hpp"

namespace _462 {



struct SphereLight
{
    struct Attenuation
    {
        real_t constant;
        real_t linear;
        real_t quadratic;
    };

    SphereLight();

    bool intersect(const Ray& r, real_t& t);
	real_t get_intensity(real_t time) const;

    // The position of the light, relative to world origin.
    Vector3 position;
    // The color of the light (both diffuse and specular)
    Color3 color;
    // attenuation
    Attenuation attenuation;
    real_t radius;
	Vector3 sample() const;
};

/**
 * The container class for information used to render a scene composed of
 * Geometries.
 */
class Scene
{
public:

    /// the camera
    Camera camera;
    /// the background color
    Color3 background_color;
    /// the amibient light of the scene
    Color3 ambient_light;
    /// the refraction index of air
    real_t refractive_index;

	/// pool controller
	PoolController pool_controller;

	EnvMap envmap;
	
	// Depth of field options
	real_t dof_samples;
	real_t focus;
	real_t aperture;

	// Switches
	int depth_of_field_enabled = 0;
	int glossy_enabled = 0;
	int specular_highlight_enabled = 0;
	int bump_map_enabled = 1;
	
    /// Creates a new empty scene.
    Scene();

    /// Destroys this scene. Invokes delete on everything in geometries.
    ~Scene();
	bool animated = false;
	real_t animation_duration = 0;
	real_t animation_fps;

    bool initialize();
	bool post_initialize();

    // accessor functions
    Geometry* const* get_geometries() const;
    size_t num_geometries() const;
    const SphereLight* get_lights() const;
    size_t num_lights() const;
    Material* const* get_materials() const;
    size_t num_materials() const;
    Mesh* const* get_meshes() const;
    size_t num_meshes() const;

	Physics* get_physics();

	Animator * const * get_animators() const;
	size_t num_animators() const;

	void update_animation(real_t time);
    /// Clears the scene, and invokes delete on everything in geometries.
    void reset();

    // functions to add things to the scene
    // all pointers are deleted by the scene upon scene deconstruction.
    void add_geometry( Geometry* g );
    void add_material( Material* m );
    void add_mesh( Mesh* m );
    void add_light( const SphereLight& l );
	void add_animator(Animator *animator);
    
	void update(real_t dt);
	void handle_event(const SDL_Event& event);
	// the physics engine
	Physics phys;

	bool is_pool;
private:

    typedef std::vector< SphereLight > SphereLightList;
    typedef std::vector< Material* > MaterialList;
    typedef std::vector< Mesh* > MeshList;
    typedef std::vector< Geometry* > GeometryList;
	typedef std::vector< Animator* > AnimatorList;

    // list of all lights in the scene
    SphereLightList point_lights;
    // all materials used by geometries
    MaterialList materials;
    // all meshes used by models
    MeshList meshes;
    // list of all geometries. deleted in dctor, so should be allocated on heap.
    GeometryList geometries;

	AnimatorList animators;

    // no meaningful assignment or copy
    Scene(const Scene&);
    Scene& operator=(const Scene&);

};
} /* _462 */

#endif /* _462_SCENE_SCENE_HPP_ */
