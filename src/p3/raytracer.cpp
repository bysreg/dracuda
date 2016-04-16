/**
 * @file raytacer.cpp
 * @brief Raytracer class
 *
 * Implement these functions for project 4.
 *
 * @author H. Q. Bovik (hqbovik)
 * @bug Unimplemented
 */

#include <typeinfo>
#include "raytracer.hpp"
#include "scene/scene.hpp"
#include "math/quickselect.hpp"
#include "p3/randomgeo.hpp"
#include "scene/utils.hpp"
#include <SDL_timer.h>

#include <scene/sphere.hpp>
#include <scene/model.hpp>
#include <scene/triangle.hpp>

using namespace std;
namespace _462 {

//number of rows to render before updating the result
static const unsigned STEP_SIZE = 1;
static const unsigned CHUNK_SIZE = 1;

Raytracer::Raytracer() {
        scene = 0;
        width = 0;
        height = 0;
    }

Raytracer::~Raytracer() { }

/**
 * Initializes the raytracer for the given scene. Overrides any previous
 * initializations. May be invoked before a previous raytrace completes.
 * @param scene The scene to raytrace.
 * @param width The width of the image being raytraced.
 * @param height The height of the image being raytraced.
 * @return true on success, false on error. The raytrace will abort if
 *  false is returned.
 */
bool Raytracer::initialize(Scene* scene, size_t num_samples,
               size_t width, size_t height)
{
    this->scene = scene;
    this->num_samples = num_samples;
    this->width = width;
    this->height = height;

    current_row = 0;

    projector.init(scene->camera);



    scene->initialize();
    photonMap.initialize(scene);
    return true;
}

const unsigned int max_depth = 4;

bool Raytracer::trace_shadow_ray(Ray &ray, real_t max_time)
{
	Ray transformed_ray;
	Geometry * const* geometries = scene->get_geometries();
	Intersection current_intersection;
	Vector4 ray_d = Vector4(ray.d.x, ray.d.y, ray.d.z, 0);
	Vector4 ray_e = Vector4(ray.e.x, ray.e.y, ray.e.z, 1);
	for (unsigned int i = 0; i < scene->num_geometries(); i++) {
		Geometry *g = geometries[i];
		transformed_ray.d = (g->invMat * ray_d).xyz();
		transformed_ray.e = (g->invMat * ray_e).xyz();
		real_t time = g->intersect_ray(transformed_ray, current_intersection);
		if ((time >= EPS) && (time < max_time))
			return true;
	}
	return false;
}

bool refract (Vector3 d, Vector3 normal, real_t ratio, Vector3 *t)
{
	real_t dn = dot(d, normal);
	real_t delta = 1 - ratio * ratio * (1 - dn * dn);
	if (delta < 0)
		return false;
	*t = normalize(ratio * (d - normal * dn) - normal * sqrt(delta));
	return true;
}

Color3 Raytracer::trace_ray(Ray &ray, unsigned int depth, bool inside_geometry){
	if (depth > max_depth)
		return Color3::Black();

	Geometry * const* geometries = scene->get_geometries();
	
	real_t min_time = -1;
	bool intersected = false;
	Geometry * hit_geometry;
	Intersection current_intersection;
	Intersection hit_intersection;
	Ray transformed_ray;
	Vector4 ray_d = Vector4(ray.d.x, ray.d.y, ray.d.z, 0);
	Vector4 ray_e = Vector4(ray.e.x, ray.e.y, ray.e.z, 1);
	for (unsigned int i = 0; i < scene->num_geometries(); i++) {
		Geometry *g = geometries[i];
		transformed_ray.d = (g->invMat * ray_d).xyz();
		transformed_ray.e = (g->invMat * ray_e).xyz();

		real_t time = g->intersect_ray(transformed_ray, current_intersection);
		if (time > EPS) {
			if (!intersected || (time < min_time)) {
				min_time = time;
				hit_geometry = g;
				hit_intersection = current_intersection;
				intersected = true;
			}
		}
	}
	if (intersected) {
		if (scene->bump_map_enabled)
			hit_geometry->bump_normal(hit_intersection);
		Vector3 hit_point = ray.e + min_time * ray.d;
		InterpolatedMaterial i_material;
		hit_geometry->interpolate_material(hit_intersection, i_material);
		real_t r_index = i_material.refractive_index;

		Vector3 normal = hit_intersection.normal;
		// Ambient
		Color3 color = scene->ambient_light * i_material.ambient;
		// Diffuse & (Specular highlight)
		for (unsigned int i = 0; i < scene->num_lights(); i++) {
			const SphereLight *light = scene->get_lights() + i;
			Vector3 incident = normalize(light->position - hit_point);
			real_t dot_product = dot (incident, normal);
			if (dot_product < 0)
				continue;
			Vector3 light_sample = light->sample();
			Ray light_ray = Ray(hit_point, normalize(light_sample - hit_point));
			real_t max = length(light_sample - hit_point);
			if (!trace_shadow_ray(light_ray, max)) {
				color += i_material.diffuse * light->color * light->get_intensity(max) * dot_product;
				if (scene->specular_highlight_enabled) {
					Vector3 halfway = normalize(incident - ray.d);
					real_t halfway_dot_product = dot(halfway, normal);
					if (halfway_dot_product > 0)
						color += i_material.specular * light->color * light->get_intensity(max)
							* pow(halfway_dot_product, i_material.shininess);
				}
			}
		}

		if (scene->glossy_enabled) {
			double alpha = i_material.shininess;
			double random = random_uniform();
			double theta = acos(pow(random, 1 / (alpha + 1)));
			double phi = random_uniform() * 2 * PI;
			double theta2 = acos(ray.d.z);
			Vector3 tangent = Vector3(cos(theta2) , sin(theta2) * sin(phi) , sin(theta2) * cos(phi));
			Vector3 bend = ray.d * cos(theta) + tangent * sin(theta);
			ray = Ray (hit_point, bend);
		}

		real_t specular_dot_product = dot(-ray.d, normal);
			
		bool has_specular = (!inside_geometry && specular_dot_product > 0)
			|| (inside_geometry && specular_dot_product < 0);
		if (has_specular) {
			Vector3 reflected = 2 * specular_dot_product * normal + ray.d;
			reflected = normalize(reflected);
			Ray reflected_ray = Ray(hit_point, reflected);
			if (r_index < EPS) {
				// Opaque
				color += i_material.specular * trace_ray(reflected_ray, depth + 1, false);
			} else {
				// Transparent
				real_t ratio = scene->refractive_index / r_index;
				Vector3 d = ray.d;
				Vector3 t;
				real_t c, nt;
				bool total = false;
				if (!inside_geometry) {
					// From air to geo
					refract(d, normal, ratio, &t);
					c = specular_dot_product;
					nt = r_index;
				} else {
					// From geo to air
					if (refract(d, -normal, 1 / ratio, &t))
						c = dot (t, normal);
					else
						total = true;
					nt = scene->refractive_index;
				}
				if (total) {
					color += trace_ray(reflected_ray, depth + 1, inside_geometry);
				} else {
					Ray refracted_ray = Ray(hit_point, t);
					real_t r0 = (nt - 1) / (nt + 1);
					r0 *= r0;
					real_t r = r0 + (1 - r0) * pow(1 - c, 5);
					if (r < random_uniform())
						color += trace_ray(refracted_ray, depth + 1, !inside_geometry);
					else
						color += trace_ray(reflected_ray, depth + 1, inside_geometry);
				}
			}
		}
		color *= i_material.tex_color;
		return color;
	} else {
		Color3 backgroundColor;
		if (scene->envmap.intersect(ray, backgroundColor))
			return backgroundColor;
		else
			return scene->background_color;
	}

}

/**
 * Performs a raytrace on the given pixel on the current scene.
 * The pixel is relative to the bottom-left corner of the image.
 * @param scene The scene to trace.
 * @param x The x-coordinate of the pixel to trace.
 * @param y The y-coordinate of the pixel to trace.
 * @param width The width of the screen in pixels.
 * @param height The height of the screen in pixels.
 * @return The color of that pixel in the final image.
 */
Color3 Raytracer::trace_pixel(size_t x,
                  size_t y,
                  size_t width,
                  size_t height)
{
    assert(x < width);
    assert(y < height);

    real_t dx = real_t(1)/width;
    real_t dy = real_t(1)/height;

    Color3 res = Color3::Black();
	Vector3 camera_dir = normalize(scene->camera.get_direction());
	double focus = scene->focus;
	double focus_n2_product = focus * dot(camera_dir, camera_dir);
	Vector3 camera_pos = scene->camera.get_position();
	int sample = scene->dof_samples;
	Vector3 dof_y = scene->camera.get_up();
	Vector3 dof_x = normalize(cross(dof_y, camera_dir));
	if (scene->depth_of_field_enabled) {
		for (unsigned int iter_x = 0; iter_x < num_samples; iter_x++)
			for (unsigned int iter_y = 0; iter_y < num_samples; iter_y++) {
				real_t i = real_t(2)*(real_t(x) + (real_t)iter_x / num_samples + random_uniform() / num_samples)*dx - real_t(1);
				real_t j = real_t(2)*(real_t(y) + (real_t)iter_y / num_samples + random_uniform() / num_samples)*dy - real_t(1);
				Vector3 d = projector.get_pixel_dir(i, j);
				Ray r = Ray(camera_pos, d);
				Color3 temp = Color3::Black();
				temp += trace_ray(r, 0, false);
				double t = focus_n2_product / (dot(d, camera_dir));
				Vector3 p = camera_pos + d * t;
				Ray r0;
				for (int k = 0; k < sample; k++) {
					for (int l = 0; l < sample; l++) {
						real_t ax = real_t(2) * (real_t(k) + random_uniform()) / sample - real_t(1);
						real_t ay = real_t(2) * (real_t(l) + random_uniform()) / sample - real_t(1);
						Vector3 e0 = camera_pos + scene->aperture * (dof_x * ax + dof_y * ay);
						r0 = Ray(e0, normalize(p - e0));
						temp += trace_ray(r0, 0, false);
					}
				}
				temp *= 1.0 / (sample * sample + 1);
				res += temp;
			}
	} else {
		for (unsigned int iter_x = 0; iter_x < num_samples; iter_x++)
			for (unsigned int iter_y = 0; iter_y < num_samples; iter_y++) {
				real_t i = real_t(2)*(real_t(x) + (real_t)iter_x / num_samples + random_uniform() / num_samples)*dx - real_t(1);
				real_t j = real_t(2)*(real_t(y) + (real_t)iter_y / num_samples + random_uniform() / num_samples)*dy - real_t(1);
				Vector3 d = projector.get_pixel_dir(i, j);
				Ray r = Ray(camera_pos, d);
				res += trace_ray(r, 0, false);
			}
	}
    return res*(real_t(1)/ num_samples / num_samples);
}

/**
 * Raytraces some portion of the scene. Should raytrace for about
 * max_time duration and then return, even if the raytrace is not copmlete.
 * The results should be placed in the given buffer.
 * @param buffer The buffer into which to place the color data. It is
 *  32-bit RGBA (4 bytes per pixel), in row-major order.
 * @param max_time, If non-null, the maximum suggested time this
 *  function raytrace before returning, in seconds. If null, the raytrace
 *  should run to completion.
 * @return true if the raytrace is complete, false if there is more
 *  work to be done.
 */
bool Raytracer::raytrace(unsigned char* buffer, real_t* max_time)
{
    
    static const size_t PRINT_INTERVAL = 64;

    // the time in milliseconds that we should stop
    unsigned int end_time = 0;
    bool is_done;

    if (max_time)
    {
        // convert duration to milliseconds
        unsigned int duration = (unsigned int) (*max_time * 1000);
        end_time = SDL_GetTicks() + duration;
    }

    // until time is up, run the raytrace. we render an entire group of
    // rows at once for simplicity and efficiency.
    for (; !max_time || end_time > SDL_GetTicks(); current_row += STEP_SIZE)
    {
        // we're done if we finish the last row
        is_done = current_row >= height;
        // break if we finish
        if (is_done) break;

        int loop_upper = std::min(current_row + STEP_SIZE, height);

        for (int c_row = current_row; c_row < loop_upper; c_row++)
        {
            /*
             * This defines a critical region of code that should be
             * executed sequentially.
             */
#pragma omp critical
            {
                if (c_row % PRINT_INTERVAL == 0)
                    printf("Raytracing (Row %d)\n", c_row);
            }
            
        // This tells OpenMP that this loop can be parallelized.
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
#ifdef WIN32
			for (long x = 0; x < width; x++)
#else
            for (size_t x = 0; x < width; x++)
#endif
            {
                // trace a pixel
                Color3 color = trace_pixel(x, c_row, width, height);
                // write the result to the buffer, always use 1.0 as the alpha
                color.to_array4(&buffer[4 * (c_row * width + x)]);
            }
#pragma omp barrier

        }
    }

    if (is_done) printf("Done raytracing!\n");

    return is_done;
}

} /* _462 */
