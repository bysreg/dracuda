/**
 * @file sphere.cpp
 * @brief Function defnitions for the Sphere class.
 *
 * @author Kristin Siu (kasiu)
 * @author Eric Butler (edbutler)
 */

#include "scene/sphere.hpp"
#include "scene/utils.hpp"
#include "application/opengl.hpp"
#include <algorithm>

namespace _462 {

#define SPHERE_NUM_LAT 80
#define SPHERE_NUM_LON 100

#define SPHERE_NUM_VERTICES ( ( SPHERE_NUM_LAT + 1 ) * ( SPHERE_NUM_LON + 1 ) )
#define SPHERE_NUM_INDICES ( 6 * SPHERE_NUM_LAT * SPHERE_NUM_LON )
// index of the x,y sphere where x is lat and y is lon
#define SINDEX(x,y) ((x) * (SPHERE_NUM_LON + 1) + (y))
#define VERTEX_SIZE 8
#define TCOORD_OFFSET 0
#define NORMAL_OFFSET 2
#define VERTEX_OFFSET 5
#define BUMP_FACTOR 1

static unsigned int Indices[SPHERE_NUM_INDICES];
static float Vertices[VERTEX_SIZE * SPHERE_NUM_VERTICES];

static void init_sphere()
{
    static bool initialized = false;
    if ( initialized )
        return;

    for ( int i = 0; i <= SPHERE_NUM_LAT; i++ ) {
        for ( int j = 0; j <= SPHERE_NUM_LON; j++ ) {
            real_t lat = real_t( i ) / SPHERE_NUM_LAT;
            real_t lon = real_t( j ) / SPHERE_NUM_LON;
            float* vptr = &Vertices[VERTEX_SIZE * SINDEX(i,j)];

            vptr[TCOORD_OFFSET + 0] = lon;
            vptr[TCOORD_OFFSET + 1] = 1-lat;

            lat *= PI;
            lon *= 2 * PI;
            real_t sinlat = sin( lat );

            vptr[NORMAL_OFFSET + 0] = vptr[VERTEX_OFFSET + 0] = sinlat * sin( lon );
            vptr[NORMAL_OFFSET + 1] = vptr[VERTEX_OFFSET + 1] = cos( lat ),
            vptr[NORMAL_OFFSET + 2] = vptr[VERTEX_OFFSET + 2] = sinlat * cos( lon );
        }
    }

    for ( int i = 0; i < SPHERE_NUM_LAT; i++ ) {
        for ( int j = 0; j < SPHERE_NUM_LON; j++ ) {
            unsigned int* iptr = &Indices[6 * ( SPHERE_NUM_LON * i + j )];

            unsigned int i00 = SINDEX(i,  j  );
            unsigned int i10 = SINDEX(i+1,j  );
            unsigned int i11 = SINDEX(i+1,j+1);
            unsigned int i01 = SINDEX(i,  j+1);

            iptr[0] = i00;
            iptr[1] = i10;
            iptr[2] = i11;
            iptr[3] = i11;
            iptr[4] = i01;
            iptr[5] = i00;
        }
    }

    initialized = true;
}

Sphere::Sphere()
    : radius(0), material(0) {}

Sphere::~Sphere() {}

real_t Sphere::intersect_ray(Ray &ray, Intersection &intersection) const
{
	real_t A = dot(ray.d, ray.d);
	real_t B = dot(2 * ray.d, ray.e);
	real_t C = dot(ray.e, ray.e) - radius * radius;
	real_t time = solve_time(A, B, C);
	if (time > EPS) {
		Vector3 hit_point = ray.e + ray.d * time;
		Vector3 normal = normalize(hit_point);
		intersection.normal = normalize(normMat * normal);
		intersection.tex_coord = Vector2(atan2(normal.x, normal.z) / (2 * PI), asin(normal.y) / PI + 0.5);
	}
	return time;
}

bool Sphere::post_initialize()
{
	return true;
}

void Sphere::bump_normal(Intersection &intersection) const
{
	if (material->bump.data) {
		Vector3 normal = intersection.normal;
		Vector3 tangent = normalize(Vector3(-1 / normal.x, 0, 1 / normal.z));
		Vector3 bitangent = normalize(Vector3(-normal.x, 1 / normal.y - normal.y, -normal.z));
		intersection.normal = normalize(normal - tangent * material->bump.sample_bump_u(intersection.tex_coord) - bitangent * material->bump.sample_bump_v(intersection.tex_coord));
	}
}

void Sphere::intersect_ray_interval(Ray &ray, CSGIntervals &interval) const
{
	real_t A = dot(ray.d, ray.d);
	real_t B = dot(2 * ray.d, ray.e);
	real_t C = dot(ray.e, ray.e) - radius * radius;
	real_t x1, x2;
	if (solve_time2(A, B, C, x1, x2))
	{
		Interval i;
		i.min = x1;
		i.max = x2;
		interval.add(i);
	}
}

void Sphere::interpolate_material (Intersection &intersection, InterpolatedMaterial &i_material) const {
	i_material.ambient = material->ambient;
	i_material.diffuse = material->diffuse;
	i_material.specular = material->specular;
	i_material.tex_color = material->texture.sample(intersection.tex_coord);
	i_material.shininess = material->shininess;
	i_material.refractive_index = material->refractive_index;
}

void Sphere::render() const
{
    // create geometry if we haven't already
    init_sphere();

    if ( material )
        material->set_gl_state();

    // just scale by radius and draw unit sphere
    glPushMatrix();
    glScaled( radius, radius, radius );
    glInterleavedArrays( GL_T2F_N3F_V3F, VERTEX_SIZE * sizeof Vertices[0], Vertices );
    glDrawElements( GL_TRIANGLES, SPHERE_NUM_INDICES, GL_UNSIGNED_INT, Indices );
    glPopMatrix();

    if ( material )
        material->reset_gl_state();
}

} /* _462 */

