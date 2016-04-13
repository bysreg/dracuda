/**
 * @file triangle.cpp
 * @brief Function definitions for the Triangle class.
 *
 * @author Eric Butler (edbutler)
 */

#include "scene/triangle.hpp"
#include "application/opengl.hpp"
#include "math/math.hpp"
#include "scene/utils.hpp"

namespace _462 {

Triangle::Triangle()
{
    vertices[0].material = 0;
    vertices[1].material = 0;
    vertices[2].material = 0;
    isBig=true;
}

Triangle::~Triangle() { }

real_t Triangle::intersect_ray(Ray &ray, Intersection &intersection) const
{
	real_t beta, gamma, time;
	bool ret = TriangleIntersection(ray, vertices[0].position, vertices[1].position, vertices[2].position, beta, gamma, time);
	if (ret) {
		intersection.tex_coord = vertices[0].tex_coord * (1 - beta - gamma) + vertices[1].tex_coord * beta + vertices[2].tex_coord * gamma;
		intersection.normal = normalize(cross(vertices[1].position - vertices[0].position, vertices[2].position - vertices[0].position));
		intersection.normal = normalize(normMat * intersection.normal);
		intersection.beta = beta;
		intersection.gamma = gamma;
		intersection.alpha = 1 - beta - gamma;
		return time;
	} else {
		return -1;
	}
}

bool Triangle::post_initialize()
{
	// create bounding box
	boundBox = createBoundingBox();

	return true;
}

void Triangle::intersect_ray_interval(Ray &ray, CSGIntervals &interval) const
{
}

void Triangle::bump_normal(Intersection &intersection) const
{
	if (vertices[0].material->bump.data) {
		Vector2 t10 = vertices[1].tex_coord - vertices[0].tex_coord;
		Vector2 t20 = vertices[2].tex_coord - vertices[0].tex_coord;
		Vector3 a1 = vertices[1].position - vertices[0].position;
		Vector3 a2 = vertices[2].position - vertices[0].position;
		Vector3 T, B;
		for (int i = 0; i < 3; i++) {
			T[i] = (a1[i] * t20.y - a2[i] * t10.y) / (t10.x * t20.y - t20.x * t10.y);
			B[i] = (a1[i] * t20.x - a2[i] * t10.x) / (t10.y * t20.x - t20.y * t10.x);
		}
		real_t bump_u = intersection.alpha * vertices[0].material->bump.sample_bump_u(intersection.tex_coord)
			+ intersection.beta * vertices[1].material->bump.sample_bump_u(intersection.tex_coord)
			+ intersection.gamma * vertices[2].material->bump.sample_bump_u(intersection.tex_coord);
		real_t bump_v = intersection.alpha * vertices[0].material->bump.sample_bump_v(intersection.tex_coord)
			+ intersection.beta * vertices[1].material->bump.sample_bump_v(intersection.tex_coord)
			+ intersection.gamma * vertices[2].material->bump.sample_bump_v(intersection.tex_coord);
		intersection.normal = normalize(intersection.normal - T * bump_u - B * bump_v);
	}

}

void Triangle::interpolate_material (Intersection &intersection, InterpolatedMaterial &i_material) const {
	i_material.ambient = vertices[0].material->ambient * intersection.alpha
		+ vertices[1].material->ambient * intersection.beta
		+ vertices[2].material->ambient * intersection.gamma;
	i_material.diffuse = vertices[0].material->diffuse * intersection.alpha
		+ vertices[1].material->diffuse * intersection.beta
		+ vertices[2].material->diffuse * intersection.gamma;
	i_material.specular = vertices[0].material->specular * intersection.alpha
		+ vertices[1].material->specular * intersection.beta
		+ vertices[2].material->specular * intersection.gamma;
	i_material.tex_color = vertices[0].material->texture.sample(intersection.tex_coord) * intersection.alpha
		+ vertices[1].material->texture.sample(intersection.tex_coord) * intersection.beta
		+ vertices[2].material->texture.sample(intersection.tex_coord) * intersection.gamma;
	i_material.refractive_index = vertices[0].material->refractive_index * intersection.alpha
		+ vertices[1].material->refractive_index * intersection.beta
		+ vertices[2].material->refractive_index * intersection.gamma;
	i_material.shininess = vertices[0].material->shininess * intersection.alpha
		+ vertices[1].material->shininess * intersection.beta
		+ vertices[2].material->shininess * intersection.gamma;
}
void Triangle::render() const
{
    bool materials_nonnull = true;
    for ( int i = 0; i < 3; ++i )
        materials_nonnull = materials_nonnull && vertices[i].material;

    // this doesn't interpolate materials. Ah well.
    if ( materials_nonnull )
        vertices[0].material->set_gl_state();

    glBegin(GL_TRIANGLES);

#if REAL_FLOAT
    glNormal3fv( &vertices[0].normal.x );
    glTexCoord2fv( &vertices[0].tex_coord.x );
    glVertex3fv( &vertices[0].position.x );

    glNormal3fv( &vertices[1].normal.x );
    glTexCoord2fv( &vertices[1].tex_coord.x );
    glVertex3fv( &vertices[1].position.x);

    glNormal3fv( &vertices[2].normal.x );
    glTexCoord2fv( &vertices[2].tex_coord.x );
    glVertex3fv( &vertices[2].position.x);
#else
    glNormal3dv( &vertices[0].normal.x );
    glTexCoord2dv( &vertices[0].tex_coord.x );
    glVertex3dv( &vertices[0].position.x );

    glNormal3dv( &vertices[1].normal.x );
    glTexCoord2dv( &vertices[1].tex_coord.x );
    glVertex3dv( &vertices[1].position.x);

    glNormal3dv( &vertices[2].normal.x );
    glTexCoord2dv( &vertices[2].tex_coord.x );
    glVertex3dv( &vertices[2].position.x);
#endif

    glEnd();

    if ( materials_nonnull )
        vertices[0].material->reset_gl_state();
}

Bound Triangle::createBoundingBox() {
	Vector3 min = vertices[0].position;
	Vector3 max = vertices[0].position;

	for (int i = 1; i < 3; ++i) {
		if (vertices[i].position.x < min.x) min.x = vertices[i].position.x;
		if (vertices[i].position.y < min.y) min.y = vertices[i].position.y;
		if (vertices[i].position.z < min.z) min.z = vertices[i].position.z;
		if (vertices[i].position.x > max.x) max.x = vertices[i].position.x;
		if (vertices[i].position.y > max.y) max.y = vertices[i].position.y;
		if (vertices[i].position.z > max.z) max.z = vertices[i].position.z;
	}

	return Bound(min, max);
}

} /* _462 */
