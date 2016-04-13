#include "p3/physics.hpp"

namespace _462 {

Physics::Physics()
{
    reset();
}

Physics::~Physics()
{
    reset();
}

void Physics::step( real_t dt )
{
    // TODO step the world forward by dt. Need to detect collisions, apply
    // forces, and integrate positions and orientations.
    //
    // Note: put RK4 here, not in any of the physics bodies
    //
    // Must use the functions that you implemented
    //
    // Note, when you change the position/orientation of a physics object,
    // change the position/orientation of the graphical object that represents
    // it

	// Check Collision
	check_collision();

	// RK4
	integrate(dt);

	// Update Step & Orientation
	update_geometries(dt);
}

void Physics::apply_forces(SphereBody* body, const State &initial) {
	// clear force
	body->clear_force();
	
	// apply forces -> assign force and torque
	// gravity
	body->apply_force(gravity, Vector3::Zero());
	// spring
	for (size_t i = 0; i < num_springs(); ++i) {
		if (springs[i]->body1->id == body->id) {
			springs[i]->step(initial);
		}
	}
}

Derivative Physics::evaluate(SphereBody* body, const State &initial) {
	Derivative output;
	
	// inertia
	real_t I = 0.4f * body->mass * body->radius * body->radius;
	apply_forces(body, initial);

	output.d_pos = initial.vel;
	output.d_vel = body->force / body->mass;
	output.d_rot = initial.rot;
	output.d_ang_vel = body->torque / I;

	return output;
}

Derivative Physics::evaluate(SphereBody* body, const State &initial, float dt, const Derivative &d) {
	State state;
	state.pos = initial.pos + d.d_pos * dt;
	state.vel = initial.vel + d.d_vel * dt;
	state.rot = add_quaternion(initial.rot, d.d_ang_vel * dt * dt); // state.rot = initial.rot + d.d_rot * dt;
	state.ang_vel = initial.ang_vel + d.d_ang_vel * dt;

	// inertia
	real_t I = 0.4f * body->mass * body->radius * body->radius;
	apply_forces(body, state);

	Derivative output;
	output.d_pos = state.vel;
	output.d_vel = body->force / body->mass;
	output.d_rot = state.rot;
	output.d_ang_vel = body->torque / I;

	return output;
}

void Physics::integrate(real_t dt) {
	// spheres
	for (size_t i = 0; i < num_spheres(); ++i) {
		State state;
		state.pos = spheres[i]->position;
		state.vel = spheres[i]->velocity;
		state.rot = spheres[i]->orientation;
		state.ang_vel = spheres[i]->angular_velocity;

		Derivative k1 = evaluate(spheres[i], state);
		Derivative k2 = evaluate(spheres[i], state, dt * 0.5f, k1);
		Derivative k3 = evaluate(spheres[i], state, dt * 0.5f, k2);
		Derivative k4 = evaluate(spheres[i], state, dt, k3);

		const Vector3 d_pos_dt = 1.0f / 6.0f * (k1.d_pos + 2.0f*(k2.d_pos + k3.d_pos) + k4.d_pos);
		const Vector3 d_vel_dt = 1.0f / 6.0f * (k1.d_vel + 2.0f*(k2.d_vel + k3.d_vel) + k4.d_vel);
		const Vector3 d_ang_vel_dt = 1.0f / 6.0f * (k1.d_ang_vel + 2.0f*(k2.d_ang_vel + k3.d_ang_vel) + k4.d_ang_vel);

		// position
		spheres[i]->position += d_pos_dt * dt;
		spheres[i]->velocity += d_vel_dt * dt;
		// orientation
		spheres[i]->angular_velocity += d_ang_vel_dt * dt;
		spheres[i]->orientation = add_quaternion(spheres[i]->orientation, spheres[i]->angular_velocity * dt);
	}
}

Quaternion Physics::add_quaternion(Quaternion first_orientation, Vector3 delta_orientation) {
	Quaternion result = first_orientation;

	// pitch, yaw, roll
	result = normalize(Quaternion(result * Vector3::UnitX(), delta_orientation.x) * result);
	result = normalize(Quaternion(result * Vector3::UnitY(), delta_orientation.y) * result);
	result = normalize(Quaternion(result * Vector3::UnitZ(), delta_orientation.z) * result);

	return result;
}

void Physics::update_geometries(real_t dt) {
	// spheres
	for (size_t i = 0; i < num_spheres(); ++i) {
		spheres[i]->step_position(dt, collision_damping);
		spheres[i]->step_orientation(dt, collision_damping);
	}
}

void Physics::check_collision() {
	for (size_t i = 0; i < num_spheres(); ++i) {
		// check order: sphere, triangle, model, plane
		// spheres
		for (size_t j = i + 1; j < num_spheres(); ++j) {
			collides(*spheres[i], *spheres[j], collision_damping);
		}
		// triangles
		for (size_t j = 0; j < num_triangles(); ++j) {
			collides(*spheres[i], *triangles[j], collision_damping);
		}
		// models
		for (size_t j = 0; j < num_models(); ++j) {
			collides(*spheres[i], *models[j], collision_damping);
		}
		// planes
		for (size_t j = 0; j < num_planes(); ++j) {
			collides(*spheres[i], *planes[j], collision_damping);
		}
	}
}

void Physics::add_sphere( SphereBody* b )
{
    spheres.push_back( b );
}

size_t Physics::num_spheres() const
{
    return spheres.size();
}

void Physics::add_plane( PlaneBody* p )
{
    planes.push_back( p );
}

size_t Physics::num_planes() const
{
    return planes.size();
}

void Physics::add_triangle( TriangleBody* t )
{
    triangles.push_back( t );
}

size_t Physics::num_triangles() const
{
    return triangles.size();
}

void Physics::add_model( ModelBody* m )
{
    models.push_back( m );
}

size_t Physics::num_models() const
{
    return models.size();
}

void Physics::add_spring( Spring* s )
{
    springs.push_back( s );
}

size_t Physics::num_springs() const
{
    return springs.size();
}

void Physics::reset()
{
    for ( SphereList::iterator i = spheres.begin(); i != spheres.end(); i++ ) {
        delete *i;
    }
    for ( PlaneList::iterator i = planes.begin(); i != planes.end(); i++ ) {
        delete *i;
    }
    for ( TriangleList::iterator i = triangles.begin(); i != triangles.end(); i++ ) {
        delete *i;
    }
    for ( ModelList::iterator i = models.begin(); i != models.end(); i++) {
        delete *i;
    }
    for ( SpringList::iterator i = springs.begin(); i != springs.end(); i++ ) {
        delete *i;
    }

    spheres.clear();
    planes.clear();
    triangles.clear();
    springs.clear();
    models.clear();

    gravity = Vector3::Zero();
	collision_damping = 0.0;
}

}
