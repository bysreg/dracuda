#include "collisions.hpp"
#include "scene/meshtree.hpp"

namespace _462 {

bool collides(SphereBody& body1, SphereBody& body2, real_t collision_damping)
{
    // TODO detect collision. If there is one, update velocity

	// relative velocity of one object to the other is positive
	if (relative_velocity(body1, body2.velocity, body2.position) >= 0) {
		real_t distance = length(body2.position - body1.position);

		// collide
		if (distance <= body1.radius + body2.radius) {
			// calculate new velocity
			Vector3 direction = (body2.position - body1.position) / distance;

			Vector3 v1_new, v2_new;

			v1_new = body1.velocity - body2.velocity;
			v2_new = body2.velocity + (2 * direction * (body1.mass / (body1.mass + body2.mass)) * dot(v1_new, direction));
			v1_new = ((body1.mass * body1.velocity) + (body2.mass * body2.velocity) - (body2.mass * v2_new)) / body1.mass;

			// update velocity
			body1.velocity = damping(v1_new, collision_damping);
			body2.velocity = damping(v2_new, collision_damping);

			return true;
		}
	}
    return false;
}

bool collides(SphereBody& body1, TriangleBody& body2, real_t collision_damping)
{
	// TODO detect collision. If there is one, update velocity

	// check bounding box
	if (body2.triangle->boundBox.collides(body1.position, body1.radius)) {
		// get the triangle vertices
		Vector3 v_a = body2.vertices[0];
		Vector3 v_b = body2.vertices[1];
		Vector3 v_c = body2.vertices[2];

		// calculate normal
		Vector3 normal = normalize(cross(v_b - v_a, v_c - v_a));

		real_t distance = dot(body1.position - body2.position, normal);
		Vector3 projection_point = body1.position - distance * normal;

		// relative velocity of one object to the other is positive
		if (relative_velocity(body1, body2.velocity, projection_point) >= 0) {
			// check distance and check whether the hit_point within barycentric coordinates
			if (abs(distance) <= body1.radius && (is_within_barycentric(projection_point, body2.vertices) || is_hit_triangle(body1, body2.vertices))) {
				Vector3 v_new = body1.velocity - (real_t(2) * dot(body1.velocity, normal) * normal);

				// update velocity
				body1.velocity = damping(v_new, collision_damping);

				return true;
			}
		}
	}

	return false;
}

bool collides(SphereBody& body1, PlaneBody& body2, real_t collision_damping)
{
    // TODO detect collision. If there is one, update velocity

	real_t distance = dot(body1.position - body2.position, body2.normal);
	Vector3 projection_point = body1.position - distance * body2.normal;

	// relative velocity of one object to the other is positive
	if (relative_velocity(body1, body2.velocity, projection_point) >= 0) {
		// collide
		if (abs(distance) <= body1.radius) {
			Vector3 v_new = body1.velocity - (2 * dot(body1.velocity, body2.normal) * body2.normal);
			
			// update velocity
			body1.velocity = damping(v_new, collision_damping);

			return true;
		}
		return false;
	}
}

bool collides(SphereBody& body1, ModelBody& body2, real_t collision_damping)
{
    // TODO detect collision. If there is one, update velocity

	// iterate through all the triangles in the mesh
	bool is_collide = false;

	// check collision with tree
	collides_node(body1, body2, collision_damping, body2.model->tree->root, is_collide);

	return is_collide;
}

void collides_node(SphereBody& body1, ModelBody& body2, real_t collision_damping, MeshTreeNode* node, bool &is_collide) {
	if (node == NULL || is_collide) {
		return;
	}

	// collide with bounding box
	if (node->bbox->collides(body1.position - body2.position, body1.radius)) {
		// still has child
		if (node->left != NULL || node->right != NULL) {
			collides_node(body1, body2, collision_damping, node->left, is_collide);
			collides_node(body1, body2, collision_damping, node->right, is_collide);
		}
		else { // we have reached leaf
			const Mesh* mesh = body2.model->mesh;

			// iterate all triangles in node
			for (size_t idxTri = 0; idxTri < node->triangles.size(); ++idxTri) {
				// get the triangle vertices
				const MeshVertex* vertices = mesh->get_vertices();

				Vector3 p_a = vertices[node->triangles[idxTri].vertices[0]].position;
				Vector3 p_b = vertices[node->triangles[idxTri].vertices[1]].position;
				Vector3 p_c = vertices[node->triangles[idxTri].vertices[2]].position;

				// multiply with transformation matrix
				Vector4 t_p_a = body2.model->mat * Vector4(p_a.x, p_a.y, p_a.z, 0);
				Vector4 t_p_b = body2.model->mat * Vector4(p_b.x, p_b.y, p_b.z, 0);
				Vector4 t_p_c = body2.model->mat * Vector4(p_c.x, p_c.y, p_c.z, 0);

				Vector3 v_a = Vector3(t_p_a.x, t_p_a.y, t_p_a.z);
				Vector3 v_b = Vector3(t_p_b.x, t_p_b.y, t_p_b.z);
				Vector3 v_c = Vector3(t_p_c.x, t_p_c.y, t_p_c.z);

				// calculate normal
				Vector3 normal_plane = normalize(cross(v_b - v_a, v_c - v_a));

				real_t distance = dot(body1.position - (body2.position + v_a), normal_plane);
				Vector3 projection_point = body1.position - distance * normal_plane;

				// relative velocity of one object to the other is positive
				if (relative_velocity(body1, body2.velocity, projection_point) >= 0) {
					Vector3 vertices_pos[3] { body2.position + v_a, body2.position + v_b, body2.position + v_c };

					// check distance and check whether the hit_point within barycentric coordinates
					if (abs(distance) <= body1.radius && (is_within_barycentric(projection_point, vertices_pos) || is_hit_triangle(body1, vertices_pos))) {
						Vector3 v_new = body1.velocity - (real_t(2) * dot(body1.velocity, normal_plane) * normal_plane);

						// update velocity
						body1.velocity = damping(v_new, collision_damping);

						is_collide = true;
					}
				}
			}
		}
	}
}

real_t relative_velocity(SphereBody& body1, Vector3 velocity, Vector3 position) {
	Vector3 r_v = body1.velocity - velocity;
	return dot(normalize(r_v), normalize(position - body1.position));
}

Vector3 damping(Vector3 velocity, real_t collision_damping) {
	// check epsilon, so that we don't keep on making minor changes
	if (length(velocity) < 1e-9)
		return Vector3::Zero();

	return velocity - collision_damping * velocity;
}

bool is_hit_triangle(SphereBody& body1, const Vector3 vertices[3]) {
	for (size_t i = 0; i < 3; ++i) {
		size_t nextI = i + 1;
		if (nextI >= 3) nextI = 0;

		if (minimum_distance(body1.position, vertices[i], vertices[nextI]) <= body1.radius) {
			return true;
		}
	}

	return false;
}

bool is_within_barycentric(const Vector3 point, const Vector3 vertices[3]) {
	// compute vector
	Vector3 v0 = vertices[2] - vertices[0]; // C - A
	Vector3 v1 = vertices[1] - vertices[0]; // B - A
	Vector3 v2 = point - vertices[0]; // P - A

	// compute dot products
	real_t dot00 = dot(v0, v0);
	real_t dot01 = dot(v0, v1);
	real_t dot02 = dot(v0, v2);
	real_t dot11 = dot(v1, v1);
	real_t dot12 = dot(v1, v2);

	// Compute barycentric coordinates
	real_t invDenom = real_t(1) / (dot00 * dot11 - dot01 * dot01);
	real_t beta = (dot00 * dot12 - dot01 * dot02) * invDenom;
	real_t gamma = (dot11 * dot02 - dot01 * dot12) * invDenom;

	// Check if point is in triangle
	return (beta >= 0) && (gamma >= 0) && (beta + gamma < 1);
}

real_t minimum_distance(Vector3 p, Vector3 v, Vector3 w) {
	// Return minimum distance between line segment vw and point p
	const float l2 = pow(distance(v, w), 2);  // i.e. |w-v|^2 -  avoid a sqrt
	if (l2 == 0.0) return distance(p, v);   // v == w case

	// Consider the line extending the segment, parameterized as v + t (w - v).
	// We find projection of point p onto the line. 
	// It falls where t = [(p-v) . (w-v)] / |w-v|^2
	const float t = dot(p - v, w - v) / l2;
	if (t < 0.0) return distance(p, v);       // Beyond the 'v' end of the segment
	else if (t > 1.0) return distance(p, w);  // Beyond the 'w' end of the segment
	const Vector3 projection = v + t * (w - v);  // Projection falls on the segment
	
	return distance(p, projection);
}
}
