#include <stdio.h>
#include "cudaScene.hpp"
#include "raytracer_cuda.hpp"
#include "helper_math.h"
#include <curand.h>
#include <curand_kernel.h>
#define EPS 0.0001

inline __host__ __device__ float3 quaternionXvector(float4 q, float3 vec)
{
	float3 qvec = make_float3(q.x, q.y, q.z);
	float3 uv = cross(qvec, vec);
	float3 uuv = cross(qvec, uv);
	uv *= (2.0 * q.w);
	uuv *= 2.0;
	return vec + uv + uuv;
}

inline __host__ __device__ float4 quaternionConjugate(float4 q)
{
	return make_float4(-q.x, -q.y, -q.z, q.w);
}

__constant__ cudaScene cuScene;

__device__ float intersectionTest(int type, float3 ray_d, float3 ray_e, int geom)
{
	if (type == 1) {
		float A = dot(ray_d, ray_d);
		float B = dot(ray_d, ray_e);
		float C = dot(ray_e, ray_e) - 1;
		float B24AC = B * B - A * C;
		if (B24AC >= 0) {
			float SB24AC = sqrt(B24AC);
			return (-B - SB24AC) / A;
		}
		return -1;
	} else if (type == 2) {
		float3 v0 = ((float3 *)cuScene.vertex0)[geom];
		float3 v1 = ((float3 *)cuScene.vertex1)[geom];
		float3 v2 = ((float3 *)cuScene.vertex2)[geom];
		float3 t1 = cross(v0 - v2, ray_d);
		float3 t2 = cross(v0 - v1, v0 - ray_e);
		float detA = dot((v0 - v1) ,  t1);
		float distance = dot(v2 - v0, t2) / detA;
		if (distance < EPS)
			return -1;
		float beta = dot(v0 - ray_e, t1) / detA;
		if (beta < 0)
			return -1;
		float gamma = dot(ray_d , t2) / detA;
		if (gamma >= 0 && (beta + gamma) <= 1)
			return distance;
		return -1;
	}
	return -1;
}

#define NSAMPLES 10
__global__
void cudaRayTraceKernel (unsigned char *img)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = y * cuScene.width + x;
	curand_init(1578, w, 0, cuScene.curand + w);

	img[4 * w + 0] = 255;
	img[4 * w + 1] = 0;
	img[4 * w + 2] = 0;
	img[4 * w + 3] = 0;

	// Calc Ray
	float3 dir = quaternionXvector(*((float4 *)cuScene.cam_orientation), make_float3(0, 0, -1));
	float3 up = quaternionXvector(*((float4 *)cuScene.cam_orientation), make_float3(0, 1, 0));
	float AR = cuScene.aspect;
	float3 cR = cross(dir, up);
	float3 cU = cross(cR, dir);
	float dist = tan(cuScene.fov / 2.0);
	float3 accumulated_color = make_float3(0.0, 0.0, 0.0);

	// Jittered Sampling

	for (int sampleX = 0; sampleX < NSAMPLES; sampleX++)
		for (int sampleY = 0; sampleY < NSAMPLES; sampleY++) {

	float di = (x + (sampleX + curand_uniform(cuScene.curand + w)) / NSAMPLES) / cuScene.width * 2 - 1;
	float dj = (y + (sampleY + curand_uniform(cuScene.curand + w)) / NSAMPLES) / cuScene.height * 2 - 1;
	float3 ray_d = normalize(dir + dist * (dj * cU + di * AR * cR));
	float3 ray_e = *((float3 *) cuScene.cam_position);

	float3 *pos_ptr = (float3 *)cuScene.position;
	float4 *rot_ptr = (float4 *)cuScene.rotation;
	float3 *scl_ptr = (float3 *)cuScene.scale;

	float3 color_mask = make_float3(1.0, 1.0, 1.0);
	for (int bounce = 0; bounce < 5; bounce ++) {
		int geom = -1;
		float tmin = 10000.0;
		float3 averager = make_float3(1.0 / 3, 1.0 / 3, 1.0 / 3);

		for (int i = 0; i < cuScene.N; i++) {
			float3 t_ray_d = ray_d;
			float3 t_ray_e = ray_e - pos_ptr[i];
			t_ray_d = quaternionXvector(quaternionConjugate(rot_ptr[i]), t_ray_d);
			t_ray_e = quaternionXvector(quaternionConjugate(rot_ptr[i]), t_ray_e);
			t_ray_d = t_ray_d / scl_ptr[i];
			t_ray_e = t_ray_e / scl_ptr[i];
			// Intersection test
			float t = intersectionTest(cuScene.type[i], t_ray_d, t_ray_e, i);
			if (t > EPS && t < tmin) {
				geom = i;
				tmin = t;
			}
		}
		float3 color = make_float3(0, 0, 0);
		// Light
		for (int i = 0; i < cuScene.N_light; i++)
		{
			float radius = cuScene.light_radius[i];
			float3 ray2 = ray_e - ((float3 *)cuScene.light_pos)[i];
			float A = dot(ray_d, ray_d);
			float B = dot(ray_d, ray2);
			float C = dot(ray2, ray2) - radius * radius;
			float B24AC = B * B - A * C;
			if (B24AC >= 0) {
				float SB24AC = sqrt(B24AC);
				if (B + SB24AC < 0) {
					color += ((float3 *)cuScene.light_col)[i];
				}
			}
			
		}
		if (geom >= 0) {
			int material = cuScene.material[geom];
			float3 hit = tmin * ray_d + ray_e - pos_ptr[geom];
			hit = quaternionXvector(quaternionConjugate(rot_ptr[geom]), hit) / scl_ptr[geom];
			int type = cuScene.type[geom];
			float3 normal;
			// Calc normal
			if (type == 1) {
				normal = hit;
			} else if (type == 2) {
				float3 v0 = ((float3 *)cuScene.vertex0)[geom];
				float3 v1 = ((float3 *)cuScene.vertex1)[geom];
				float3 v2 = ((float3 *)cuScene.vertex2)[geom];
				normal = cross(v1 - v0, v2 - v0);
			}
			// Normal matrix
			normal = normal / scl_ptr[geom];
			normal = quaternionXvector(rot_ptr[geom], normal);
			normal = normalize(normal);

			// Ambient
			float3 ambient = ((float3 *)cuScene.ambient)[material];
			color += ambient * (*(float3 *)cuScene.ambient_light_col);

			// Russian roulette
			float3 diffuse = ((float3 *)cuScene.diffuse)[material];
			float3 specular = ((float3 *)cuScene.specular)[material];
			float diffuse_prob = dot(diffuse, averager);
			float specular_prob = dot(specular, averager);
			float sum = diffuse_prob + specular_prob;
			diffuse_prob /= sum;
			specular_prob /= sum;

			if (curand_uniform(cuScene.curand + w) < diffuse_prob) {
				// Direct diffuse
				float3 surface_color = make_float3(0, 0, 0);
				for (int i = 0; i < cuScene.N_light; i++) {
					float3 light_pos = ((float3 *)cuScene.light_pos)[i];
					float3 light_dir = normalize(light_pos - hit); 
					float cos_factor = dot(light_dir, normal);
					if (cos_factor > 0)
						surface_color += diffuse * cos_factor;
				}
				color += surface_color;
				accumulated_color += color * color_mask;
				color_mask *= surface_color;
				float3 sdir, tdir;
				// Random direction
				float u = curand_uniform(cuScene.curand + w);
				float phi = 2 * 3.1415926535 * curand_uniform(cuScene.curand + w);
				if (abs(normal.x) < 0.5) {
					sdir = cross(normal, make_float3(1, 0, 0));
				} else {
					sdir = cross(normal, make_float3(0, 1, 0));
				}
				tdir = cross(normal, sdir);
				ray_d = sqrt(u) * (cos(phi) * sdir + sin(phi) * tdir) + sqrt(1 - u) * normal;
				ray_e = hit;
			} else {
				// Specular
				accumulated_color += color * color_mask;
				color_mask *= specular;
				ray_e = hit;
				ray_d = ray_d - 2 * normal * dot(normal, ray_d);
			}
		} else {
		}
	}
	}
	
	accumulated_color /= NSAMPLES * NSAMPLES;
	img[4 * w + 0] = clamp(accumulated_color.x * 255, 0.0, 255.0);
	img[4 * w + 1] = clamp(accumulated_color.y * 255, 0.0, 255.0);
	img[4 * w + 2] = clamp(accumulated_color.z * 255, 0.0, 255.0);
	img[4 * w + 3] = 255;
}

void cudaRayTrace(cudaScene *scene, unsigned char *img)
{
	printf("%p\n", img);
	gpuErrchk(cudaMemcpyToSymbol(cuScene, scene, sizeof(cudaScene)));
	dim3 dimBlock(16, 16);
	dim3 dimGrid(scene->width / 16, scene->height / 16);
	cudaRayTraceKernel<<<dimGrid, dimBlock>>>(img);
}

void helloInvoke()
{
	int a[64];
	int b[64];
	int *dev_a;
	int *dev_b;
	for (int i = 0; i < 64; i++)
		b[i] = i;
	gpuErrchk(cudaMalloc((void **)&dev_a, sizeof(int) * 64));
	gpuErrchk(cudaMalloc((void **)&dev_b, sizeof(int) * 64));
	gpuErrchk(cudaMemcpy(dev_b, b, sizeof(int) * 64, cudaMemcpyHostToDevice));
	dim3 dimBlock(64);
	dim3 dimGrid(1);
	//hello<<<dimGrid, dimBlock>>>(dev_a, dev_b);
	gpuErrchk(cudaMemcpy(a, dev_a, sizeof(int) * 64, cudaMemcpyDeviceToHost));
	cudaFree(dev_a);
	cudaFree(dev_b);
	for (int i = 0; i < 64; i++)
		printf("%d\n", a[i]);
	return;
	
}
