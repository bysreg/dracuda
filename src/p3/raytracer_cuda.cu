#include <stdio.h>
#include "cudaScene.hpp"
#include "raytracer_cuda.hpp"
#include "helper_math.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cycleTimer.h"
#include "constants.hpp"
#include "math/random462.hpp"
#define PI 3.1415926535

#define EPS 0.0001

#define NSAMPLES 5
#define SHADOW_RAYS 5

unsigned char *cudaBuffer;

inline __device__ float3 quaternionXvector(float4 q, float3 vec)
{
	float3 qvec = make_float3(q.x, q.y, q.z);
	float3 uv = cross(qvec, vec);
	float3 uuv = cross(qvec, uv);
	uv *= (2.0 * q.w);
	uuv *= 2.0;
	return vec + uv + uuv;
}

inline __device__ float3 quaternionXCvector(float4 q, float3 vec)
{
	float3 qvec = make_float3(-q.x, -q.y, -q.z);
	float3 uv = cross(qvec, vec);
	float3 uuv = cross(qvec, uv);
	uv *= (2.0 * q.w);
	uuv *= 2.0;
	return vec + uv + uuv;
}

__constant__ CudaScene cuScene;
__constant__ PoolConstants cuConstants;

__device__ float sphereShadowTest(float3 ray_d, float3 ray_e)
{
	float res = 1.0;
	float b = -dot(ray_e, ray_d);
	if (b < 0.0) {
		res = 1.0;
	} else {
		float h = sqrt(dot(ray_e, ray_e) - b * b) - 1;
		res = clamp(16.0f * h / b, 0.0f, 1.0f);
	}
	return res;
}

__device__ float planeIntersectionTestX(float3 ray_d, float3 ray_e, int geom)
{
	float t = (cuConstants.positions[geom] - ray_e.x) / ray_d.x;
	float3 hit = t * ray_d + ray_e;
	float3 vec = (hit - cuConstants.lower_bounds[geom]) * (hit - cuConstants.upper_bounds[geom]);
	if (vec.y > 0 || vec.z > 0)
		t = -1;
	return t;
}

__device__ float planeIntersectionTestY(float3 ray_d, float3 ray_e, int geom)
{
	float t = (cuConstants.positions[geom] - ray_e.y) / ray_d.y;
	float3 hit = t * ray_d + ray_e;
	float3 vec = (hit - cuConstants.lower_bounds[geom]) * (hit - cuConstants.upper_bounds[geom]);
	if (vec.x > 0 || vec.z > 0)
		t = -1;
	return t;
}
__device__ float planeIntersectionTestZ(float3 ray_d, float3 ray_e, int geom)
{
	float t = (cuConstants.positions[geom] - ray_e.z) / ray_d.z;
	float3 hit = t * ray_d + ray_e;
	float3 vec = (hit - cuConstants.lower_bounds[geom]) * (hit - cuConstants.upper_bounds[geom]);
	if (vec.x > 0 || vec.y > 0)
		t = -1;
	return t;
}

__device__ float distanceToSegment( float2 a, float2 b, float2 p )
{
	float2 pa = p - a;
	float2 ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h );
}

__device__ float circle2(float2 pos, float2 center, float radius, float dist, float begin, float interval)
{
	float2 diff = pos - center;
	float angle = atan2(diff.y, diff.x) - begin;
	if (angle < 0.0)
		angle = angle + 2.0 * PI;
	float d = sqrt(dot(diff, diff));
	float k = abs(d - radius) / dist;
	if (angle > interval)
		k = 1.0;
	dist = dist * dist / 1.07;
	float2 point = make_float2(cos(begin) * radius + center.x, sin(begin) * radius + center.y);
	diff = pos - point;
	d = dot(diff, diff);
	if (d < dist)
		k = min(k, d / dist);
	point = make_float2(cos(begin + interval) * radius + center.x, sin(begin + interval) * radius + center.y);
	diff = pos - point;
	d = dot(diff, diff);
	if (d < dist)
		k = min(k, d / dist);
	return k;
}
	
__device__ float trace_shadow(float3 ray_e, float3 ray_d)
{
	// Spheres Itest
	if (ray_d.y < 0)
		return 0.0;
	float res = 1.0, t;
	for (int i = 0; i < SPHERES; i++) {
		// Intersection test
		t = sphereShadowTest(ray_d, ray_e - cuScene.ball_position[i]);
		res = min(t, res);
	}
	return res;
}

__device__ float3 do_material (int geom, float3 pos)
{
	float3 mate = cuConstants.sphere_colors[geom];
	float3 cue_color = make_float3(0.29, 0.27, 0.25);
	if (geom < SOLIDS) {
		mate = lerp(mate, cue_color, smoothstep(0.9, 0.91, abs(pos.y)));
	} else {
		mate = lerp(mate, cue_color, smoothstep(0.9, 0.91, abs(pos.y)) + smoothstep(0.55, 0.56, abs(pos.z)));
	}
	float d1 = 1.0, d2 = 1.0, d3 = 1.0, d4 = 1.0, k1 = 1.0, k2 = 1.0;
	float d, k;
	float2 xz;
	if (pos.y > 0) { 
		xz = make_float2(pos.x, pos.z);
	} else {
		xz = make_float2(-pos.x, pos.z);
	}
	switch(geom) {
		case 1:
			d1 = distanceToSegment( make_float2(0.0,0.22), make_float2(0.0,-0.22), xz);
			break;
		case 2:
			d1 = distanceToSegment(make_float2(-0.1, 0.22), make_float2(0.075, 0.0), xz);
			d2 = distanceToSegment(make_float2(-0.1, 0.22), make_float2(0.1, 0.22), xz);
			k1 = circle2(xz, make_float2(0.0, -0.08), 0.107, 0.045, PI, PI + 0.62);
			break;
		case 3:
			k1 = circle2(xz, make_float2(0.0, -0.105), 0.107, 0.045, 0.5 - PI, PI + 1.0);
			k2 = circle2(xz, make_float2(0.0, 0.105), 0.107, 0.045, -1.5, PI + 1.0);
			break;
		case 4:
			d1 = distanceToSegment(make_float2(0.0, -0.22), make_float2(0.0, 0.22), xz);
			d2 = distanceToSegment(make_float2(0.0, -0.22), make_float2(-0.2, 0.07), xz);
			d3 = distanceToSegment(make_float2(-0.2, 0.07), make_float2(0.04, 0.07), xz);
			break;
		case 5:
			d1 = distanceToSegment(make_float2(-0.09, -0.22), make_float2(-0.09, 0.0), xz);
			d2 = distanceToSegment(make_float2(-0.09, -0.22), make_float2(0.11, -0.22), xz);
			k1 = circle2(xz, make_float2(0.0, 0.1), 0.13, 0.045, -2.1, PI + 1.4);
			break;
		case 6:
			k1 = circle2(xz, make_float2(0.0, 0.11), 0.12, 0.045, 0.0, PI * 2.0);
			k2 = circle2(xz, make_float2(0.0, -0.11), 0.12, 0.045, PI, 2.9);
			d1 = distanceToSegment(make_float2(-0.12, -0.12), make_float2(-0.12, 0.1), xz);
			break;
		case 7:
			d1 = distanceToSegment(make_float2(0.1, -0.22), make_float2(-0.03, 0.22), xz);
			d2 = distanceToSegment(make_float2(-0.1, -0.22), make_float2(0.1, -0.22), xz);
			break;
		case 8:
			k1 = circle2(xz, make_float2(0.0, -0.11), 0.1, 0.045, 0.0, PI * 2.0);
			k2 = circle2(xz, make_float2(0.0, 0.12), 0.12, 0.045, 0.0, PI * 2.0);
			break;
		case 9:
			k1 = circle2(xz, make_float2(0.0, -0.105), 0.12, 0.045, 0.0, PI * 2.0);
			k2 = circle2(xz, make_float2(0.0, 0.105), 0.12, 0.045, 3.03 - PI, 2.6);
			d1 = distanceToSegment(make_float2(0.12, -0.12), make_float2(0.12, 0.1), xz);
			break;
		case 10:
			d1 = distanceToSegment(make_float2(-0.16, 0.22), make_float2(-0.16, -0.22), xz);
			d2 = distanceToSegment(make_float2(-0.02, 0.12), make_float2(-0.02, -0.12), xz);
			d3 = distanceToSegment(make_float2(0.221, 0.12), make_float2(0.221, -0.12), xz);
			k1 = circle2(xz, make_float2(0.1, -0.105), 0.12, 0.045, PI, PI);
			k2 = circle2(xz, make_float2(0.1, 0.105), 0.12, 0.045, 0.0, PI);
			break;
		case 11:
			d1 = distanceToSegment(make_float2(0.12, 0.22), make_float2(0.12, -0.22), xz);
			d2 = distanceToSegment(make_float2(-0.12, 0.22), make_float2(-0.12, -0.22), xz);
			break;
		case 12:
			d3 = distanceToSegment(make_float2(-0.16, 0.22), make_float2(-0.16, -0.19), xz);
			d1 = distanceToSegment(make_float2(0.0, 0.22), make_float2(0.175, 0.0), xz);
			d2 = distanceToSegment(make_float2(0.0, 0.22), make_float2(0.2, 0.22), xz);
			k1 = circle2(xz, make_float2(0.1, -0.08), 0.107, 0.045, PI, PI + 0.62);
			break;
		case 13:
			d3 = distanceToSegment(make_float2(-0.16, 0.21), make_float2(-0.16, -0.21), xz);
			k1 = circle2(xz, make_float2(0.1, -0.105), 0.107, 0.045, 0.5 - PI, PI + 1.0);
			k2 = circle2(xz, make_float2(0.1, 0.105), 0.107, 0.045, -1.5, PI + 1.0);
			break;
		case 14:
			d4 = distanceToSegment(make_float2(-0.16, 0.22), make_float2(-0.16, -0.22), xz);
			d1 = distanceToSegment(make_float2(0.15, -0.22), make_float2(0.15, 0.22), xz);
			d2 = distanceToSegment(make_float2(0.15, -0.22), make_float2(-0.05, 0.07), xz);
			d3 = distanceToSegment(make_float2(-0.05, 0.07), make_float2(0.19, 0.07), xz);
			break;
		case 15:
			d3 = distanceToSegment(make_float2(-0.16, 0.22), make_float2(-0.16, -0.22), xz);
			d1 = distanceToSegment(make_float2(0.01, -0.22), make_float2(0.01, 0.0), xz);
			d2 = distanceToSegment(make_float2(0.01, -0.22), make_float2(0.21, -0.22), xz);
			k1 = circle2(xz, make_float2(0.1, 0.1), 0.13, 0.045, -2.0, PI + 1.3);
			break;
		default:
			break;
	}
	d = min(min(d1, d4), min(d2, d3));
	k = min(k1, k2);
	mate *= smoothstep(0.04, 0.045, d) * smoothstep(0.88, 1.0, k);
	return mate;
}


__global__
void curandSetupKernel()
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = y * WIDTH + x;
	curand_init(1578, w, 0, cuConstants.curand + w);
}

__device__ float sphereIntersectionTestAll(float3 ray_d, float3 ray_e, int &geom)
{
	float tmin = 10000.0;
	// Spheres Itest
	for (int i = 0; i < SPHERES; i++) {
		float3 t_ray_e = ray_e - cuScene.ball_position[i];
		// Intersection test
		float t = 10000.0;
		float B = dot(ray_d, t_ray_e);
		float C = dot(t_ray_e, t_ray_e) - 1;
		float B24AC = B * B - C;
		if (B24AC >= 0)
			t = -B - sqrt(B24AC);
		if (t > EPS && t < tmin) {
			geom = i;
			tmin = t;
		}
	}
	return tmin;
}

__device__
float planeIntersectionTestAll(float3 ray_d, float3 ray_e, int &geom, float tmin)
{
	// Planes Itest
	float t;
	t = planeIntersectionTestY(ray_d, ray_e, 0);
	if (t > EPS && t < tmin) { tmin = t; geom = 0; }
	t = planeIntersectionTestY(ray_d, ray_e, 1);
	if (t > EPS && t < tmin) { tmin = t; geom = 1; }
	t = planeIntersectionTestY(ray_d, ray_e, 2);
	if (t > EPS && t < tmin) { tmin = t; geom = 2; }
	t = planeIntersectionTestY(ray_d, ray_e, 3);
	if (t > EPS && t < tmin) { tmin = t; geom = 3; }
	t = planeIntersectionTestY(ray_d, ray_e, 4);
	if (t > EPS && t < tmin) { tmin = t; geom = 4; }
	t = planeIntersectionTestX(ray_d, ray_e, 5);
	if (t > EPS && t < tmin) { tmin = t; geom = 5; }
	t = planeIntersectionTestX(ray_d, ray_e, 6);
	if (t > EPS && t < tmin) { tmin = t; geom = 6; }
	t = planeIntersectionTestZ(ray_d, ray_e, 7);
	if (t > EPS && t < tmin) { tmin = t; geom = 7; }
	t = planeIntersectionTestZ(ray_d, ray_e, 8);
	if (t > EPS && t < tmin) { tmin = t; geom = 8; }
	return tmin;
}

__global__
void cudaRayTraceKernel (unsigned char *img, int y_start)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + y_start;
	int w = y * WIDTH + x;

	if(w >= WIDTH * HEIGHT)
		return;

	curandState *curand = cuConstants.curand + w;

	// Calc Ray
	float3 accumulated_color = make_float3(0.0, 0.0, 0.0);

	// Jittered Sampling
	for (int sampleX = 0; sampleX < NSAMPLES; sampleX++)
	for (int sampleY = 0; sampleY < NSAMPLES; sampleY++) {
		float di = (x + (sampleX + curand_uniform(curand)) / NSAMPLES) / WIDTH * 2 - 1;
		float dj = (y + (sampleY + curand_uniform(curand)) / NSAMPLES) / HEIGHT * 2 - 1;
		float3 ray_d = normalize(cuScene.dir + dj * cuScene.cU + di * cuScene.ARcR);
		float3 ray_e = cuScene.cam_position;

		int geom = -1, geom2 = -1;
		bool IsSpheres = true;

		float tmin = sphereIntersectionTestAll(ray_d, ray_e, geom);
		tmin = planeIntersectionTestAll(ray_d, ray_e, geom2, tmin);
		if (geom2 >= 0) {
			IsSpheres = false;
			geom = geom2;
		}
		tmin -= 0.001;
		float3 color = make_float3(0, 0, 0);

		if (geom >= 0) {
			float3 hit = tmin * ray_d + ray_e;
			// Normal
			float3 normal;
			if (IsSpheres) {
				normal = normalize(hit - cuScene.ball_position[geom]);
			} else {
				normal = cuConstants.normals[geom];
			}
			// Shadow Factor
			float shadow_factor = 0.0;
			for (int i = 0; i < SHADOW_RAYS; i++) {
				float3 sdir, tdir;
				float u = curand_uniform(curand);
				float phi = 2 * 3.1415926535 * curand_uniform(curand);
				if (abs(normal.x) < 0.5) {
					sdir = cross(normal, make_float3(1, 0, 0));
				} else {
					sdir = cross(normal, make_float3(0, 1, 0));
				}
				tdir = cross(normal, sdir);
				float3 light_dir = sqrt(u) * (cos(phi) * sdir + sin(phi) * tdir) + sqrt(1 - u) * normal;
				shadow_factor += trace_shadow(hit, light_dir);
			}
			shadow_factor /= (SHADOW_RAYS + 0.0);
			if (IsSpheres) {
				float3 orig_hit = quaternionXCvector(cuScene.ball_orientation[geom], hit - cuScene.ball_position[geom]);
				float3 m = do_material(geom, orig_hit);
				// Bounce
				float3 surface_color = shadow_factor * 2.9f + 1.5f * clamp(0.3f-0.7f*normal.y,0.0f,1.0f)*make_float3(0.0,0.2,0.0);
				// Specular
				float fre = 0.04 + 4 * powf(clamp( 1.0 + dot(normal, ray_d), 0.0f, 1.0f ), 5.0f) ;
				float3 ref = normalize(ray_d - 2 * normal * dot(normal, ray_d));
				int geom_ref = -1;

				float tmin = sphereIntersectionTestAll(ref, hit, geom_ref);
				float3 fresnel_color = make_float3(0, 0, 0);
				if (geom_ref >= 0) {
					float3 hit_ref = ref * tmin + hit;
					float3 orig_hit_ref = quaternionXCvector(cuScene.ball_orientation[geom], hit_ref - cuScene.ball_position[geom]);
					float3 m_ref = do_material(geom_ref, orig_hit_ref);
					fresnel_color = m_ref;
				}
				float3 light_dir = normalize(make_float3(0.7, 1.0, -0.8));
				accumulated_color += surface_color * m + fre * fresnel_color / 2;
				accumulated_color += make_float3(0.3, 0.3, 0.3) * powf(clamp(dot(light_dir, ref), 0.0f, 1.0f), 50.0f);
			} else {
				accumulated_color += cuConstants.plane_colors[geom] * shadow_factor;
			}
		} else {
			accumulated_color += make_float3(0.7, 0.9, 1.0);
		}
	}
	
	accumulated_color /= NSAMPLES * NSAMPLES;
	// using 3 color per pixel
	uchar3 col0;
	col0.x = clamp(__powf(accumulated_color.x, 0.45) * 255, 0.0, 255.0);
	col0.y = clamp(__powf(accumulated_color.y, 0.45) * 255, 0.0, 255.0);
	col0.z = clamp(__powf(accumulated_color.z, 0.45) * 255, 0.0, 255.0);
	// col0.w = 255;
	*((uchar3 *)img + x + (y - y_start) * WIDTH) = col0;
	
}

void cudaInitialize()
{
	initialize_constants();
	gpuErrchk(cudaMalloc((void **)&cudaBuffer, PIXEL_SIZE * HEIGHT * WIDTH));
	gpuErrchk(cudaMalloc((void **)&poolConstants.curand, sizeof(curandState) * WIDTH * HEIGHT));
	gpuErrchk(cudaMemcpyToSymbol(cuConstants, &poolConstants, sizeof(PoolConstants)));
	dim3 dimBlock(16, 16);
	dim3 dimGrid(WIDTH / 16, HEIGHT / 16);
	curandSetupKernel<<<dimGrid, dimBlock>>>();
	cudaDeviceSynchronize();
}

void cudaRayTrace(CudaScene *scene, unsigned char *img)
{
	//printf("CudaRayTrace\n");
	//printf("%p\n", scene);
	gpuErrchk(cudaMemcpyToSymbol(cuScene, scene, sizeof(CudaScene)));
	int height = scene->render_height;

	dim3 dimBlock(16, 16);
	dim3 dimGrid(WIDTH / 16, (height + 16 - 1) / 16);

	double startTime = CycleTimer::currentSeconds();
	cudaRayTraceKernel<<<dimGrid, dimBlock>>>(cudaBuffer, scene->y0);
	cudaDeviceSynchronize();
	printf("CUDA rendering time: %lf\n", CycleTimer::currentSeconds() - startTime);

	cudaError_t error = cudaGetLastError();
	if ( cudaSuccess != error )
			printf( "Error: %d\n", error );
	gpuErrchk(cudaMemcpy(img, cudaBuffer, PIXEL_SIZE * WIDTH * height , cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(img, cudaBuffer + (scene->y0 * WIDTH * PIXEL_SIZE), PIXEL_SIZE * WIDTH * height, cudaMemcpyDeviceToHost));
}
