#include <stdio.h>
#include "cudaScene.hpp"
#include "raytracer_cuda.hpp"
#include "helper_math.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cycleTimer.h"
#include "constants.hpp"
#define PI 3.1415926535

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

inline __device__ float3 pow_vec (float3 vec, float value)
{
	return make_float3(powf(vec.x, value), powf(vec.y, value), powf(vec.z, value));
}

inline __host__ __device__ float4 quaternionConjugate(float4 q)
{
	return make_float4(-q.x, -q.y, -q.z, q.w);
}

__constant__ CudaScene cuScene;
__constant__ PoolConstants cuConstants;

texture <float4, cudaTextureType2D> envmap;

void bindEnvmap (cudaArray * array, cudaChannelFormatDesc &channelDesc)
{
	envmap.addressMode[0] = cudaAddressModeWrap;
	envmap.addressMode[1] = cudaAddressModeWrap;
	envmap.filterMode = cudaFilterModeLinear;
	envmap.normalized = true;
	cudaBindTextureToArray(envmap, array, channelDesc);
}

__device__ float sphereIntersectionTest(float3 ray_d, float3 ray_e)
{
		float A = dot(ray_d, ray_d);
		float B = dot(ray_d, ray_e);
		float C = dot(ray_e, ray_e) - 1;
		float B24AC = B * B - A * C;
		if (B24AC >= 0) {
			float SB24AC = sqrt(B24AC);
			return (-B - SB24AC) / A;
		}
		return -1;
}

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

__device__ float3 doEnvironment( float3 rd )
{
	float r = 1 / 3.1415926 * acos(rd.z) / sqrt(rd.x * rd.x + rd.y * rd.y);
	
	float4 color_raw = tex2D(envmap, rd.x * r / 2 + 0.5, rd.y * r / 2 + 0.5);

	float3 color = make_float3(color_raw.x, color_raw.y, color_raw.z);

	//float3 color = make_float3(color_raw.x / 255.0, color_raw.y / 255.0, color_raw.z / 255.0);
	//return make_float3(500000.0, 0, 0);
	return 24 * make_float3(powf(color.x, 2.2), powf(color.y, 2.2), powf(color.z, 2.2));
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
	return k;
}
	
__device__ float sign(float x)
{
	if (x > 0.0)
		return 1.0;
	else
		return -1.0;
}


__device__ float trace_shadow(float3 ray_e, float3 ray_d)
{
	// Spheres Itest
	float res = 1.0, t = 0.0;
	float3 *pos_ptr = (float3 *)(cuScene.data + 4 * SPHERES);
	for (int i = 0; i < SPHERES; i++) {
		float3 t_ray_d = ray_d;
		float3 t_ray_e = ray_e - pos_ptr[i];
		// Intersection test
		t = sphereShadowTest(t_ray_d, t_ray_e);
		res = min(t, res);
	}
	return res;
}

__device__ float3 do_material (int geom, float3 diffuse, float3 normal, float3 pos)
{
	float3 mate = diffuse;
	float3 cue_color = make_float3(0.29, 0.27, 0.25);
	if (geom < SOLIDS) {
		mate = lerp(diffuse, cue_color, smoothstep(0.9, 0.91, abs(pos.y)));
	} else {
		mate = lerp(diffuse, cue_color, smoothstep(0.9, 0.91, abs(pos.y)) + smoothstep(0.55, 0.56, abs(pos.z)));
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


#define NSAMPLES 10
#define SHADOW_RAYS 5

__global__
void curandSetupKernel()
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = y * WIDTH + x;
	curand_init(1578, w, 0, cuConstants.curand + w);
}
		/*
		float di = (x + (sampleX + 0.5) / NSAMPLES) / WIDTH * 2 - 1;
		float dj = (y + (sampleY + 0.5) / NSAMPLES) / HEIGHT * 2 - 1;
		*/
__global__
void cudaRayTraceKernel (unsigned char *img, int y_start)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + y_start;
	int w = y * WIDTH + x;

	if(w >= WIDTH * HEIGHT)
		return;

	curandState *curand = cuConstants.curand + w;

	float3 *pos_ptr = (float3 *)(cuScene.data + 4 * SPHERES);
	float4 *rot_ptr = (float4 *)(cuScene.data);

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
		float di = (x + (sampleX + curand_uniform(curand)) / NSAMPLES) / WIDTH * 2 - 1;
		float dj = (y + (sampleY + curand_uniform(curand)) / NSAMPLES) / HEIGHT * 2 - 1;
		float3 ray_d = normalize(dir + dist * (dj * cU + di * AR * cR));
		float3 ray_e = *((float3 *) cuScene.cam_position);

		int geom = -1;
		float tmin = 10000.0;
		bool IsSpheres = true;
		// Spheres Itest
		for (int i = 0; i < SPHERES; i++) {
			float3 t_ray_d = ray_d;
			float3 t_ray_e = ray_e - pos_ptr[i];
			t_ray_d = quaternionXvector(quaternionConjugate(rot_ptr[i]), t_ray_d);
			t_ray_e = quaternionXvector(quaternionConjugate(rot_ptr[i]), t_ray_e);
			// Intersection test
			float t = sphereIntersectionTest(t_ray_d, t_ray_e);
			if (t > EPS && t < tmin) {
				geom = i;
				tmin = t;
			}
		}

		float t;
		// Planes Itest
		t = planeIntersectionTestY(ray_d, ray_e, 0);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 0; }
		t = planeIntersectionTestY(ray_d, ray_e, 1);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 1; }
		t = planeIntersectionTestY(ray_d, ray_e, 2);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 2; }
		t = planeIntersectionTestY(ray_d, ray_e, 3);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 3; }
		t = planeIntersectionTestY(ray_d, ray_e, 4);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 4; }
		t = planeIntersectionTestX(ray_d, ray_e, 5);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 5; }
		t = planeIntersectionTestX(ray_d, ray_e, 6);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 6; }
		t = planeIntersectionTestZ(ray_d, ray_e, 7);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 7; }
		t = planeIntersectionTestZ(ray_d, ray_e, 8);
		if (t > EPS && t < tmin) { IsSpheres = false; tmin = t; geom = 8; }

		tmin -= 0.001;
		float3 color = make_float3(0, 0, 0);

		if (geom >= 0) {
			float3 hit = tmin * ray_d + ray_e;
			// Normal
			float3 normal;
			if (IsSpheres) {
				normal = normalize(hit - pos_ptr[geom]);
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
				float3 orig_hit = quaternionXvector(quaternionConjugate(rot_ptr[geom]), hit - pos_ptr[geom]);
				float3 diffuse = cuConstants.sphere_colors[geom];

				// Direct diffuse
				float3 surface_color = make_float3(0, 0, 0);
				surface_color += shadow_factor * 2.9;

				float3 m = do_material(geom, diffuse, normal, orig_hit);
				// Bounce
				surface_color += 1.5f * clamp(0.3f-0.7f*normal.y,0.0f,1.0f)*make_float3(0.0,0.2,0.0);
				// Specular
				float fre = 0.04 + 4 * powf(clamp( 1.0 + dot(normal, ray_d), 0.0f, 1.0f ), 5.0f) ;
				float step = 0.0;
				float3 ref = normalize(ray_d - 2 * normal * dot(normal, ray_d));
				if (ref.y > 0)
					step = 1.0;
				//if (geom <= 3)
					surface_color += 1.0 * fre * step;// * doEnvironment(ref);
				color += surface_color * m;
				accumulated_color += color;
			} else {
				accumulated_color += cuConstants.plane_colors[geom] * shadow_factor;
			}
		} else {
			accumulated_color += make_float3(0.7, 0.9, 1.0);
		}
	}
	
	accumulated_color /= NSAMPLES * NSAMPLES;
	uchar4 col0;
	col0.x = clamp(__powf(accumulated_color.x, 0.45) * 255, 0.0, 255.0);
	col0.y = clamp(__powf(accumulated_color.y, 0.45) * 255, 0.0, 255.0);
	col0.z = clamp(__powf(accumulated_color.z, 0.45) * 255, 0.0, 255.0);
	col0.w = 255;
	*((uchar4 *)img + w) = col0;
	
}

void cudaInitialize()
{
	initialize_constants();
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
	cudaRayTraceKernel<<<dimGrid, dimBlock>>>(img, scene->y0);
	cudaDeviceSynchronize();
	printf("CUDA rendering time: %lf\n", CycleTimer::currentSeconds() - startTime);

	cudaError_t error = cudaGetLastError();
	if ( cudaSuccess != error )
			printf( "Error: %d\n", error );
}

				// Rim
				/*
				if (geom <= 3)
				surface_color *= 1.0 + 6.0 * m* powf( clamp( 1.0 + dot(normal, ray_d), 0.0f, 1.0f ), 2.0 );
				*/

