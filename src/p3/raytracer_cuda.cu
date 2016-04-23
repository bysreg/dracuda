#include <stdio.h>
#include "cudaScene.hpp"
#include "raytracer_cuda.hpp"
#include "helper_math.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cycleTimer.h"
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

__constant__ cudaScene cuScene;

texture <uchar4, cudaTextureTypeCubemap> envmap;

void bindEnvmap (cudaArray * array, cudaChannelFormatDesc &channelDesc)
{
	envmap.addressMode[0] = cudaAddressModeWrap;
	envmap.addressMode[1] = cudaAddressModeWrap;
	envmap.filterMode = cudaFilterModePoint;
	envmap.normalized = true;
	cudaBindTextureToArray(envmap, array, channelDesc);
}

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
__device__ float3 doEnvironment( float3 rd )
{
	uchar4 color_raw = texCubemap(envmap, rd.x, rd.y, rd.z);

	float3 color = make_float3(color_raw.x / 255.0, color_raw.y / 255.0, color_raw.z / 255.0);
	return 24.0 * make_float3(powf(color.x, 2.2), powf(color.y, 2.2), powf(color.z, 2.2));
}


__device__ float trace_shadow(float3 ray_e, float3 ray_d, float time)
{
	float3 *pos_ptr = (float3 *)cuScene.position;
	float4 *rot_ptr = (float4 *)cuScene.rotation;
	float3 *scl_ptr = (float3 *)cuScene.scale;
	for (int i = 0; i < cuScene.N; i++) {
		float3 t_ray_d = ray_d;
		float3 t_ray_e = ray_e - pos_ptr[i];
		t_ray_d = quaternionXvector(quaternionConjugate(rot_ptr[i]), t_ray_d);
		t_ray_e = quaternionXvector(quaternionConjugate(rot_ptr[i]), t_ray_e);
		t_ray_d = t_ray_d / scl_ptr[i];
		t_ray_e = t_ray_e / scl_ptr[i];
		// Intersection test
		float t = intersectionTest(cuScene.type[i], t_ray_d, t_ray_e, i);
		if (t > EPS && t < time) {
			return 0;
		}
	}
	return 1;
}

__device__ float distanceToSegment( float2 a, float2 b, float2 p )
{
		float2 pa = p - a;
			float2 ba = b - a;
				float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
					
					return length( pa - ba*h );
}

__device__ float sign(float x)
{
	if (x > 0.0)
		return 1.0;
	else
		return -1.0;
}
__device__ float3 do_material (int geom, float3 diffuse, float3 normal, float3 pos)
{
	float3 mate = diffuse;
	float3 cue_color = make_float3(0.29, 0.27, 0.25);
	// Cue
	if( geom == 3) {
		mate = make_float3(0.30,0.25,0.20)*1.25; }
	// Blue
	if( geom == 2) {
		mate = make_float3(0.00,0.10,0.20)*1.25; 
		mate = lerp( mate, make_float3(0.29, 0.27, 0.25 ), smoothstep( 0.9, 0.91, abs(pos.z) ) ); 
		float d = distanceToSegment( make_float2(0.22,0.0), make_float2(-0.22,0.0), make_float2(pos.y, pos.x) );
		mate *= smoothstep( 0.04, 0.05, d );
	}
	// Yellow 11
	if (geom == 1) {
		mate *= 1.25;
		mate = lerp( mate, cue_color, smoothstep( 0.9, 0.91, abs(pos.x) ) ); 
		float d1 = distanceToSegment( make_float2(0.22,0.12), make_float2(-0.22,0.12), make_float2(pos.y, pos.z) );
		float d2 = distanceToSegment( make_float2(0.22,-0.12), make_float2(-0.22,-0.12), make_float2(pos.y, pos.z) );
		float d = min( d1, d2 );
		mate *= smoothstep( 0.04, 0.05, d );
	}
	if (geom == 0) {
		mate *= 1.25;
		float2 yz = make_float2(pos.y, pos.z);
		mate = lerp( mate, cue_color, smoothstep( 0.9, 0.91, abs(pos.x) ) + smoothstep( 0.55, 0.56, abs(pos.y) ) ); 
		float d1 = distanceToSegment( make_float2(0.22,0.0), make_float2(-0.22,0.0), yz );
		float d2 = distanceToSegment( make_float2(0.22,0.0), make_float2( -0.07,-0.2), yz*make_float2(1.0,-sign(pos.x)) );
		float d3 = distanceToSegment( make_float2(-0.07,-0.2), make_float2(-0.07,0.04), yz*make_float2(1.0,-sign(pos.x)) );
		float d = min(d1,min(d2,d3));
		mate *= smoothstep( 0.04, 0.05, d );
	}
	if (geom > 3) {
		mate *= 1.25;
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int w = y * cuScene.width + x;
		//mate *= 0.78 + 0.22 * curand_uniform(cuScene.curand + w);
	}
	return mate;
}

#define NSAMPLES 10

__global__
void curandSetupKernel()
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = y * cuScene.width + x;
	curand_init(1578, w, 0, cuScene.curand + w);
}
__global__
void cudaRayTraceKernel (unsigned char *img)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = y * cuScene.width + x;

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
	for (int bounce = 0; bounce < 1; bounce ++) {
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
		tmin -= 0.001;
		float3 color = make_float3(0, 0, 0);

		if (geom >= 0) {
			int material = cuScene.material[geom];
			float3 hit = tmin * ray_d + ray_e;
			float3 orig_hit = quaternionXvector(quaternionConjugate(rot_ptr[geom]), hit - pos_ptr[geom]) / scl_ptr[geom];
			int type = cuScene.type[geom];
			float3 normal;
			// Calc normal
			if (type == 1) {
				normal = orig_hit;
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
			// Russian roulette
			float3 diffuse = ((float3 *)cuScene.diffuse)[material];
			float3 specular = ((float3 *)cuScene.specular)[material];

			// Direct diffuse
			float3 surface_color = make_float3(0, 0, 0);
			for (int i = 0; i < 3; i++) {
				float3 sdir, tdir;
				float u = curand_uniform(cuScene.curand + w);
				float phi = 2 * 3.1415926535 * curand_uniform(cuScene.curand + w);
				if (abs(normal.x) < 0.5) {
					sdir = cross(normal, make_float3(1, 0, 0));
				} else {
					sdir = cross(normal, make_float3(0, 1, 0));
				}
				tdir = cross(normal, sdir);
				float3 light_dir = sqrt(u) * (cos(phi) * sdir + sin(phi) * tdir) + sqrt(1 - u) * normal;
				float shadow_factor = trace_shadow(hit, light_dir, 10000.0);
				surface_color += shadow_factor;
			}
			surface_color *= 1 / 3.0 * 2.9;

			float3 m = do_material(geom, diffuse, normal, orig_hit);
			surface_color += 1.5f * clamp(0.3f-0.7f*normal.y,0.0f,1.0f)*make_float3(0.0,0.2,0.0);
			// Rim
			if (geom <= 3)
			surface_color *= 1.0 + 6.0 * m* powf( clamp( 1.0 + dot(normal, ray_d), 0.0f, 1.0f ), 2.0 );

			float fre = 0.04 + 4 * powf(clamp( 1.0 + dot(normal, ray_d), 0.0f, 1.0f ), 5.0f) ;
			float step = 0.0;
			float3 ref = ray_d - 2 * normal * dot(normal, ray_d);
			if (ref.y > 0)
				step = 1.0;

			if (geom <= 3)
			surface_color += 1.0 * doEnvironment(ref ) * fre * step;
			color += surface_color * m;
			accumulated_color += color;//; * color_mask;
			color_mask *= surface_color;
		} else {
			accumulated_color += (*(float3 *)cuScene.ambient_light_col + color) * color_mask;
			break;
		}
	}
	}
	
	accumulated_color /= NSAMPLES * NSAMPLES;
	img[4 * w + 0] = clamp(__powf(accumulated_color.x, 0.45) * 255, 0.0, 255.0);
	img[4 * w + 1] = clamp(__powf(accumulated_color.y, 0.45) * 255, 0.0, 255.0);
	img[4 * w + 2] = clamp(__powf(accumulated_color.z, 0.45) * 255, 0.0, 255.0);
	img[4 * w + 3] = 255;
}

void cudaRayTrace(cudaScene *scene, unsigned char *img)
{
	printf("%p\n", img);
	gpuErrchk(cudaMemcpyToSymbol(cuScene, scene, sizeof(cudaScene)));
	dim3 dimBlock(16, 16);
	dim3 dimGrid(scene->width / 16, scene->height / 16);
	cudaGetLastError();
	curandSetupKernel<<<dimGrid, dimBlock>>>();

	double startTime = CycleTimer::currentSeconds();
	cudaRayTraceKernel<<<dimGrid, dimBlock>>>(img);
	printf("CUDA rendering time: %lf\n", CycleTimer::currentSeconds() - startTime);

	cudaError_t error = cudaGetLastError();
	if ( cudaSuccess != error )
			printf( "Error: %d\n", error );

}
