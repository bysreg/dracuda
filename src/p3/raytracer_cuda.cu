#include <stdio.h>
#include "cudaScene.hpp"
#include "raytracer_cuda.hpp"
#include "helper_math.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cycleTimer.h"
#include "constants.hpp"

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
__device__ float3 doEnvironment( float3 rd )
{
	float r = 1 / 3.1415926 * acos(rd.z) / sqrt(rd.x * rd.x + rd.y * rd.y);
	
	float4 color_raw = tex2D(envmap, rd.x * r / 2 + 0.5, rd.y * r / 2 + 0.5);

	float3 color = make_float3(color_raw.x, color_raw.y, color_raw.z);

	//float3 color = make_float3(color_raw.x / 255.0, color_raw.y / 255.0, color_raw.z / 255.0);
	//return make_float3(500000.0, 0, 0);
	return 24 * make_float3(powf(color.x, 2.2), powf(color.y, 2.2), powf(color.z, 2.2));
}


/*
__device__ float trace_shadow(float3 ray_e, float3 ray_d, float time)
{
	float3 *pos_ptr = (float3 *)cuScene.position;
	float4 *rot_ptr = (float4 *)cuScene.rotation;
	for (int i = 0; i < cuScene.N; i++) {
		float3 t_ray_d = ray_d;
		float3 t_ray_e = ray_e - pos_ptr[i];
		t_ray_d = quaternionXvector(quaternionConjugate(rot_ptr[i]), t_ray_d);
		t_ray_e = quaternionXvector(quaternionConjugate(rot_ptr[i]), t_ray_e);
		// Intersection test
		float t = sphereIntersectionTest(1, t_ray_d, t_ray_e, i);
		if (t > EPS && t < time) {
			return 0;
		}
	}
	return 1;
}
*/

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
	if (geom < SOLIDS) {
		mate = lerp(diffuse, cue_color, smoothstep(0.9, 0.91, abs(pos.y)));
	} else {
		mate = lerp(diffuse, cue_color, smoothstep(0.9, 0.91, abs(pos.y)) + smoothstep(0.55, 0.56, abs(pos.z)));
	}

	/*
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
	*/
	return mate;
}

#define NSAMPLES 10
#define SHADOW_RAYS 5

__global__
void curandSetupKernel()
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = y * 800 + x;
	curand_init(1578, w, 0, cuConstants.curand + w);
}
__global__
void cudaRayTraceKernel (unsigned char *img)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = y * WIDTH + x;


	__shared__ float mem[400];
	int w2 = threadIdx.y * blockDim.x + threadIdx.x;
	if (w2 < 7 * SPHERES)
		mem[w2] = cuScene.data[w2];
	__syncthreads();
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

				/*
		float di = (x + (sampleX + curand_uniform(cuScene.curand + w)) / NSAMPLES) / cuScene.width * 2 - 1;
		float dj = (y + (sampleY + curand_uniform(cuScene.curand + w)) / NSAMPLES) / cuScene.height * 2 - 1;
		*/
		float di = (x + (sampleX + 0.5) / NSAMPLES) / WIDTH * 2 - 1;
		float dj = (y + (sampleY + 0.5) / NSAMPLES) / HEIGHT * 2 - 1;
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

		for (int i = 0; i < PLANES; i++) {
			int axes = cuConstants.plane_axes[i];
			float c;
			if(axes == 0)
				c = ray_e.x;
			else if (axes == 1)
				c = ray_e.y;
			else
				c = ray_e.z;
			
			float d;
			if(axes == 0)
				d = ray_d.x;
			else if (axes == 1)
				d = ray_d.y;
			else
				d = ray_d.z;
			
			float t = (cuConstants.positions[i] - c) / d;
			if (t > EPS && t < tmin) {
				float3 hit = t * ray_d + ray_e;
				float3 vec = (hit - cuConstants.lower_bounds[i]) * (hit - cuConstants.upper_bounds[i]);
				if (vec.x < 0 && vec.y < 0 && vec.z < 0) {
					IsSpheres = false;
					tmin = t;
					geom = i;
				}
			}
		}
		tmin -= 0.001;
		float3 color = make_float3(0, 0, 0);

		if (geom >= 0) {
			if (IsSpheres) {
				float3 hit = tmin * ray_d + ray_e;
				float3 orig_hit = quaternionXvector(quaternionConjugate(rot_ptr[geom]), hit - pos_ptr[geom]);
				float3 normal;
				normal = orig_hit;
				normal = quaternionXvector(rot_ptr[geom], normal);
				normal = normalize(normal);
				float3 diffuse = cuConstants.sphere_colors[geom];//make_float3(0.20, 0.30, 0.40);//((float3 *)cuScene.diffuse)[material];

				// Direct diffuse
				float3 surface_color = make_float3(0, 0, 0);
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
					float shadow_factor = 1;//trace_shadow(hit, light_dir, 10000.0);
					surface_color += shadow_factor;
				}
				surface_color *= 1 / (SHADOW_RAYS + 0.0) * 2.9;

				float3 m = do_material(geom, diffuse, normal, orig_hit);
				surface_color += 1.5f * clamp(0.3f-0.7f*normal.y,0.0f,1.0f)*make_float3(0.0,0.2,0.0);
				// Rim
				if (geom <= 3)
				surface_color *= 1.0 + 6.0 * m* powf( clamp( 1.0 + dot(normal, ray_d), 0.0f, 1.0f ), 2.0 );

				float fre = 0.04 + 4 * powf(clamp( 1.0 + dot(normal, ray_d), 0.0f, 1.0f ), 5.0f) ;
				float step = 0.0;
				float3 ref = normalize(ray_d - 2 * normal * dot(normal, ray_d));
				if (ref.y > 0)
					step = 1.0;

				if (geom <= 3)
					surface_color += 1.0 * fre * step * doEnvironment(ref);
				color += surface_color * m;
				accumulated_color += color;
			} else {
				accumulated_color += make_float3(0.0, 1.0, 0.0);
				/*
				float3 hit = tmin * ray_d + ray_e;
				if (hit.x > -1 && hit.x < 1 && hit.z > -1 && hit.z < 1) {
					float4 c = tex2D(envmap, hit.x, hit.z);
					accumulated_color += make_float3(c.x, c.y, c.z);
				}
				*/

			}
		} else {
			break;
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
	gpuErrchk(cudaMalloc((void **)&poolConstants.curand, sizeof(curandState) * 800 * 600));
	gpuErrchk(cudaMemcpyToSymbol(cuConstants, &poolConstants, sizeof(PoolConstants)));
	dim3 dimBlock(16, 16);
	dim3 dimGrid(800 / 16, 600 / 16);
	curandSetupKernel<<<dimGrid, dimBlock>>>();
	cudaDeviceSynchronize();
}

void cudaRayTrace(CudaScene *scene, unsigned char *img)
{
	printf("CudaRayTrace\n");
	printf("%p\n", scene);
	gpuErrchk(cudaMemcpyToSymbol(cuScene, scene, sizeof(CudaScene)));

	dim3 dimBlock(16, 16);
	dim3 dimGrid(WIDTH / 16, HEIGHT / 16);

	double startTime = CycleTimer::currentSeconds();
	cudaRayTraceKernel<<<dimGrid, dimBlock>>>(img);
	cudaDeviceSynchronize();
	printf("CUDA rendering time: %lf\n", CycleTimer::currentSeconds() - startTime);

	cudaError_t error = cudaGetLastError();
	if ( cudaSuccess != error )
			printf( "Error: %d\n", error );
}
