#include <stdio.h>
#include "cudaScene.hpp"
#include "raytracer_cuda.hpp"
#include "helper_math.h"
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

__constant__ cudaScene cuScene;


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
	float di = (x + 0.5) / cuScene.width * 2 - 1;
	float dj = (y + 0.5) / cuScene.height * 2 - 1;
	float3 ray_d = normalize(dir + dist * (dj * cU + di * AR * cR));
	float3 ray_e = *((float3 *) cuScene.cam_position);

	float3 *pos_ptr = (float3 *)cuScene.position;
	for (int i = 0; i < cuScene.N; i++) {
		float3 t_ray_d = ray_d;
		float3 t_ray_e = ray_e - pos_ptr[i];
		float A = dot(t_ray_d, t_ray_d);
		float B = dot(2 * t_ray_d, t_ray_e);
		float C = dot(t_ray_e, t_ray_e) - cuScene.radius[i] * cuScene.radius[i];
		float B24AC = B * B - 4 * A * C;
		if (B24AC >= 0) {
			float SB24AC = sqrt(B24AC);
			float x1 = (-B + SB24AC) / 2 * A;
			float x2 = (-B - SB24AC) / 2 * A;
			if (x1 > EPS || x2 > EPS) {
				img[4 * w + 0] = 0;
				break;
			}
		}
	}
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
