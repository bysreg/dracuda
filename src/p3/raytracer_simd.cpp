#include <stdio.h>
#include <omp.h>
#include "cudaScene.hpp"
#include "raytracer_cuda.hpp"
#include "helper_math.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cycleTimer.h"
#include "constants.hpp"
#include "math/random462.hpp"
#include <immintrin.h>
#define PI 3.1415926535

#define EPS 0.0001

static PoolConstants cuConstants;// = poolConstants;
static CudaScene cuScene;


inline  static float3 quaternionXvector(float4 q, float3 vec)
{
	float3 qvec = make_float3(q.x, q.y, q.z);
	float3 uv = cross(qvec, vec);
	float3 uuv = cross(qvec, uv);
	uv *= (2.0 * q.w);
	uuv *= 2.0;
	return vec + uv + uuv;
}

inline  static float3 quaternionXCvector(float4 q, float3 vec)
{
	float3 qvec = make_float3(-q.x, -q.y, -q.z);
	float3 uv = cross(qvec, vec);
	float3 uuv = cross(qvec, uv);
	uv *= (2.0 * q.w);
	uuv *= 2.0;
	return vec + uv + uuv;
}

static float distanceToSegment( float2 a, float2 b, float2 p )
{
	float2 pa = p - a;
	float2 ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h );
}

 static float circle2(float2 pos, float2 center, float radius, float dist, float begin, float interval)
{
	float2 diff = pos - center;
	float angle = atan2(diff.y, diff.x) - begin;
	if (angle < 0.0)
		angle = angle + 2.0 * PI;
	float d = sqrt(dot(diff, diff));
	float k = fabs(d - radius) / dist;
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
	
 static float3 do_material (int geom, float3 pos)
{
	//pos = pos - cuScene.ball_position[geom];
	float3 mate = cuConstants.sphere_colors[geom];
	float3 cue_color = make_float3(0.29, 0.27, 0.25);
	if (geom < SOLIDS) {
		mate = lerp(mate, cue_color, smoothstep(0.9, 0.91, fabs(pos.y)));
	} else {
		mate = lerp(mate, cue_color, smoothstep(0.9, 0.91, fabs(pos.y)) + smoothstep(0.55, 0.56, fabs(pos.z)));
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
	d = fminf(fminf(d1, d4), fminf(d2, d3));
	k = fminf(k1, k2);
	mate *= smoothstep(0.04, 0.045, d) * smoothstep(0.88, 1.0, k);
	return mate;
}

float *printBuffer_real;
float *tempBuffer_real;

void simdInitialize()
{
	posix_memalign((void **)&printBuffer_real, 32, sizeof(float) * 12800);
	posix_memalign((void **)&tempBuffer_real, 32, sizeof(float) * 12800);
}


void printVec(__m128 vec)
{
	_mm_store_ps(printBuffer_real, vec);
	for (int i = 0; i < 4; i++)
		printf("V%d: %f ", i, printBuffer_real[i]);
	printf("\n");
}

#define SET _mm_set_ps
#define SET1 _mm_set1_ps
#define ADD _mm_add_ps
#define SUB _mm_sub_ps
#define MUL _mm_mul_ps
#define DIV _mm_div_ps
#define CMP _mm_cmp_ps
#define LT _mm_cmplt_ps
#define GT _mm_cmpgt_ps
#define CMP _mm_cmp_ps
#define BLEND _mm_blendv_ps

#define PLANE_INTERSECT(geo, a0, a1, a2, v0, v1, v2) \
B = SET1(cuConstants.positions[geo] - ray_e.a0); \
A = DIV(B, v0);\
C = MUL(A, v1); B = SET1(ray_e.a1); C = ADD(C, B);\
B = SET1(cuConstants.lower_bounds[geo].a1);\
M1 = GT(C, B);\
B = SET1(cuConstants.upper_bounds[geo].a1);\
M2 = LT(C, B);\
M1 = _mm_and_ps(M1, M2);\
C = MUL(A, v2);\
B = SET1(ray_e.a2);\
C = ADD(C, B);\
B = SET1(cuConstants.lower_bounds[geo].a2);\
M2 = GT(C, B);\
M1 = _mm_and_ps(M1, M2);\
B = SET1(cuConstants.upper_bounds[geo].a2);\
M2 = LT(C, B);\
M1 = _mm_and_ps(M1, M2);\
B = SET1(0.0001); \
M2 = GT(A, B);\
M1 = _mm_and_ps(M1, M2);\
M2 = LT(A, D);\
M1 = _mm_and_ps(M1, M2);\
D = _mm_blendv_ps(D, A, M1);\
B = SET1(geo + 0.0);\
G = _mm_blendv_ps(G, B, M1);\
M0 = _mm_or_ps(M0, M1);\

#define RANDOMIZE  \
for (int q = 0; q < 4; q++) { \
	tempBuffer[q] = random_uniform(); \
} \
R = _mm_load_ps(tempBuffer);\
/*
T = SET1(295.258643906); \
R = MUL(R, T); \
T = SET1(19.9132548); \
R = ADD(R, T); \
T = _mm256_floor_ps(R); \
R = SUB(R, T); 
*/

void simdRayTrace(CudaScene *scene, unsigned char *img)
{
	double startTime = CycleTimer::currentSeconds();
	cuScene = *scene;
	cuConstants = poolConstants;
	int tid = 0;
	float3 ray_e = cuScene.cam_position;
#ifdef MTHREAD
	#pragma omp parallel 
#endif
	{
	__m128 A, B, C, D, E, S, S0, G, V0x, V0y, V0z, V1x, V1y, V1z, C0x, C0y, C0z, M0, M1, M2, M3, S1;
	__m128 C1x, C1y, C1z;
	__m128 C2x, C2y, C2z;
	__m128 V2x, V2y, V2z;
	__m128 V3x, V3y, V3z;
	__m128 V4x, V4y, V4z;
	__m128 R;
#ifdef MTHREAD
	tid = omp_get_thread_num();
#else
	tid = 0;
#endif
	float *printBuffer = printBuffer_real + 128 * tid;
	float *tempBuffer = tempBuffer_real + 128 * tid;
	R = _mm_set_ps(random_uniform(), random_uniform(), random_uniform(),
			random_uniform());
#ifdef MTHREAD
	#pragma omp for private(tid) schedule(dynamic)
#endif
	for (int y = cuScene.y0; y < cuScene.render_height; y++) {
		for (int x0 = 0; x0 < WIDTH; x0 += 4) {
			int w = (y  - cuScene.y0)* WIDTH + x0;
			C2x = SET1(0); C2y = SET1(0); C2z = SET1(0);
			for (int sampleX = 0; sampleX < NSAMPLES; sampleX++)
			for (int sampleY = 0; sampleY < NSAMPLES; sampleY++) {
			//printf("%d\n", tid);
			V0x = SET1(cuScene.dir.x);
			V0y = SET1(cuScene.dir.y);
			V0z = SET1(cuScene.dir.z);
			D = SET1(1.0 / NSAMPLES);
			A = SET(3.5, 2.5, 1.5, 0.5);
			B = SET1((sampleX + 0.0) / NSAMPLES);
			A = ADD(A, B);
			B = SET1(x0);
			A = ADD(A, B);
			B = SET1(2.0f / WIDTH);
			A = MUL(A, B);
			B = SET1(1.0);
			A = SUB(A, B); // di (mul ARcR)
			B = SET1(cuScene.ARcR.x); C = MUL(A, B); V0x = ADD(V0x, C);
			B = SET1(cuScene.ARcR.y); C = MUL(A, B); V0y = ADD(V0y, C);
			B = SET1(cuScene.ARcR.z); C = MUL(A, B); V0z = ADD(V0z, C); // V0 = dir + dj * cU + di * ARcR;

			A = SET1((y + (sampleY + 0.5) / NSAMPLES) * 2.0 / float(HEIGHT) - 1.0);
			
			B = SET1(cuScene.cU.x); C = MUL(A, B); V0x = ADD(V0x, C);
			B = SET1(cuScene.cU.y); C = MUL(A, B); V0y = ADD(V0y, C);
			B = SET1(cuScene.cU.z); C = MUL(A, B); V0z = ADD(V0z, C); // V0 = dir + dj * cU
			// Normalize
			A = MUL(V0x, V0x); B = MUL(V0y, V0y); A = ADD(A, B); B = MUL(V0z, V0z); A = ADD(A, B);
			A = _mm_sqrt_ps(A);
			C = SET1(1.0);
			A = _mm_div_ps(C, A);
			V0x = MUL(V0x, A); V0y = MUL(V0y, A); V0z = MUL(V0z, A); // ray_d
			G = _mm_set1_ps(-1.0);
			int isSpheres, isPlanes;
			D = SET1(10000.0); // tmin
			for (int i = 0; i < SPHERES; i++) {
				float3 t_ray_e = cuScene.cam_position - cuScene.ball_position[i];
				float SC = dot(t_ray_e, t_ray_e) - 1;
				B = SET1(t_ray_e.x); A = MUL(V0x, B);
				B = SET1(t_ray_e.y); C = MUL(V0y, B); A = ADD(A, C);
				B = SET1(t_ray_e.z); C = MUL(V0z, B); A = ADD(A, C); // A = dot(ray_d, t_ray_e)
				C = A;
				B = SET1(0);
				C = SUB(B, C); // C = -B
				A = MUL(A, A); // A = B2
				B = SET1(SC);
				A = SUB(A, B); // A = B2 - C
				A = _mm_sqrt_ps(A); // A = sqrt(B2 - C)
				C = SUB(C, A); // C = -B - sqrt(B2 - C) == t
				B = SET1(EPS);
				M0 = GT(C, B); // M0 = A > B
				M1 = LT(C, D); // M0 = A > B
				M0 = _mm_and_ps(M0, M1);
				A = SET1(i);
				G = _mm_blendv_ps(G, A, M0); // G is geom
				D = _mm_blendv_ps(D, C, M0); // D is tmin
			}
			A = SET1(9999.0);
			M0 = LT(D, A);
			M3 = M0;
			isSpheres = _mm_movemask_ps(M0);
			isSpheres &= 255;
			M0 = SET1(0);
			PLANE_INTERSECT(0, y, x, z, V0y, V0x, V0z);
			PLANE_INTERSECT(1, y, x, z, V0y, V0x, V0z);
			PLANE_INTERSECT(2, y, x, z, V0y, V0x, V0z);
			PLANE_INTERSECT(3, y, x, z, V0y, V0x, V0z);
			PLANE_INTERSECT(4, y, x, z, V0y, V0x, V0z);
			PLANE_INTERSECT(5, x, y, z, V0x, V0y, V0z);
			PLANE_INTERSECT(6, x, y, z, V0x, V0y, V0z);
			PLANE_INTERSECT(7, z, x, y, V0z, V0x, V0y);
			PLANE_INTERSECT(8, z, x, y, V0z, V0x, V0y);
			isPlanes = _mm_movemask_ps(M0);
			isPlanes &= 255;
			isSpheres &= ~isPlanes;
			M3 = _mm_andnot_ps(M0, M3);
			V1x = MUL(D, V0x);
			V1y = MUL(D, V0y);
			V1z = MUL(D, V0z);
			B = SET1(ray_e.x); V1x = ADD(V1x, B);
			B = SET1(ray_e.y); V1y = ADD(V1y, B);
			B = SET1(ray_e.z); V1z = ADD(V1z, B);
			V4x = V1x;
			V4y = V1y;
			V4z = V1z;
			_mm_store_ps(tempBuffer, V1x);
			_mm_store_ps(tempBuffer + 8, V1y);
			_mm_store_ps(tempBuffer + 16, V1z);
			_mm_store_ps(printBuffer, G);

			// Colors
			for (int i = 0; i < 8; i++) {
				int geom = printBuffer[i];
				if ((isSpheres >> i) & 1) {
					float3 hit = normalize(make_float3(tempBuffer[i], tempBuffer[8 + i], tempBuffer[16 + i]) - cuScene.ball_position[geom]);
					float3 m = do_material(geom, quaternionXCvector(cuScene.ball_orientation[geom], hit));
					//float3 m = cuConstants.sphere_colors[geom];
					tempBuffer[i] = hit.x;
					tempBuffer[8 + i] = hit.y;
					tempBuffer[16 + i] = hit.z;
					tempBuffer[24 + i] = m.x;
					tempBuffer[32 + i] = m.y;
					tempBuffer[40 + i] = m.z;
				} else if ((isPlanes >> i) & 1) {
					tempBuffer[i] = cuConstants.normals[geom].x;
					tempBuffer[8 + i] = cuConstants.normals[geom].y;
					tempBuffer[16 + i] = cuConstants.normals[geom].z;
					tempBuffer[24 + i] = cuConstants.plane_colors[geom].x;
					tempBuffer[32 + i] = cuConstants.plane_colors[geom].y;
					tempBuffer[40 + i] = cuConstants.plane_colors[geom].z;
				} else {
					tempBuffer[24 + i] = 0.7;
					tempBuffer[32 + i] = 0.8;
					tempBuffer[40 + i] = 1.0;
				}
			}	
			V1x = _mm_load_ps(tempBuffer);
			V1y = _mm_load_ps(tempBuffer + 8);
			V1z = _mm_load_ps(tempBuffer + 16); // Normals
			C0x = _mm_load_ps(tempBuffer + 24);
			C0y = _mm_load_ps(tempBuffer + 32);
			C0z = _mm_load_ps(tempBuffer + 40);
			D = SET1(0.0); // shadow_factor
			S0 = SET1(0.0);
			for (int i = 0; i < SHADOW_RAYS; i++) {
				B = SET1(0.0);
				C = SUB(B, V1x);
				C = _mm_max_ps(V1x, C);
				//printVec(C);
				B = SET1(0.5);
				M0 = LT(C, B); // C abs(normal.x)
				A = SET1(0); V3x = BLEND(V1z, A, M0);
				V3x = SUB(A, V3x); // x = 0 / -z;
				A = SET1(0); V3y = BLEND(A, V1z, M0); // y = z / 0
				A = SUB(A, V1y); V3z = BLEND(V1x, A, M0); // z = -y / x V3 = sdir
				// Cross product
				A = MUL(V1y, V3z); B = MUL(V1z, V3y); V2x = SUB(A, B);
				A = MUL(V1z, V3x); B = MUL(V1x, V3z); V2y = SUB(A, B);
				A = MUL(V1x, V3y); B = MUL(V1y, V3x); V2z = SUB(A, B); // V2 = tdir
				RANDOMIZE;
				B = SET1(3.1415926535 * 2);
				C = MUL(R, B);
				// SIN
				_mm_store_ps(tempBuffer, C);
				for (int k = 0; k < 4; k++) {
					tempBuffer[8 + k] = cosf(tempBuffer[k]);
					tempBuffer[k] = sinf(tempBuffer[k]);
				}
				A = _mm_load_ps(tempBuffer); // A = sin(2pi * r);
				V2x = MUL(V2x, A); V2y = MUL(V2y, A); V2z = MUL(V2z, A); // tdir * sin(R)
				A = _mm_load_ps(tempBuffer + 8); // A = cos(2pi * r);
				V3x = MUL(V3x, A); V3y = MUL(V3y, A); V3z = MUL(V3z, A); // sdir * cos(R)
				V2x = ADD(V2x, V3x); V2y = ADD(V2y, V3y); V2z = ADD(V2z, V3z); //tdir * sinr + sdir * cosr
				RANDOMIZE;
				A = _mm_sqrt_ps(R);
				V2x = MUL(V2x, A); V2y = MUL(V2y, A); V2z = MUL(V2z, A);
				B = SET1(1.0); A = SUB(B, R); A = _mm_sqrt_ps(A); // A = sqrt(1 - u);
				B = MUL(A, V1x); V2x = ADD(V2x, B);
				B = MUL(A, V1y); V2y = ADD(V2y, B);
				B = MUL(A, V1z); V2z = ADD(V2z, B); //V2 = light_dir
				S = SET1(1.0);
				// Shadow
				for (int j = 0; j < SPHERES; j++) {
					B = SET1(cuScene.ball_position[j].x); V3x = SUB(V4x, B);
					B = SET1(cuScene.ball_position[j].y); V3y = SUB(V4y, B);
					B = SET1(cuScene.ball_position[j].z); V3z = SUB(V4z, B); // V3 = V4 - ballpos = ray_e

					//float3 t_ray_e = ray_e - cuScene.ball_position[j];
					A = MUL(V2x, V3x);
					C = MUL(V2y, V3y); A = ADD(A, C);
					C = MUL(V2z, V3z); A = ADD(A, C); // A = dot(ray_d, ray_e);
					B = SET1(0); A = SUB(B, A); // A = -dot(ray_d, ray_e);
					C = MUL(A, A); // C = b * b;
					B = MUL(V3x, V3x); E = MUL(V3y, V3y);
					B = ADD(B, E); // B = dot(ray_e, ray_e);
					E = MUL(V3z, V3z); B = ADD(B, E); B = SUB(B, C); // B = dot - b * b;
					B = _mm_sqrt_ps(B); C = SET1(1.0);
					B = SUB(B, C); // B = h
					C = SET1(16.0); B = MUL(B, C); B = DIV(B, A); // B = res
					E = SET1(0.0);
					M0 = GT(A, E); M1 = LT(B, E);
					B = BLEND(B, E, M1);

					M1 = LT(B, S);
					M0 = _mm_and_ps(M0, M1);
					S = BLEND(S, B, M0);
				}
				S0 = ADD(S0, S);
			}
			B = SET1(1.0 / SHADOW_RAYS);
			S0 = MUL(S0, B);
			B = SET1(2.9);
			S1 = MUL(S0, B);
			C1x = MUL(S1, C0x);
			C1z = MUL(S1, C0z);
			B = SET1(-0.7);
			A = MUL(V1y, B); //A = normal.y * -0.7
			B = SET1(0.3);
			A = ADD(A, B); // A = normal.y * -0.7 + 0.3
			B = SET1(0.0);
			A = _mm_max_ps(A, B);
			B = SET1(1.0);
			A = _mm_min_ps(A, B); // A = clamp
			B = SET1(0.3);
			A = MUL(A, B); // Green V
			S1 = ADD(A, S1);
			C1y = MUL(S1, C0y);
			// Ref ray
			A = MUL(V0x, V1x); B = MUL(V0y, V1y); A = ADD(A, B); B = MUL(V0z, V1z); A = ADD(A, B);
			B = SET1(-2.0);
			A = MUL(A, B); //A = -2 * dot(normal, ray_d);
			V3x = MUL(V1x, A); V3y = MUL(V1y, A); V3z = MUL(V1z, A);
			V3x = ADD(V3x, V1x); V3y = ADD(V3y, V1y); V3z = ADD(V3z, V1z); // V3 = Ref ray
			A = MUL(V3x, V3x); B = MUL(V3y, V3y); A = ADD(A, B); B = MUL(V3z, V3z); A = ADD(A, B);
			A = _mm_sqrt_ps(A); C = SET1(1.0); A = _mm_div_ps(C, A);
			V3x = MUL(V3x, A); V3y = MUL(V3y, A); V3z = MUL(V3z, A); // ray_d

			// Specular Highlight
			B = SET1(0.479632); A = MUL(B, V3x);
			B = SET1(0.685189); C = MUL(B, V3y); A = ADD(A, C);
			B = SET1(-0.548151); C = MUL(B, V3z); A = ADD(A, C); // A = dot
			B = SET1(0); A = _mm_max_ps(A, B); B = SET1(1); A = _mm_min_ps(A, B);

			A = MUL(A, A); A = MUL(A, A); A = MUL(A, A); A = MUL(A, A); A = MUL(A, A); A = MUL(A, A);
			B = SET1(0.3); A = MUL(A, B);

			C1x = ADD(C1x, A); C1y = ADD(C1y, A); C1z = ADD(C1z, A);

			C0x = MUL(C0x, S0); C0y = MUL(C0y, S0); C0z = MUL(C0z, S0);
			C0x = BLEND(C0x, C1x, M3); C0y = BLEND(C0y, C1y, M3); C0z = BLEND(C0z, C1z, M3);

			B = SET1(0.0); M0 = LT(G, B);
			C2x = ADD(C2x, C0x); C2y = ADD(C2y, C0y); C2z = ADD(C2z, C0z);
		} // SAMPLES
			C2x = _mm_sqrt_ps(C2x);
			C2y = _mm_sqrt_ps(C2y);
			C2z = _mm_sqrt_ps(C2z);
			B = SET1(1.0 / NSAMPLES);
			C2x = MUL(C2x, B);
			C2y = MUL(C2y, B);
			C2z = MUL(C2z, B);

			B = SET1(1.0);
			M2 = GT(C2x, B); C2x = BLEND(C2x, B, M2); //CLAMP
			M2 = GT(C2y, B); C2y = BLEND(C2y, B, M2);
			M2 = GT(C2z, B); C2z = BLEND(C2z, B, M2);
			_mm_store_ps(tempBuffer, C2x);
			_mm_store_ps(tempBuffer + 8, C2y);
			_mm_store_ps(tempBuffer + 16, C2z);
			for (int i = 0; i < 4; i++) {
				img[3 * w + 3 * i] = 255 * tempBuffer[i];
				img[3 * w + 3 * i + 1] = 255 * tempBuffer[8 + i];
				img[3 * w + 3 * i + 2] = 255 * tempBuffer[16 + i];
			}
		} // x = 0 -> WIDTH
	} // y = 0 -> HEIGHT
	}
	printf("SIMD rendering time: %lf\n", CycleTimer::currentSeconds() - startTime);
}
