#ifndef CUDAVECTOR_H
#define CUDAVECTOR_H

struct float2 {
	float x, y;
};
struct float3 {
	float x, y, z;
};
struct float4 {
	float x, y, z, w;
};

inline float2 make_float2(float x, float y) {
	float2 a;
	a.x = x;
	a.y = y;
	return a;
}

inline float3 make_float3(float x, float y, float z) {
	float3 a;
	a.x = x;
	a.y = y;
	a.z = z;
	return a;
}
inline float4 make_float4(float x, float y, float z, float w) {
	float4 a;
	a.x = x;
	a.y = y;
	a.z = z;
	a.w = w;
	return a;
}
#endif
