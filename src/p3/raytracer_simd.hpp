#ifndef SIMD_RAYTRACER_HPP
#define SIMD_RAYTRACER_HPP

#include "cudaScene.hpp"

extern void simdRayTrace(CudaScene *scene, unsigned char *img);

extern void simdInitialize();


#endif
