#ifndef CUDA_RAYTRACER_HPP
#define CUDA_RAYTRACER_HPP

#include "cudaScene.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

extern void cudaRayTrace(cudaScene *scene, unsigned char *img);
extern void cudaInitialize();

void bindEnvmap (cudaArray *array, cudaChannelFormatDesc &channelDesc);




#endif
