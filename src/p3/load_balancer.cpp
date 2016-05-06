#include "load_balancer.hpp"
#include "slave_info.hpp"
#include "raytracer_application.hpp"

// if the frame count has not reached this number yet
// we always divide equally
static const int WARM_FRAMES = 60;

void LoadBalancer::calc(const RaytracerApplication* app, SlaveInfo* input, double* output, int size) 
{
	static bool warmed_up = false;

	// // always divide it equally at first
	// if(app->cur_frame_number <= WARM_FRAMES) {
		calc_equal(input, output, size);
	// }else{

	// 	if(!warmed_up) {
	// 		warmed_up = true;
	// 		std::cout<<"starting dynamic load balancing ... "<<std::endl;
	// 	}

	// 	// pick one of the techniques	
	// 	calc_naive(input, output, size);
	// }	
}

void LoadBalancer::calc_naive(SlaveInfo* input, double* output, int size) 
{
	//calculate sum of the inverse
	double sum_of_inv = 0;
	for(int i=0;i<size;i++)
	{
		sum_of_inv += (1.0 / input[i].response_duration);
	}

	for(int i=0;i<size;i++)
	{
		output[i] = (1.0 / input[i].response_duration) / sum_of_inv;
	}
}

void LoadBalancer::calc_equal(SlaveInfo* input, double* output, int size)
{
	double w = 1.0 / size;
	for(int i=0;i<size;i++)
	{
		output[i] = w;
	}
}