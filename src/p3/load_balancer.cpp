#include "load_balancer.hpp"
#include "slave_info.hpp"
#include "raytracer_application.hpp"
#include "constants.hpp"

#include <iostream>

// if the frame count has not reached this number yet
// we always divide equally
static const int WARM_FRAMES = 60;

void LoadBalancer::calc(const RaytracerApplication* app, SlaveInfo* input, double* output, int size) 
{
	static bool warmed_up = false;

	// always divide it equally at first
	if(app->cur_frame_number <= WARM_FRAMES) {
		calc_equal(input, output, size);
	}else{

		if(!warmed_up) {
			warmed_up = true;
			std::cout<<"starting dynamic load balancing ... "<<std::endl;
		}

		// pick one of the techniques	
		calc_equal(input, output, size);
		// calc_naive(input, output, size);
		//calc_ab(input, output, size);
	}	

	std::cout<<"weight : ";
	for(int i=0;i<size;i++) 
	{
		std::cout<<output[i]<<" ";
	}
	std::cout<<std::endl;
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

void LoadBalancer::calc_ab(SlaveInfo* input, double* output, int size)
{
	double cur_workload = 0;
	double total_workload = HEIGHT;
	double sum_inv_b = 0;
	int last_slave_idx = 0;;
	double balanced_resp_time = 0;

	for(int i=1;i<size;i++)
	{
		double diff_a =  input[i].get_avg_network_latency() - input[i-1].get_avg_network_latency();
		sum_inv_b += input[i-1].get_avg_rendering_factor();
		
		cur_workload == diff_a / sum_inv_b;
		
		if(cur_workload >= total_workload){
			last_slave_idx = i;
			break;
		}
	}

	if(cur_workload > total_workload) {
		double diff_workload = cur_workload - total_workload;
		double diff_response_time = diff_workload / (sum_inv_b);
		balanced_resp_time = input[last_slave_idx].get_avg_network_latency() 
								+ diff_response_time; 
	}else if(cur_workload < total_workload) {
		double diff_workload = total_workload - cur_workload;

		// add the last inv_b to the sum_inv_b
		sum_inv_b += input[size-1].get_avg_rendering_factor();

		double diff_response_time = diff_workload / sum_inv_b;
		balanced_resp_time = input[size-1].get_avg_network_latency() 
								+ diff_response_time; 
	}else{
		balanced_resp_time = input[size-1].get_avg_network_latency();
	}

	// calculate the workload for each slave
	for(int i=0;i<size;i++)
	{
		output[i] = ((balanced_resp_time - input[i].get_avg_network_latency()) 
						/ input[i].get_avg_rendering_factor()) / total_workload;

	}
}

