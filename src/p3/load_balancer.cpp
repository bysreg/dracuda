#include "load_balancer.hpp"
#include "slave_info.hpp"
#include "raytracer_application.hpp"
#include "constants.hpp"

#include <iostream>
#include <algorithm>

// if the frame count has not reached this number yet
// we always divide equally
static const int WARM_FRAMES = 60;

void LoadBalancer::calc(const RaytracerApplication* app, SlaveInfo* input, double* output, int size) 
{
	static bool warmed_up = false;

	// always divide it equally at first
	if(app->cur_render_frame_number <= WARM_FRAMES) {
		calc_equal(input, output, size);
	}else{

		if(!warmed_up) {
			warmed_up = true;
			std::cout<<"starting dynamic load balancing ... "<<std::endl;
		}

		// pick one of the techniques	
		// calc_equal(input, output, size);
		// calc_naive(input, output, size);
		calc_ab(input, output, size, HEIGHT);
	}	

	// std::cout<<"weight : ";
	// for(int i=0;i<size;i++) 
	// {
	// 	std::cout<<output[i]<<" ";
	// }
	// std::cout<<std::endl;

	//test
	// SlaveInfo test_input[3];
	// test_input[2].sum_network_latency = 1;
	// test_input[2].sum_rendering_factor= 2;
	// test_input[1].sum_network_latency = 3;
	// test_input[1].sum_rendering_factor= 4;
	// test_input[0].sum_network_latency = 6;
	// test_input[0].sum_rendering_factor= 1;
	// for(int i=0;i<3;i++) {
	// 	test_input[i].messages_received = 1;
	// }
	// double* test_output = new double[3];
	// calc_ab(test_input, test_output, 3, 2);

	// std::cout<<"test calc ab : ";
	// for(int i=0;i<3;i++) {
	// 	std::cout<<test_output[i] << " ";
	// }
	// std::cout<<std::endl;
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

// sort using a custom function object
struct {
    bool operator()(const SlaveInfo& a, const SlaveInfo& b)
    {   
        return a.get_avg_network_latency() < b.get_avg_network_latency();
    }   
} compare_slave;

void LoadBalancer::calc_ab(SlaveInfo* input, double* output, int size, double total_workload)
{
	double cur_workload = 0;
	double sum_inv_b = 0;
	int last_slave_idx = 0;;
	double balanced_resp_time = 0;

	static SlaveInfo copy[MAX_SLAVE];
	for(int i=0;i<size;i++) {
		copy[i].messages_received = input[i].messages_received;
		copy[i].sum_network_latency = input[i].sum_network_latency;
		copy[i].sum_rendering_factor = input[i].sum_rendering_factor;
	}

	// sort copy based on average network latency
	std::sort(copy, copy + size, compare_slave);

	for(int i=1;i<size;i++)
	{
		double diff_a =  copy[i].get_avg_network_latency() - copy[i-1].get_avg_network_latency();
		sum_inv_b += (1 / copy[i-1].get_avg_rendering_factor());
		
		cur_workload += diff_a * sum_inv_b;
		
		// std::cout<<"cur workload "<<cur_workload<<std::endl;

		if(cur_workload >= total_workload){
			last_slave_idx = i;
			break;
		}
	}

	if(cur_workload > total_workload) {
		// std::cout<<"aaa"<<std::endl;

		double diff_workload = cur_workload - total_workload;
		double diff_response_time = diff_workload / sum_inv_b;
		balanced_resp_time = copy[last_slave_idx].get_avg_network_latency() 
								- diff_response_time; 
	}else if(cur_workload < total_workload) {
		// std::cout<<"bbb"<<std::endl;

		double diff_workload = total_workload - cur_workload;

		// add the last inv_b to the sum_inv_b
		sum_inv_b += (1 / copy[size-1].get_avg_rendering_factor());

		double diff_response_time = diff_workload / sum_inv_b;
		balanced_resp_time = copy[size-1].get_avg_network_latency() 
								+ diff_response_time; 
	}else{
		// std::cout<<"ccc"<<std::endl;
		balanced_resp_time = copy[size-1].get_avg_network_latency();
	}

	// calculate the workload for each slave
	for(int i=0;i<size;i++)
	{
		double delt_y = balanced_resp_time - copy[i].get_avg_network_latency();

		if(delt_y <= 0)
			output[i] = 0;
		else
			output[i] = (delt_y / copy[i].get_avg_rendering_factor()) / total_workload;

	}
}

