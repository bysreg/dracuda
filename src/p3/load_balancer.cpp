#include "load_balancer.hpp"
#include "slave_info.hpp"

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