#pragma once

class SlaveInfo;

class LoadBalancer
{
	public:

	// receives input with varying numbers
	// this function will populate the output 
	// with the weights so that , an input with a high number
	// will get less weight
	// input is an array of response time
	static void calc_naive(SlaveInfo* input, double* output, int size);

	// no matter the input, will populate the output 
	// with equal weight for all elements
	static void calc_equal(SlaveInfo* input, double* output, int size);

	private:
	LoadBalancer();	
};