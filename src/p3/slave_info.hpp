struct SlaveInfo
{
	int y0;

	// the workload
	int render_height;

	// how many messages have we received 
	// so far from this slave
	int messages_received;

	// timestamp where we start to send message to the slave
	double send_time; 
	
	// time it takes from send_time until the time
	// we get back the response from slave (in seconds)
	// This value is taken from last frame
	double response_duration;

	// time it takes to send and receive the message
	// excluding the time it takes to create the message
	// which is the rendering_latency
	// This value is taken from last frame
	double network_latency;

	// This value is taken from last frame
	double rendering_latency;

	// this value is taken from last frame
	double rendering_factor;

	double sum_network_latency;

	double sum_rendering_factor;

	inline double get_avg_network_latency() const
	{
		if(messages_received == 0)
			return 0;
		return sum_network_latency / messages_received;
	}

	inline double get_avg_rendering_factor() const
	{
		if(messages_received == 0)
			return 0;
		return sum_rendering_factor / messages_received;
	}
};