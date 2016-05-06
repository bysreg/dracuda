struct SlaveInfo
{
	int y0;

	int render_height;

	// timestamp where we start to send message to the slave
	double send_time; 
	
	// time it takes from send_time until the time
	// we get back the response from slave (in seconds)
	double response_duration;

	// time it takes to send and receive the message
	// excluding the time it takes to create the message
	// which is the rendering_latency
	double network_latency;

	double rendering_latency;
};