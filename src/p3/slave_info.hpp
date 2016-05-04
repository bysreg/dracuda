struct SlaveInfo
{
	int y0;

	// timestamp where we start to send message to the slave
	double send_time; 
	
	// time it takes from send_time until the time
	// we get back the response from slave (in seconds)
	double response_duration;
};