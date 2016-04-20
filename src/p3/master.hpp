#pragma once

#include <boost/asio.hpp>

#include "message.hpp"

class Master
{

public:

	static boost::asio::io_service io_service;

	Master()
	{

	}

	// create a new thread to run the master tcp async server
	void run();

private:


};
