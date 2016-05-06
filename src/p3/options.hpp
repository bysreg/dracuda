#pragma once

#include <string>

struct Options
{
	bool master;
	bool slave;
	// for master:
	int min_slave_to_start = 0;

	// for slave:
	// host to connect from slave
	std::string host;
};