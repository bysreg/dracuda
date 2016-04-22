#pragma once

#include <boost/asio.hpp>
#include <deque>

#include "message.hpp"

class Slave
{
private:
	typedef boost::asio::ip::tcp tcp;
	typedef std::deque<Message> MessageQueue;

public:

	// create a new thread to run slave tcp 
	static void start(const std::string& host);

	// stop slave tcp 
	static void stop();


	Slave(boost::asio::io_service& io_service, 
		tcp::resolver::iterator endpoint_iterator)
		: io_service(io_service), 
		  socket(io_service)
	{

	}

	// communications
	void send_anjing();

	void send(const Message& message);

private:
	boost::asio::io_service& io_service;
	tcp::socket socket;
	Message read_msg;
	MessageQueue write_msgs;

	void run(boost::asio::ip::tcp::resolver::iterator endpoint_iterator);
	void do_connect(tcp::resolver::iterator endpoint_iterator);
	void do_read_header();
	void do_read_body();
	void do_write();
	void process_message(const Message&);
};