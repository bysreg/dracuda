#pragma once

#include <boost/asio.hpp>
#include <deque>

#include "message.hpp"

class Slave
{
private:
	typedef boost::asio::ip::tcp tcp;
	typedef std::deque<Message*> MessageQueue;

public:

	// create a new thread to run slave tcp 
	static Slave& start(const std::string& host);

	// stop slave tcp 
	static void stop();


	Slave(boost::asio::io_service& io_service, 
		tcp::resolver::iterator endpoint_iterator);	

	template<typename T>
	void send(const T& value) {
		Message* msg = new Message(sizeof(T));

		msg->set_body_length(sizeof(T));
		std::memcpy(msg->body(), &value, sizeof(T));
		msg->encode_header();
		send(msg);
	}

	// communications
	void send(const std::string& str);
	void send(Message* message);

	// callbacks
	void set_on_message_received(std::function<void(const Message&)> const& cb);

private:
	boost::asio::io_service& io_service;
	tcp::socket socket;
	Message read_msg;
	MessageQueue write_msgs;

	// callbacks
	std::function<void(const Message&)> on_message_received;

	void run(boost::asio::ip::tcp::resolver::iterator endpoint_iterator);
	void do_connect(tcp::resolver::iterator endpoint_iterator);
	void do_read_header();
	void do_read_body();
	void do_write();
	void process_message(const Message&);
};