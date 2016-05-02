#pragma once

#include <boost/asio.hpp>
#include <deque>

#include "message.hpp"

class Slave
{
private:
	typedef boost::asio::ip::tcp tcp;
	typedef std::deque<MessagePtr> MessageQueue;

public:

	static int read_msg_max_length;

	// create a new thread to run slave tcp 
	static Slave& start(const std::string& host);

	// stop slave tcp 
	// FIXME : not completed yet
	static void stop();

	Slave(boost::asio::io_service& io_service, 
		tcp::resolver::iterator endpoint_iterator);	

	template<typename T>
	void send(const T& value) {
		MessagePtr msg =  std::make_shared<Message>(sizeof(T));

		msg->set_body_length(sizeof(T));
		std::memcpy(msg->body(), &value, sizeof(T));
		msg->encode_header();
		send(msg);	
	}

	void send(const unsigned char*, int size);
	void send(const std::string& str);
	void send(MessagePtr message);

	void run();

	// callbacks
	void set_on_message_received(std::function<void(const Message&)> const& cb);

private:
	boost::asio::io_service& io_service;
	tcp::socket socket;
	Message read_msg;
	MessageQueue write_msgs;
	tcp::resolver::iterator endpoint_iterator;

	// callbacks
	std::function<void(const Message&)> on_message_received;
	
	void do_connect(tcp::resolver::iterator endpoint_iterator);
	void do_read_header();
	void do_read_body();
	void do_write();
	void process_message(const Message&);
};