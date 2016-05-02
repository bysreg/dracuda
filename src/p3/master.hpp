#pragma once

#include <ctime>
#include <string>
#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <deque>
#include <iostream>
#include <vector>

#include "message.hpp"

class Master;

class Connection
: public std::enable_shared_from_this<Connection>
{
private:
	typedef boost::asio::ip::tcp tcp;
	typedef std::deque<MessagePtr> MessageQueue;

public:
	Connection(tcp::socket socket, Master& master);	

	void start();

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
	void send(MessagePtr msg); // connection will responsible for t

	int idx;

private:

	void do_read_header();
	void do_read_body();
	void do_write();

	tcp::socket socket;
	Message read_msg;
	MessageQueue write_msgs;
	Master& master;
};

typedef std::shared_ptr<Connection> ConnectionPtr;

class Master
{

	friend class Connection;

private:
	typedef boost::asio::ip::tcp tcp;

public:	

	static int read_msg_max_length;

	static Master& start();	

	template<typename T>
	void send_all(const T& value) {
		MessagePtr msg =  std::make_shared<Message>(sizeof(T));

		msg->set_body_length(sizeof(T));
		std::memcpy(msg->body(), &value, sizeof(T));
		msg->encode_header();
		send_all(msg);	
	}

	void send_all(const std::string& str);
	void send_all(MessagePtr msg);

	template<typename T>
	void send(int conn_idx, const T& value) {
		MessagePtr msg =  std::make_shared<Message>(sizeof(T));

		msg->set_body_length(sizeof(T));
		std::memcpy(msg->body(), &value, sizeof(T));
		msg->encode_header();
		send(conn_idx, msg);			
	}
	void send(int conn_idx, MessagePtr msg);

	int get_connections_count() const;

	// callbacks
	void set_on_message_received(std::function<void(int conn_idx, const Message&)> const& cb);
	void set_on_connection_started(std::function<void(Connection&)> const& cb);

private:

	Master(boost::asio::io_service& io_service);

	// prevent from copying
	Master(Master const& other) = delete;
	void operator=(Master const& other) = delete;

	// callbacks
	std::function<void(int conn_idx, const Message&)> on_message_received;
	std::function<void(Connection&)> on_connection_started;

	void do_accept();

	tcp::acceptor acceptor;
	tcp::socket socket;

	std::vector<ConnectionPtr> connections;
};
