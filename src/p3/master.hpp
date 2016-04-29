#pragma once

#include <ctime>
#include <string>
#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <deque>
#include <iostream>

#include "message.hpp"

class Connection
: public std::enable_shared_from_this<Connection>
{
private:
	typedef boost::asio::ip::tcp tcp;
	typedef std::deque<Message*> MessageQueue;

public:
	Connection(tcp::socket socket)
	: socket_(std::move(socket))
	{}

	void start();
private:

	void send(const std::string& str);
	void send(Message* msg);
	void do_read_header();
	void do_read_body();
	void do_write();

	tcp::socket socket_;
	Message read_msg_;
	MessageQueue write_msgs_;
};

class Master
{
private:
	typedef boost::asio::ip::tcp tcp;

public:
	Master(boost::asio::io_service& io_service)
	: acceptor_(io_service, tcp::endpoint(tcp::v4(), 50000)),
	socket_(io_service)
	{
		do_accept();
	}

	static void start();

private:
	void do_accept();

	tcp::acceptor acceptor_;
	tcp::socket socket_;
};
