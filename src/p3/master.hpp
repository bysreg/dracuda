#pragma once

#include <ctime>
#include <string>
#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>

#include "message.hpp"

class Connection 
	: public boost::enable_shared_from_this<Connection> 
{	
private:
	typedef boost::asio::ip::tcp tcp;

public:
	typedef boost::shared_ptr<Connection> pointer;

	static pointer create(boost::asio::io_service& io_service) {
		return pointer(new Connection(io_service));
	}

	tcp::socket& socket()
	{
		return socket_;
	}

private:	
	tcp::socket socket_;
	char* data;

	Connection(boost::asio::io_service& io_service)
		: socket_(io_service) 
	{}

};

class Master
{

private:
	typedef boost::asio::ip::tcp tcp;

public:	

	// FIXME : should be private
	Master(boost::asio::io_service& io_service) 
		: acceptor(io_service, tcp::endpoint(tcp::v4(), 1030)) // 1030 is the port number
	{}	

	// create a new thread to run the master tcp async server
	static void start();

	// stop master tcp async server
	static void stop();

	static const char ACK = 1;

private:		

	tcp::acceptor acceptor;

	void run();
	void start_accept();
	void handle_accept(Connection::pointer new_conn, 
		const boost::system::error_code& error);
	void send_complete(const boost::system::error_code& /*error*/,
      size_t /*bytes_transferred*/);

	// communications
	void send_ack(Connection::pointer);
	void send_helloworld(Connection::pointer);
	
	void send(Connection::pointer, char code);
	void send(Connection::pointer, const std::string& s);

	static std::string make_daytime_string()
	{
	  using namespace std; // For time_t, time and ctime;
	  time_t now = time(0);
	  return ctime(&now);
	}
};
