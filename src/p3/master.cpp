#include <boost/thread/thread.hpp>
#include <iostream>

#include "master.hpp"
#include "message.hpp"

using namespace std;

void Master::start() 
{
	using boost::asio::ip::tcp;

	static boost::asio::io_service io_service;
	static Master master(io_service);

	cout<<"starting master server..."<<endl;

	boost::thread t(boost::bind(&Master::run, &master));
}

void Master::stop() 
{
	// FIXME: should stop all the running threads
}

void Master::run() 
{
	start_accept();

	acceptor.get_io_service().run();
}

void Master::start_accept() 
{
	Connection::pointer new_conn = 
		Connection::create(acceptor.get_io_service());

	acceptor.async_accept(new_conn->socket(), 
		boost::bind(&Master::handle_accept, this, new_conn,
			boost::asio::placeholders::error));
}

void Master::handle_accept(Connection::pointer new_conn, 
	const boost::system::error_code& error) 
{
	if(error) {
		cerr << "error!! " << error << endl;		
	}else{
		// for now, always send ACK
		//send_ack(new_conn);
		send_helloworld(new_conn);
	}

	start_accept();
}

void Master::send_complete(const boost::system::error_code& /*error*/,
	  size_t /*bytes_transferred*/)
{	
} 

void Master::send_helloworld(Connection::pointer connection)
{	
	send(connection, "2helloworld!!\n");
}

void Master::send_ack(Connection::pointer connection)
{
	send(connection, static_cast<char>(0));	
}

void Master::send(Connection::pointer connection, char code)
{
	char buf[1];
	buf[1] = code;
	boost::asio::async_write(connection->socket(), 
		boost::asio::buffer(buf), 
		boost::bind(&Master::send_complete, 
			this,
			boost::asio::placeholders::error,
			boost::asio::placeholders::bytes_transferred));
}

void Master::send(Connection::pointer connection, const string& str)
{
	Message msg;
	msg.set_body_length(str.length());
	std::memcpy(msg.body(), str.c_str(), str.length());
	msg.encode_header();

	send(connection, msg);	
}

void Master::send(Connection::pointer connection, const Message& msg)
{
	boost::asio::async_write(connection->socket(), 
		boost::asio::buffer(msg.data(), msg.length()), 
		boost::bind(&Master::send_complete, 
			this,
			boost::asio::placeholders::error,
			boost::asio::placeholders::bytes_transferred));
}