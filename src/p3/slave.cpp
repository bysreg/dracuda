#include "slave.hpp"

#include <boost/thread/thread.hpp>
#include <boost/asio.hpp>
#include <string>
#include <cstdlib>
#include <boost/lexical_cast.hpp>

void Slave::start(const std::string& host) 
{
	using boost::asio::ip::tcp;

	static boost::asio::io_service io_service;

	std::cout<<"starting slave server..."<<std::endl;	

	tcp::resolver resolver(io_service);
	auto endpoint_iterator = resolver.resolve({ host, boost::lexical_cast<std::string>(1030)}); // 1030 is the port number
	static Slave slave(io_service, endpoint_iterator);	

	boost::thread t(boost::bind(&Slave::run, &slave, endpoint_iterator));
}

void Slave::stop()
{
}

void Slave::run(boost::asio::ip::tcp::resolver::iterator endpoint_iterator)
{
	do_connect(endpoint_iterator);

	io_service.run();
}

void Slave::do_connect(tcp::resolver::iterator endpoint_iterator)
{
	// when we first receive data, we check for the header first
	boost::asio::async_connect(socket, endpoint_iterator,
		[this](boost::system::error_code ec, tcp::resolver::iterator)
		{
			if (!ec)
			{
				do_read_header();
			}
		});
}

void Slave::do_read_header()
{
	boost::asio::async_read(socket,
	boost::asio::buffer(read_msg.data(), Message::header_length),
	[this](boost::system::error_code ec, std::size_t /*length*/)
	{
		if (!ec && read_msg.decode_header())
		{
			do_read_body();
		}
		else
		{
			socket.close();
		}
	});
}

void Slave::do_read_body()
{
	boost::asio::async_read(socket,
	boost::asio::buffer(read_msg.body(), read_msg.body_length()),
	[this](boost::system::error_code ec, std::size_t /*length*/)
	{
		if (!ec)
		{
			//printf("%.*s\n", read_msg.body_length(), read_msg.body());

			std::cout.write(read_msg.body(), read_msg.body_length());
			std::cout << "\n";
			
			process_message(read_msg);

			do_read_header();
		}
		else
		{
			socket.close();
		}
	});
}

void Slave::do_write()
{
	boost::asio::async_write(socket,
	boost::asio::buffer(write_msgs.front().data(),
		write_msgs.front().length()),
		[this](boost::system::error_code ec, std::size_t /*length*/)
		{
			if (!ec)
			{
				write_msgs.pop_front();
				if (!write_msgs.empty())
				{
				  do_write();
				}
			}
			else
			{
				socket.close();
			}
		});
}

void Slave::send_anjing()
{
	Message msg;
	std::string anjing = "anjing";

	msg.set_body_length(anjing.length());
	std::memcpy(msg.body(), anjing.c_str(), anjing.length());
	msg.encode_header();
	send(msg);
}

void Slave::send(const Message& msg)
{
	io_service.post(
	[this, msg]()
	{
		bool write_in_progress = !write_msgs.empty();
		write_msgs.push_back(msg);
		if (!write_in_progress)
		{
			do_write();
		}
	});
}

void Slave::process_message(const Message& message)
{
	send_anjing();

}