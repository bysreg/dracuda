#include "slave.hpp"

#include <boost/thread/thread.hpp>
#include <boost/asio.hpp>
#include <string>
#include <cstdlib>
#include <boost/lexical_cast.hpp>

static const int write_msg_max_length = 800*600*3;
static const int read_msg_max_length = 1000;

Slave::Slave(boost::asio::io_service& io_service, 
	tcp::resolver::iterator endpoint_iterator_)
	: io_service(io_service), endpoint_iterator(endpoint_iterator_),
	  socket(io_service), read_msg(read_msg_max_length)
{}

Slave& Slave::start(const std::string& host) 
{
	using boost::asio::ip::tcp;

	static boost::asio::io_service io_service;

	std::cout<<"starting slave server..."<<std::endl;	

	tcp::resolver resolver(io_service);
	auto endpoint_iterator = resolver.resolve({ host, boost::lexical_cast<std::string>(50000)}); // 1030 is the port number
	static Slave slave(io_service, endpoint_iterator);

	return slave;
}

void Slave::stop()
{}

void Slave::run()
{
	do_connect(endpoint_iterator);

	boost::thread t(boost::bind(&boost::asio::io_service::run,
		&io_service));
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
	// std::cout<<"trying to call read header"<<std::endl;

	// async read the message header
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
	// std::cout<<"trying to call read body"<<std::endl;

	// async read the message body
	boost::asio::async_read(socket,
		boost::asio::buffer(read_msg.body(), read_msg.body_length()),
		[this](boost::system::error_code ec, std::size_t /*length*/)
		{
			if (!ec)
			{
				//printf("%.*s\n", read_msg.body_length(), read_msg.body());
				// std::cout<<"received : ";
				// std::cout.write(read_msg.body(), read_msg.body_length());
				// std::cout << "\n";

				process_message(read_msg);
				do_read_header();
			}
			else
			{
				// something is wrong
				std::cout<<"something is wrong "<<ec<<std::endl;

				socket.close();
			}
		});
}

void Slave::send(const unsigned char* chars, int size)
{
	Message* msg = new Message(size);

	msg->set_body_length(size);
	std::memcpy(msg->body(), chars, size);
	msg->encode_header();
	send(msg);
}

void Slave::send(const std::string& str)
{
	// std::cout<<"trying to send a string"<<std::endl;

	Message* msg = new Message(str.length());

	msg->set_body_length(str.length());
	std::memcpy(msg->body(), str.c_str(), str.length());
	msg->encode_header();
	send(msg);
}

void Slave::send(Message* msg)
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

void Slave::do_write()
{
	boost::asio::async_write(socket,
		boost::asio::buffer(write_msgs.front()->data(),
		write_msgs.front()->length()),
		[this](boost::system::error_code ec, std::size_t /*length*/)
		{
			if (!ec)
			{
				// delete the recent sent message
				Message* sent = write_msgs.front();
				delete sent;

				write_msgs.pop_front();
				if (!write_msgs.empty())
				{
				  do_write();
				}
			}
			else
			{
				std::cout<<"something is wrong"<<std::endl;
				socket.close();
			}
		});
}

struct Test {
	char a;
	char b;
	char c;
	char d;
	Test() {
		a = 'a';
		b = 'n';
		c = 'j';
		d = 'i';
	}
};

void Slave::set_on_message_received(std::function<void(const Message& message)> const& cb)
{
	on_message_received = cb;	
}

void Slave::process_message(const Message& message)
{
	if(on_message_received) 
	{
		on_message_received(message);
	}
}

static void test_char_array(unsigned char* arr, int size)
{
	for(int i=0;i<size;i++)
	{
		arr[i] = (i % 10) + '0';
		// std::cout<< arr[i];
	}
	// std::cout<<std::endl;
}

// int main() 
// {
// 	std::string localhost = "localhost";
// 	Slave& slave = Slave::start(localhost);
// 	const int size = 800*600*3;
// 	unsigned char big_char_arr[size];
// 	test_char_array(big_char_arr, size);

// 	slave.set_on_message_received(
// 		[&slave, &big_char_arr, size](const Message& msg) {		
// 			std::cout.write(msg.body(), msg.body_length());
// 			std::cout << "\n";

// 			std::cout<<"^^^ receive ("<<msg.body_length()<<")" << std::endl;

// 			// slave.send("anjing");
// 			slave.send(big_char_arr, size);
// 		}
// 	);

// 	slave.run();

// 	while(true) {}

// 	return 0;
// }
