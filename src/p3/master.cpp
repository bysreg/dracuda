#include <boost/thread/thread.hpp>
#include "master.hpp"

int Master::read_msg_max_length = 800*600*3;

Connection::Connection(boost::asio::ip::tcp::socket socket_, 
	Master& master_)
	: socket(std::move(socket_)), read_msg(Master::read_msg_max_length), 
	  master(master_)
{}

void Connection::start()
{
	std::cout<<"a connection started"<<std::endl;

	if(master.on_connection_started) 
	{
		master.on_connection_started(*this);
	}

	do_read_header();
}

void Connection::send(const unsigned char* chars, int size)
{
	MessagePtr msg = std::make_shared<Message>(size);

	msg->set_body_length(size);
	std::memcpy(msg->body(), chars, size);
	msg->encode_header();
	send(msg);
}

void Connection::send(const std::string& str)
{
	MessagePtr msg = std::make_shared<Message>(str.length());

	msg->set_body_length(str.length());
	std::memcpy(msg->body(), str.c_str(), str.length());
	msg->encode_header();
	send(msg);
}

void Connection::send(MessagePtr msg)
{
	bool write_in_progress = !write_msgs.empty();
	write_msgs.push_back(msg);
	if (!write_in_progress)
	{
		do_write();
	}
}

void Connection::do_write()
{
	// std::cout<<"trying to call write"<<std::endl;

	auto self(shared_from_this());
	boost::asio::async_write(socket,
		boost::asio::buffer(write_msgs.front()->data(),
		write_msgs.front()->length()),
		[this, self](boost::system::error_code ec, std::size_t /*length*/)
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
				//something is wrong
			}
		});
}

void Connection::do_read_header()
{
	// std::cout<<"trying to call read header"<<std::endl;

	auto self(shared_from_this());
	boost::asio::async_read(socket,
		boost::asio::buffer(read_msg.data(), Message::header_length),
		[this, self](boost::system::error_code ec, std::size_t /*length*/)
		{
			if (!ec && read_msg.decode_header())
			{
				do_read_body();
			}
			else
			{
				// something is wrong
				std::cout<<"something is wrong "<<ec<<std::endl;
			}
		});
}

void Connection::do_read_body()
{
	// std::cout<<"trying to call read body"<<std::endl;

	auto self(shared_from_this());
	boost::asio::async_read(socket,
		boost::asio::buffer(read_msg.body(), read_msg.body_length()),
		[this, self](boost::system::error_code ec, std::size_t /*length*/)
		{
			if (!ec)
			{
				// std::cout.write(read_msg.body(), read_msg.body_length());
				// std::cout << "\n";
				// std::cout<<"receiving something : ";
				master.on_message_received(read_msg);
				do_read_header();
			}
			else
			{
				// something is wrong
				std::cout<<"something is wrong "<<ec<<std::endl;
			}
		});
}

Master::Master(boost::asio::io_service& io_service)
	: acceptor(io_service, tcp::endpoint(tcp::v4(), 50000)),
	socket(io_service)
{
	do_accept();
}

Master& Master::start()
{
	static boost::asio::io_service io_service;

	static Master master(io_service);

	std::cout<<"starting master..."<<std::endl;

	boost::thread t(boost::bind(&boost::asio::io_service::run,
		&io_service));

	return master;
}

void Master::do_accept()
{
	acceptor.async_accept(socket,
		[this](boost::system::error_code ec)
		{
			if (!ec)
			{
				auto conn_ptr = std::make_shared<Connection>(std::move(socket), *this);
				this->connections.insert(conn_ptr);

				conn_ptr->start();
			}

			do_accept();
		});
}

void Master::send_all(const std::string& str)
{
	MessagePtr msg = std::make_shared<Message>(str.length());

	msg->set_body_length(str.length());
	std::memcpy(msg->body(), str.c_str(), str.length());
	msg->encode_header();
	send_all(msg);	
}

void Master::send_all(MessagePtr msg)
{
	for (auto connection: connections) {
		connection->send(msg);
	}
}

int Master::get_connections_count() const
{
	return connections.size();
}

void Master::set_on_message_received(std::function<void(const Message& message)> const& cb)
{
	on_message_received = cb;
}

void Master::set_on_connection_started(std::function<void(Connection& connection)> const& cb)
{
	on_connection_started = cb;
}

struct Test {
	char a;
	char b;
	char c;
	char d;
	Test() {
		a = 'h';
		b = 'i';
		c = 'l';
		d = 'm';
	}
};

static void test_char_array(unsigned char* arr, int size)
{
	for(int i=0;i<size;i++)
	{
		arr[i] = (i % 10) + '0';
		// std::cout<< arr[i];
	}
	// std::cout<<std::endl;
}

// int main(int argc, char* argv[])
// {
// 	try
// 	{
// 		const int size = 800*600*3;
// 		unsigned char* big_char_arr = new unsigned char[size];
// 		test_char_array(big_char_arr, size);

// 		Master& master = Master::start();  		
// 		master.set_on_message_received(
// 			[&master](const Message& msg) {				
// 				std::cout.write(msg.body(), msg.body_length());
// 				std::cout << "\n";

// 				std::cout<<"^^^ receive ("<<msg.body_length()<<")" << std::endl;
// 			}
// 		);
// 		master.set_on_connection_started(
// 			[&master, &big_char_arr, size](Connection& connection) {
// 				// connection.send(big_char_arr, size);
					
// 				// first connection from a slave, 
// 				// say hello world

// 				connection.send("hello world");
// 			}
// 		);

// 		while(true) 
// 		{
// 			std::string s;
// 			std::cin>>s;

// 			master.send_all(s);

// 			// test send struct to all
// 			Test t;
// 			t.a = 'x';t.b = 'y';t.c = 'z';
// 			master.send_all(t);
// 		};

// 	}
// 	catch (std::exception& e)
// 	{
// 		std::cerr << "Exception: " << e.what() << "\n";
// 	}

// 	return 0;
// }
