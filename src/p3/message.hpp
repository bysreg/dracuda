#pragma once

#include <iostream>

class Message
{

public:
	static const int header_length = sizeof(int);
	const int max_body_length;

	// constructor
	Message(int max_body_length) 
		: body_length_(0), max_body_length(max_body_length)
	{
		// std::cout<<"Message::Message()"<<std::endl;
		data_ = new char[header_length + max_body_length];
	}

	// destructor
	~Message()
	{
		// std::cout<<"Message::~Message()"<<std::endl;
		delete[] data_;
	}

	// copy constructor
	Message(const Message& msg) = delete;

	// copy assignment operator
	Message& operator= (const Message& other) = delete;

	inline const char* data() const
	{
		return data_;
	}

	inline char* data()
	{
		return data_;
	}

	inline int length() const
	{
		return header_length + body_length_;
	}

	inline const char* body() const
	{
		return data_ + header_length;
	}

	inline char* body()
	{
		return data_ + header_length;
	}

	inline int body_length() const
	{
		return body_length_;
	}

	inline bool decode_header()
	{		
		std::memcpy(&body_length_, data_, sizeof(body_length_));				
		// std::cout<<"body length is : " << body_length_<<std::endl;
		if (body_length_ > max_body_length)
		{
			// std::cout<<"message is not valid"<<std::endl;
			body_length_ = 0;
			return false;
		}
		// std::cout<<"message is valid"<<std::endl;
		return true;
	}

	inline void encode_header()
	{
		std::memcpy(data_, &body_length_, sizeof(body_length_));
	}

	inline void set_body_length(int val)
	{
		body_length_ = val;
		if(body_length_ > val)
			body_length_ = val;
	}



	
private:
	char* data_;
	int body_length_;
};	

typedef std::shared_ptr<Message> MessagePtr;