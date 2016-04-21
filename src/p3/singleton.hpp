#pragma once

#define DRACUDA_SINGLETON(T)\
	public:\
	static T& GetInstance()\
	{\
		static T instance;\
		return instance;\
	}\
	T(T const& other) = delete;\
	void operator=(T const& other) = delete;\
	private:\
	T();

