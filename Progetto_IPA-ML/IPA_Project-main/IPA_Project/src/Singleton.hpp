#pragma once

#include <cassert>

namespace proj
{
	template<typename T>
	class Singleton
	{
	public:

		Singleton<T>()
		{
			assert(!s_instance && "Singleton already allocated");
			s_instance = static_cast<T*>(this);
		}

		~Singleton<T>()
		{
			assert(s_instance);
			s_instance = nullptr;
		}

		Singleton<T>(const Singleton<T>&) = delete;
		Singleton<T>& operator = (const Singleton<T>&) = delete;

	public:

		static T* GetPtr() 
		{
			return s_instance;
		}

		static T& Get()
		{
			assert(s_instance);
			return *s_instance;
		}

	private:
		inline static T* s_instance{ nullptr };
	};
}