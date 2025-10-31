#pragma once

#include <string_view>
#include "Singleton.hpp"

namespace proj
{
	class PythonSystem : public Singleton<PythonSystem>
	{
	public:

		PythonSystem();
		~PythonSystem();

	public:

		void ExecuteScript(std::string_view script);

	private:
		void* m_python_engine{ nullptr };
	};
}