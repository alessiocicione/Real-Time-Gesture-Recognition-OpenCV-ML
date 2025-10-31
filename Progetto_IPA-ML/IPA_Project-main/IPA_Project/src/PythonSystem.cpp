#include <pybind11/embed.h>
namespace py = pybind11;

#include "PythonSystem.hpp"

namespace proj
{
	PythonSystem::PythonSystem()
	{
		m_python_engine = new py::scoped_interpreter{};
	}

	PythonSystem::~PythonSystem()
	{
		if (m_python_engine)
		{
			delete m_python_engine;
			m_python_engine = nullptr;
		}
	}

	void
	PythonSystem::ExecuteScript(std::string_view script)
	{
		py::exec(script.data());
	}
}