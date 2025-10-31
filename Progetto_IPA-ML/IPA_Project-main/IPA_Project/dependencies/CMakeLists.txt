# PyBind11
add_subdirectory(pybind11-2.13.6)

set(IPA_Project_Deps ${IPA_Project_Deps} pybind11::embed PARENT_SCOPE)