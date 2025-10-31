cmake_minimum_required(VERSION 3.20)
project(IPA_Project)

message("IPA Project made by Carcone-Cicione")

set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# OpenCV
find_package(OpenCV REQUIRED)

# Additional deps
set(IPA_Project_Deps )
add_subdirectory(dependencies)

# Sources
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.c)
file(GLOB_RECURSE HDR_FILES CONFIGURE_DEPENDS src/*.hpp src/*.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${SRC_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Header Files" FILES ${HDR_FILES})

# Exe target
add_executable(IPA_Project ${SRC_FILES} ${HDR_FILES})
target_link_libraries(IPA_Project PRIVATE ${OpenCV_LIBS} ${IPA_Project_Deps})
target_include_directories(IPA_Project PRIVATE ${OpenCV_INCLUDE_DIRS})

if (CMAKE_GENERATOR MATCHES "Visual Studio")
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${SRC_FILES})
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Header Files" FILES ${HDR_FILES})
endif()
