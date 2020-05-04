cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(gui)

add_executable(main MACOSX_BUNDLE SegFault_main.cpp Python.h)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR} )