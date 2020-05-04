cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE gui)
add_executable(${EXECUTABLE} MACOSX_BUNDLE main.cpp)
target_link_libraries(${EXECUTABLE} fslazywindow ysbitmapfont)

# Set O3 optimization flag
set( CMAKE_CXX_FLAGS "-O3 -Wall -Wextra ")

# filesystem
target_link_libraries(${EXECUTABLE}  stdc++fs )

# Find Package Python2
find_package(PythonInterp 2.7 REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
set(PROJECT_LIBRARIES ${PYTHON_LIBRARIES} )
set(PROJECT_INCLUDES_PYTHON ${PYTHON_INCLUDE_DIRS} )
MESSAGE(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
MESSAGE(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
include_directories(${PROJECT_INCLUDES_PYTHON})
target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES})
