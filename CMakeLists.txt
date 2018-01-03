cmake_minimum_required(VERSION 3.10)
project(isa::opencl VERSION 1.1)
include(GNUInstallDirs)

add_library(isa_opencl SHARED
  src/Exceptions.cpp
  src/InitializeOpenCL.cpp
  src/Kernel.cpp
)
set_target_properties(isa_opencl PROPERTIES 
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "include/Bits.hpp;include/Exceptions.hpp;include/InitializeOpenCL.hpp;include/Kernel.hpp"
)
target_include_directories(isa_opencl PRIVATE include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g0 -march=native -mtune=native")

install(TARGETS isa_opencl
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
