cmake_minimum_required(VERSION 3.10)
project(isa::opencl VERSION 1.1)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")

# libisa_opencl
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

install(TARGETS isa_opencl
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
