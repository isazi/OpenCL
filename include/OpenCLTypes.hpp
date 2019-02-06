// Copyright 2019 Netherlands eScience Center
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <vector>
#include <string>


#pragma once

namespace isa
{
namespace OpenCL
{
/**
 ** @brief Data structure containing all the necessary OpenCL runtime objects.
 */
struct OpenCLRunTime
{
    cl::Context * context = nullptr;
    std::vector<cl::Platform> * platforms = nullptr;
    std::vector<cl::Device> * devices = nullptr;
    std::vector<std::vector<cl::CommandQueue>> * queues = nullptr;
};

/**
 ** @brief OpenCL error.
 */
class OpenCLError : public std::exception {
public:
	explicit OpenCLError(const std::string & message);
	~OpenCLError();
	const char *what() const noexcept;

private:
    std::string message;
};

/**
 ** @brief Basic OpenCL kernel configuration.
 */
class KernelConf
{
  public:
    KernelConf();
    ~KernelConf();
    /**
     ** @brief Get the integer type used in the kernel.
     ** This type is not the one associated with data, but the one used for internal variables.
     **
     ** @return 0 for "int", 1 for "unsigned int".
     */
    inline std::string getIntType() const;
    inline unsigned int getNrThreadsD0() const;
    inline unsigned int getNrThreadsD1() const;
    inline unsigned int getNrThreadsD2() const;
    inline unsigned int getNrItemsD0() const;
    inline unsigned int getNrItemsD1() const;
    inline unsigned int getNrItemsD2() const;
    /**
     ** @brief Set the integer type to use in the kernel.
     ** This type is not the one associated with data, but the one used for internal variables.
     **
     ** @param Integer representing the type: 0 is "int", 1 is "unsigned int".
     */
    inline void setIntType(unsigned int type);
    inline void setNrThreadsD0(unsigned int threads);
    inline void setNrThreadsD1(unsigned int threads);
    inline void setNrThreadsD2(unsigned int threads);
    inline void setNrItemsD0(unsigned int items);
    inline void setNrItemsD1(unsigned int items);
    inline void setNrItemsD2(unsigned int items);
    // utils
    std::string print() const;

  private:
    unsigned int nrThreadsD0, nrThreadsD1, nrThreadsD2;
    unsigned int nrItemsD0, nrItemsD1, nrItemsD2;
    unsigned int intType;
};
} // OpenCL
} // isa