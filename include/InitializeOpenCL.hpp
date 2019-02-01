// Copyright 2011 Alessio Sclocco <alessio@sclocco.eu>
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

#include "Exceptions.hpp"


#pragma once

namespace isa {
namespace OpenCL {

/**
 ** @brief Data structure holding all the OpenCL runtime objects.
 */
struct OpenCLRunTime
{
    cl::Context * context = nullptr;
    std::vector<cl::Platform> * platforms = nullptr;
    std::vector<cl::Device> * devices = nullptr;
    std::vector<std::vector<cl::CommandQueue>> * queues = nullptr;
};

/**
 ** @brief Initialize OpenCL environment.
 **
 ** @param platform OpenCL platform ID.
 ** @param nrQueues Number of queues to initialize per device.
 ** @param openclRuntime OpenCL object to initialize.
 */
void initializeOpenCL(const unsigned int platform, const unsigned int nrQueues, OpenCLRunTime &openclRuntime);

} // OpenCL
} // isa
