// Copyright 2011 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <utils.hpp>
#include <Exceptions.hpp>


#ifndef INITIALIZE_OPENCL_HPP
#define INITIALIZE_OPENCL_HPP

namespace isa {
namespace OpenCL {

void initializeOpenCL(unsigned int platform, unsigned int nrQueues, std::vector< cl::Platform > * platforms, cl::Context * context, std::vector< cl::Device > * devices, std::vector< std::vector< cl::CommandQueue > > * queues) throw (isa::OpenCL::OpenCLError)

} // OpenCL
} // isa

#endif // INITIALIZE_OPENCL_HPP

