// Copyright 2012 Alessio Sclocco <alessio@sclocco.eu>
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


#pragma once

namespace isa
{
namespace OpenCL
{
/**
 ** @brief Compile code into an OpenCL kernel.
 **
 ** @param name The name of the OpenCL function.
 ** @param code A string containing the code.
 ** @param flags Compiler flags.
 ** @param clContext The OpenCL context.
 ** @param clDevice The OpenCL device to target.
 ** @return 
 */
cl::Kernel * compile(const std::string & name, const std::string & code, const std::string & flags, cl::Context & clContext, cl::Device & clDevice);
} // namespace OpenCL
} // namespace isa
