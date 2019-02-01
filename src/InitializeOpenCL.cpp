// Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <InitializeOpenCL.hpp>

namespace isa {
namespace OpenCL {

void initializeOpenCL(const unsigned int platform, const unsigned int nrQueues, OpenCLRunTime &openclRuntime)
{
	try
    {
		std::uint64_t nrDevices = 0;
		cl::Platform::get(openclRuntime.platforms);
		cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(openclRuntime.platforms->at(platform))(), 0};
		*(openclRuntime.context) = cl::Context(CL_DEVICE_TYPE_ALL, properties);
		*(openclRuntime.devices) = openclRuntime.context->getInfo<CL_CONTEXT_DEVICES>();
		nrDevices = openclRuntime.devices->size();
		for ( unsigned int device = 0; device < nrDevices; device++ )
        {
			openclRuntime.queues->push_back(std::vector< cl::CommandQueue >());
			for ( unsigned int queue = 0; queue < nrQueues; queue++ )
            {
				(openclRuntime.queues->at(device)).push_back(cl::CommandQueue(*(openclRuntime.context), openclRuntime.devices->at(device)));;
			}
		}
	}
    catch ( cl::Error &err )
    {
		throw isa::OpenCL::OpenCLError("ERROR: impossible to initialize OpenCL \"" + std::to_string(err.err()) + "\"");
	}
}

} // OpenCL
} // isa

