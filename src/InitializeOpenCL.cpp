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

void initializeOpenCL(unsigned int platform, unsigned int nrQueues, std::vector<cl::Platform> * platforms, cl::Context * context, std::vector<cl::Device> * devices, std::vector<std::vector<cl::CommandQueue>> * queues) {
	try {
		unsigned int nrDevices = 0;

		cl::Platform::get(platforms);
		cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms->at(platform))(), 0};
		*context = cl::Context(CL_DEVICE_TYPE_ALL, properties);

		*devices = context->getInfo<CL_CONTEXT_DEVICES>();
		nrDevices = devices->size();
		for ( unsigned int device = 0; device < nrDevices; device++ ) {
			queues->push_back(std::vector< cl::CommandQueue >());
			for ( unsigned int queue = 0; queue < nrQueues; queue++ ) {
				(queues->at(device)).push_back(cl::CommandQueue(*context, devices->at(device)));;
			}
		}
	}	catch ( cl::Error & e ) {
		throw isa::OpenCL::OpenCLError("ERROR: impossible to initialize OpenCL \"" + isa::utils::toString(e.err()) + "\"");
	}
}

} // OpenCL
} // isa

