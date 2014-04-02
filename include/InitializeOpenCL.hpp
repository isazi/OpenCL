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
using std::vector;

#include <utils.hpp>
#include <Exceptions.hpp>
using isa::utils::toStringValue;
using isa::Exceptions::OpenCLError;


#ifndef INITIALIZE_OPENCL_HPP
#define INITIALIZE_OPENCL_HPP

namespace isa {

namespace OpenCL {

void initializeOpenCL(unsigned int platform, unsigned int nrQueues, vector< cl::Platform > *platforms, cl::Context *context, vector< cl::Device > *devices, vector< vector< cl::CommandQueue > > *queues) throw (OpenCLError) {
	try {
		unsigned int nrDevices = 0;

		cl::Platform::get(platforms);
		cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms->at(platform))(), 0};
		*context = cl::Context(CL_DEVICE_TYPE_ALL, properties);

		*devices = context->getInfo<CL_CONTEXT_DEVICES>();
		nrDevices = devices->size();
		for ( unsigned int device = 0; device < nrDevices; device++ ) {
			queues->push_back(vector< cl::CommandQueue >());
			for ( unsigned int queue = 0; queue < nrQueues; queue++ ) {
				(queues->at(device)).push_back(cl::CommandQueue(*context, devices->at(device)));;
			}
		}
	}
	catch ( cl::Error e ) {
		string err_s = toStringValue< cl_int >(e.err());
		throw OpenCLError("Impossible to initialize OpenCL: " + err_s + ".");
	}
}

} // OpenCL
} // isa

#endif // INITIALIZE_OPENCL_HPP

