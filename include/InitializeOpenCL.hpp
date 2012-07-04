/*
 * Copyright (C) 2011
 * Alessio Sclocco <a.sclocco@vu.nl>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <vector>
using std::vector;

#include <utils.hpp>
using isa::utils::toStringValue;
#include <Exceptions.hpp>
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

