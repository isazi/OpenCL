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
#include <string>
using std::string;
#include <utility>
using std::make_pair;

#include <GPUData.hpp>
using isa::OpenCL::GPUData;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <utils.hpp>
using isa::utils::toStringValue;


#ifndef MEMSET_HPP
#define MEMSET_HPP

namespace isa {

namespace OpenCL {

template < typename T > class Memset {
public:
	Memset(string dataType);
	~Memset();

	void compile(cl::Context *clContext, cl::Device *clDevice) throw (OpenCLError);
	void run(T value, GPUData< T > *memory) throw (OpenCLError);

	inline void setCLQueue(cl::CommandQueue *queue);
	
private:
	string dataType;
	cl::Kernel *kernel;
	cl::CommandQueue *clCommands;
};


// Implementation

template< typename T > Memset< T >::Memset(string dataType) : dataType(dataType), kernel(0), clCommands(0) {}


template< typename T > Memset< T >::~Memset() {
	if ( kernel != 0 ) {
		delete kernel;
	}
}


template< typename T > void Memset< T >::compile(cl::Context *clContext, cl::Device *clDevice) throw (OpenCLError) {

	string code = "__kernel void Memset(" + dataType + " value, __global " + dataType + " *mem) {\nmem[get_global_id(0)] = value;\n}";

	cl::Program *clProgram = 0;
	try {
		cl::Program::Sources sources(1, make_pair(code.c_str(), code.length()));
		clProgram = new cl::Program(*clContext, sources);
		clProgram->build(vector< cl::Device >(1, *clDevice));
	}
	catch ( cl::Error err ) {
		throw OpenCLError("It is not possible to build the Memset OpenCL program: " + clProgram->getBuildInfo< CL_PROGRAM_BUILD_LOG >(*clDevice) + ".");
	}

	if ( kernel != 0 ) {
		delete kernel;
	}
	try {
		kernel = new cl::Kernel(*clProgram, "Memset", NULL);
	}
	catch ( cl::Error err ) {
		string err_s = toStringValue< cl_int >(err.err());
		throw OpenCLError("It is not possible to create the kernel for Memset: " + err_s + ".");
	}
	delete clProgram;
}


template< typename T > void Memset< T >::run(T value, GPUData< T > *memory) throw (OpenCLError) {
	if ( kernel == 0 ) {
		throw OpenCLError("First generate the kernel.");
	}

	cl::NDRange globalSize(memory->getDeviceDataSize() / sizeof(T));
	kernel->setArg(0, value);
	kernel->setArg(1, *(memory->getDeviceData()));
	
	try {
		clCommands->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, cl::NullRange, NULL, NULL);
	}
	catch ( cl::Error err ) {
		string err_s = toStringValue< cl_int >(err.err());
		throw OpenCLError("Impossible to run Memset: " + err_s + ".");
	}
}


template< typename T > inline void setCLQueue(cl::CommandQueue *queue) {
	clCommands = queue;
}

} // OpenCL
} // isa

#endif // MEMSET_HPP

