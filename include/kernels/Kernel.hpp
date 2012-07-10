/*
 * Copyright (C) 2012
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

#include <Timer.hpp>
using LOFAR::NSTimer;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <GPUData.hpp>
using isa::OpenCL::GPUData;
#include <utils.hpp>
using isa::utils::toStringValue;


#ifndef KERNEL_HPP
#define KERNEL_HPP

namespace isa {
namespace OpenCL {

template< typename T > class Kernel {
public:
	Kernel(string name, string dataType);
	~Kernel();
	
	void compile(cl::Context &clContext, cl::Device &clDevice, cl::CommandQueue *queue, string &code) throw (OpenCLError);
	template< typename A > inline void setArgument(unsigned int id, A param) throw (OpenCLError);
	void run(bool async, cl::NDRange &globalSize, cl::NDRange &localSize) throw (OpenCLError);


	inline string getName() const;
	inline string getDataType() const;
	inline string getBuildLog() const;
	inline double getTime() const;

private:
	string name;
	string dataType;
	string buildLog;
	cl::Kernel *kernel;
	cl::CommandQueue *clCommands;
	NSTimer timer;
};


// Implementation

template< typename T > Kernel< T >::Kernel(string name, string dataType) : name(name), dataType(dataType), buildLog(string()), kernel(0), clCommands(0), timer(NSTimer(name, false, false)) {}


template< typename T > Kernel< T >::~Kernel() {
	if ( kernel != 0 ) {
		delete kernel;
	}
}


template< typename T > void Kernel< T >::compile(cl::Context &clContext, cl::Device &clDevice, cl::CommandQueue *queue, string &code) throw (OpenCLError) {
	clCommands = queue;

	cl::Program *program;
	try {
		cl::Program::Sources sources(1, make_pair(code.c_str(), code.length()));
		program = new cl::Program(clContext, sources, NULL);
		program->build(vector< cl::Device >(1, clDevice), "-cl-mad-enable", NULL, NULL);
		buildLog = program->getBuildInfo< CL_PROGRAM_BUILD_LOG >(clDevice);
	}
	catch ( cl::Error err ) {	
		throw OpenCLError("It is not possible to build the " + name + " OpenCL program: " + program->getBuildInfo< CL_PROGRAM_BUILD_LOG >(clDevice) + ".");
	}
	
	if ( kernel != 0 ) {
		delete kernel;
	}
	try {
		kernel = new cl::Kernel(*program, name.c_str(), NULL);
	}
	catch ( cl::Error err ) {
		delete program;
		throw OpenCLError("It is not possible to create the kernel for " + name + ": " + toStringValue< cl_int >(err.err()) + ".");
	}
	delete program;
}

	
template< typename T, typename A > inline Kernel< T >::setArgument< A >(unsigned int id, A param) throw (OpenCLError) {
	if ( kernel == 0 ) {
		throw OpenCLError("First generate the kernel for " + name + ".");
	}

	try {
		kernel->setArg(id, param);
	}
	catch ( cl::Error err ) {
		throw OpenCLError("Impossible to set " + name + " arguments: " + toStringValue< cl_int >(err.err()) + ".");
	}
}


template< typename T > void Kernel< T >::run(bool async, cl::NDRange &globalSize, cl::NDRange &localSize) throw (OpenCLError) {
	if ( kernel == 0 ) {
		throw OpenCLError("First generate the kernel for " + name + ".");
	}

	if ( async ) {
		try {
			clCommands->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, localSize, NULL, NULL);
		}
		catch ( cl::Error err ) {
			throw OpenCLError("Impossible to run " + name + ": " + toStringValue< cl_int >(err.err()) + ".");
		}
	}
	else {
		cl::Event clEvent;

		try {
			timer.start();
			clCommands->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, localSize, NULL, &clEvent);
			clEvent.wait();
			timer.stop();
		}
		catch ( cl::Error err ) {
			timer.stop();
			timer.reset();
			throw OpenCLError("Impossible to run " + name + ": " + toStringValue< cl_int >(err.err()) + ".");
		}
	}
}

	
template< typename T > inline string Kernel< T >::getName() const {
	return name;
}


template< typename T > inline string Kernel< T >::getDataType() const {
	return dataType;
}


template< typename T > inline string Kernel< T >::getBuildLog() const {
	return buildLog;
}


template< typename T > inline double Kernel< T >::getTime() const {
	return timer.getElapsed();
}

} // OpenCL
} // isa

#endif // KERNEL_HPP

