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
#include <utility>

using std::string;
using std::make_pair;

#include <Timer.hpp>
#include <Exceptions.hpp>
#include <GPUData.hpp>
#include <utils.hpp>

using LOFAR::NSTimer;
using isa::Exceptions::OpenCLError;
using isa::OpenCL::GPUData;
using isa::utils::toStringValue;


#ifndef KERNEL_HPP
#define KERNEL_HPP

namespace isa {
namespace OpenCL {

template< typename T > class Kernel {
public:
	Kernel(string name, string dataType);
	virtual ~Kernel();

	virtual void generateCode() = 0;
	inline void bindOpenCL(cl::Context *context, cl::Device *device, cl::CommandQueue *queue);
	inline void setAsync(bool asy);

	inline string getName() const;
	inline string getCode() const;
	inline string getDataType() const;
	inline string getBuildLog() const;
	inline double getTime() const;
	inline double getArithmeticIntensity() const;
	inline double getGFLOP() const;
	inline double getGB() const;

protected:
	void compile() throw (OpenCLError);
	template< typename A > inline void setArgument(unsigned int id, A param) throw (OpenCLError);
	void run(cl::NDRange &globalSize, cl::NDRange &localSize) throw (OpenCLError);

	bool async;
	string name;
	string *code;
	string dataType;
	string buildLog;
	cl::Kernel *kernel;
	cl::Context *clContext;
	cl::Device *clDevice;
	cl::CommandQueue *clCommands;
	cl::Event clEvent;
	NSTimer timer;
	
	double arInt;
	double gflop;
	double gb;
};


// Implementation

template< typename T > Kernel< T >::Kernel(string name, string dataType) : async(false), name(name), code(0), dataType(dataType), buildLog(string()), kernel(0), clContext(0), clDevice(0), clCommands(0), clEvent(cl::Event()), timer(NSTimer(name, false, false)), arInt(0.0), gflop(0.0), gb(0.0) {}


template< typename T > Kernel< T >::~Kernel() {
	if ( code != 0 ) {
		delete code;
	}
	if ( kernel != 0 ) {
		delete kernel;
	}
}


template< typename T > void Kernel< T >::compile() throw (OpenCLError) {
	cl::Program *program;
	try {
		cl::Program::Sources sources(1, make_pair(code.c_str(), code.length()));
		program = new cl::Program(*clContext, sources, NULL);
		program->build(vector< cl::Device >(1, *clDevice), "-cl-mad-enable", NULL, NULL);
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

	
template< typename T > template< typename A > inline void Kernel< T >::setArgument(unsigned int id, A param) throw (OpenCLError) {
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


template< typename T > void Kernel< T >::run(cl::NDRange &globalSize, cl::NDRange &localSize) throw (OpenCLError) {
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

	
template< typename T > inline void Kernel< T >::bindOpenCL(cl::Context *context, cl::Device *device, cl::CommandQueue *queue) {
	clContext = context;
	clDevice = device;
	clCommands = queue;
}


template< typename T > inline void Kernel< T >::setAsync(bool asy) {
	async = asy;
}


template< typename T > inline string Kernel< T >::getName() const {
	return name;
}


template< typename T > inline string Kernel< T >::getCode() const {
	if ( code != 0 ) {
		return *code;
	}
	
	return string();
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


template< typename T > inline double Kernel< T >::getArithmeticIntensity() const {
	return arInt;
}


template< typename T > inline double Kernel< T >::getGFLOP() const {
	return gflop;
}


template< typename T > inline double Kernel< T >::getGB() const {
	return gb;
}

} // OpenCL
} // isa

#endif // KERNEL_HPP

