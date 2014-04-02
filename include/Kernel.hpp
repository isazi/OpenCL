// Copyright 2012 Alessio Sclocco <a.sclocco@vu.nl>
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
#include <string>
using std::string;
#include <utility>
using std::make_pair;
#include <vector>
using std::vector;
#include <stdexcept>
using std::out_of_range;
#include <cmath>

#include <Timer.hpp>
using isa::utils::Timer;
#include <Stats.hpp>
using isa::utils::Stats;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
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

	inline void bindOpenCL(cl::Context * context, cl::Device * device, cl::CommandQueue * queue);
	inline void setAsync(bool asy);
	inline void setNvidia(bool nvd);
	inline void resetStats();

	inline string getName() const;
	inline string getCode() const;
	inline string getDataType() const;
	inline string getBuildLog() const;
	char * getBinary(unsigned int binary);
	inline Timer& getTimer();
	inline double getArithmeticIntensity() const;
	inline double getGFLOP() const;
	inline double getGB() const;
	inline double getGFLOPs() const;
	inline double getGFLOPsErr() const;
	inline double getGBs() const;
	inline double getGBsErr() const;

protected:
	void compile() throw (OpenCLError);
	template< typename A > inline void setArgument(unsigned int id, A param) throw (OpenCLError);
	void run(cl::NDRange & globalSize, cl::NDRange & localSize) throw (OpenCLError);

	bool async;
	bool nvidia;
	string name;
	string * code;
	string dataType;
	string buildLog;
	cl::Kernel * kernel;
	cl::Context * clContext;
	cl::Device * clDevice;
	cl::CommandQueue * clCommands;
	cl::Event clEvent;
	Timer timer;
	vector< char * > binaries;

	double arInt;
	double gflop;
	double gb;
	Stats< double > GFLOPs;
	Stats< double > GBs;
};


// Implementation

template< typename T > Kernel< T >::Kernel(string name, string dataType) : async(false), nvidia(false), name(name), code(0), dataType(dataType), buildLog(string()), kernel(0), clContext(0), clDevice(0), clCommands(0), clEvent(cl::Event()), timer(Timer(name)), binaries(vector< char * >()), arInt(0.0), gflop(0.0), gb(0.0), GFLOPs(Stats< double >()), GBs(Stats< double >()) {}


template< typename T > Kernel< T >::~Kernel() {
	delete code;
	delete kernel;

	for ( std::vector< char * >::iterator item = binaries.begin(); item != binaries.end(); item++ ) {
		delete [] *item;
	}
}


template< typename T > void Kernel< T >::compile() throw (OpenCLError) {
	cl::Program *program = 0;
	try {
		cl::Program::Sources sources(1, make_pair(code->c_str(), code->length()));
		program = new cl::Program(*clContext, sources, NULL);
		if ( nvidia ) {
			program->build(vector< cl::Device >(1, *clDevice), "-cl-mad-enable -cl-nv-verbose", NULL, NULL);
			program->getInfo(CL_PROGRAM_BINARIES, &binaries);
		} else {
			program->build(vector< cl::Device >(1, *clDevice), "-cl-mad-enable", NULL, NULL);
		}
		buildLog = program->getBuildInfo< CL_PROGRAM_BUILD_LOG >(*clDevice);
	} catch ( cl::Error err ) {
		throw OpenCLError("It is not possible to build the " + name + " OpenCL program: " + program->getBuildInfo< CL_PROGRAM_BUILD_LOG >(*clDevice) + ".");
	}

	if ( kernel != 0 ) {
		delete kernel;
	}
	try {
		kernel = new cl::Kernel(*program, name.c_str(), NULL);
	} catch ( cl::Error err ) {
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
	} catch ( cl::Error err ) {
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
		} catch ( cl::Error err ) {
			throw OpenCLError("Impossible to run " + name + ": " + toStringValue< cl_int >(err.err()) + ".");
		}
	} else {
		try {
			timer.start();
			clCommands->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, localSize, NULL, &clEvent);
			clEvent.wait();
			timer.stop();
			GFLOPs.addElement(gflop / timer.getLastRunTime());
			GBs.addElement(gb / timer.getLastRunTime());
		} catch ( cl::Error err ) {
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


template< typename T > inline void Kernel< T >::setNvidia(bool nvd) {
	nvidia = nvd;
}

template< typename T > inline void Kernel< T >::resetStats() {
	GFLOPs.reset();
	GBs.reset();
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


template< typename T > char *Kernel< T >::getBinary(unsigned int binary) {
	if ( nvidia ) {
		try {
			return binaries.at(binary);
		} catch ( out_of_range err ) {
			return 0;
		}
	}

	return 0;
}


template< typename T > inline Timer& Kernel< T >::getTimer() {
	return timer;
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

template< typename T > inline double Kernel< T >::getGFLOPs() const {
	return GFLOPs.getAverage();
}

template< typename T > inline double Kernel< T >::getGFLOPsErr() const {
	return GFLOPs.getStdDev();
}

template< typename T > inline double Kernel< T >::getGBs() const {
	return GBs.getAverage();
}

template< typename T > inline double Kernel< T >::getGBsErr() const {
	return GBs.getStdDev();
}

} // OpenCL
} // isa

#endif // KERNEL_HPP
