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

#include <kernels/Kernel.hpp>
using isa::OpenCL::Kernel;
#include <GPUData.hpp>
using isa::OpenCL::GPUData;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <utils.hpp>
using isa::utils::giga;


#ifndef VECTOR_ADD_HPP
#define VECTOR_ADD_HPP

namespace isa {

namespace OpenCL {

template < typename T > class VectorAdd : public Kernel< T > {
public:
	VectorAdd(string dataType);
	~VectorAdd();

	void compile(cl::Context &clContext, cl::Device &clDevice, cl::CommandQueue *clCommands) throw (OpenCLError);
	void run(GPUData< T > *a, GPUData< T > *b, GPUData< T > *c) throw (OpenCLError);
	
	inline string getCode() const;
	inline double getArithmeticIntensity() const;
	inline double getGFLOP() const;
	inline double getGB() const;

private:
	string *code;

	double arInt;
	double gflop;
	double gb;
};


// Implementation

template< typename T > VectorAdd< T >::VectorAdd(string dataType) : Kernel< T >("VectorAdd", dataType), code(0), arInt(0.0), gflop(0.0), gb(0.0) {}


template< typename T > VectorAdd< T >::~VectorAdd() {
	if ( code != 0 ) {
		delete code;
	}
}


template< typename T > void VectorAdd< T >::compile(cl::Context &clContext, cl::Device &clDevice, cl::CommandQueue *clCommands) throw (OpenCLError) {
	if ( code != 0 ) {
		delete code;
	}
	code = new string();
	*code = "__kernel void " + Kernel< T >::getName() + "(__global " + Kernel< T >::getDataType() + " *A, __global " + Kernel< T >::getDataType() + " *B, __global " + Kernel< T >::getDataType() + " *C) {\nC[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];\n}";

	Kernel< T >::compile(clContext, clDevice, clCommands, *code);
}


template< typename T > void VectorAdd< T >::run(GPUData< T > *a, GPUData< T > *b, GPUData< T > *c) throw (OpenCLError) {
	long long unsigned int ops = a->getDeviceDataSize() / sizeof(T);
	long long unsigned int memOps = (a->getDeviceDataSize() / sizeof(T)) * 12;

	arInt = ops / static_cast< double >(memOps);
	gflop = giga(ops);
	gb = giga(memOps);


	cl::NDRange globalSize(ops);
	cl::NDRange localSize(cl::NullRange);
	Kernel< T >::setArgument(0, *(a->getDeviceData()));
	Kernel< T >::setArgument(1, *(b->getDeviceData()));
	Kernel< T >::setArgument(2, *(c->getDeviceData()));

	Kernel< T >::run(globalSize, localSize);
}


template< typename T > inline string VectorAdd< T >::getCode() const {
	if ( code != 0 ) {
		return *code;
	}
	
	return string();
}


template< typename T > inline double VectorAdd< T >::getArithmeticIntensity() const {
	return arInt;
}


template< typename T > inline double VectorAdd< T >::getGFLOP() const {
	return gflop;
}


template< typename T > inline double VectorAdd< T >::getGB() const {
	return gb;
}

} // OpenCL
} // isa

#endif // VECTOR_ADD_HPP

