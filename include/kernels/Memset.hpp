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
using isa::utils::toStringValue;


#ifndef MEMSET_HPP
#define MEMSET_HPP

namespace isa {

namespace OpenCL {

template < typename T > class Memset : public Kernel {
public:
	Memset(string dataType);
	~Memset();

	void compile(cl::Context &clContext, cl::Device &clDevice, cl::CommandQueue *clCommands) throw (OpenCLError);
	void run(T value, GPUData< T > *memory) throw (OpenCLError);
	
private:
	string *code;
};


// Implementation

template< typename T > Memset< T >::Memset() : Kernel("Memset", dataType), code(0) {}


template< typename T > Memset< T >::~Memset() {
	if ( code != 0 ) {
		delete code;
	}
}


template< typename T > void Memset< T >::compile(cl::Context &clContext, cl::Device &clDevice, cl::CommandQueue *clCommands) throw (OpenCLError) {
	if ( code != 0 ) {
		delete code;
	}
	code = new string();
	*code = "__kernel void " + getName() + "(" + getDataType() + " value, __global " + getDataType() + " *mem) {\nmem[get_global_id(0)] = value;\n}";

	Kernel::compile(clContext, clDevice, clCommands, *code);
}


template< typename T > void Memset< T >::run(T value, GPUData< T > *memory) throw (OpenCLError) {

	cl::NDRange globalSize(memory->getDeviceDataSize() / sizeof(T));
	setArgument< T >(0, value);
	setArgument< cl::Buffer >(1, *(memory->getDeviceData()));

	Kernel::run(true, globalSize, cl::NullRange);
}


} // OpenCL
} // isa

#endif // MEMSET_HPP

