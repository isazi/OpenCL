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
#include <utility>

using std::string;
using std::make_pair;

#include <kernels/Kernel.hpp>
#include <GPUData.hpp>
#include <Exceptions.hpp>
#include <utils.hpp>

using isa::OpenCL::Kernel;
using isa::OpenCL::GPUData;
using isa::Exceptions::OpenCLError;
using isa::utils::toStringValue;


#ifndef MEMSET_HPP
#define MEMSET_HPP

namespace isa {

namespace OpenCL {

template < typename T > class Memset : public Kernel< T > {
public:
	Memset(string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(T value, GPUData< T > *memory) throw (OpenCLError);

	inline void setNrThreadsPerBlock(unsigned int threads);
	inline void setNrThreads(unsigned int threads);
	inline void setNrRows(unsigned int rows);
	
private:
	unsigned int nrThreadsPerBlock;
	unsigned int nrThreads;
	unsigned int nrRows;
};


// Implementation

template< typename T > Memset< T >::Memset(string dataType) : Kernel< T >("Memset", dataType), nrThreadsPerBlock(0), nrThreads(0), nrRows(0) {}


template< typename T > void Memset< T >::generateCode() throw (OpenCLError) {
	if ( this->code != 0 ) {
		delete this->code;
	}
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(" + this->dataType + " value, __global " + this->dataType + " *mem) {\n"
		"unsigned int id = (get_group_id(1) * get_num_groups(0) * get_local_size(0)) + (get_group_id(0) * get_local_size(0)) + get_local_id(0);\n"
		"mem[id] = value;\n"
		"}";

	this->setAsync(true);
	this->compile();
}


template< typename T > void Memset< T >::operator()(T value, GPUData< T > *memory) throw (OpenCLError) {
	cl::NDRange globalSize(nrThreads / nrRows, nrRows);
	cl::NDRange localSize(nrThreadsPerBlock, 1);

	this->setArgument(0, value);
	this->setArgument(1, *(memory->getDeviceData()));

	this->run(globalSize, localSize);
}


template< typename T > inline void Memset< T >::setNrThreadsPerBlock(unsigned int threads) {
	nrThreadsPerBlock = threads;
}


template< typename T > inline void Memset< T >::setNrThreads(unsigned int threads) {
	nrThreads = threads;
}


template< typename T > inline void Memset< T >::setNrRows(unsigned int rows) {
	nrRows = rows;
}


} // OpenCL
} // isa

#endif // MEMSET_HPP

