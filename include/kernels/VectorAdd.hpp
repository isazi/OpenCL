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
using isa::utils::giga;


#ifndef VECTOR_ADD_HPP
#define VECTOR_ADD_HPP

namespace isa {

namespace OpenCL {

template < typename T > class VectorAdd : public Kernel< T > {
public:
	VectorAdd(string dataType);

	void operator()(GPUData< T > *a, GPUData< T > *b, GPUData< T > *c) throw (OpenCLError);

	inline void setNrThreadsPerBlock(unsigned int threads);
	inline void setNrThreads(unsigned int threads);
	inline void setNrRows(unsigned int rows);

private:
	unsigned int nrThreadsPerBlock;
	unsigned int nrThreads;
	unsigned int nrRows;

};


// Implementation

template< typename T > VectorAdd< T >::VectorAdd(string dataType) : Kernel< T >("VectorAdd", dataType), nrThreadsPerBlock(0), nrThreads(0), nrRows(0) {}


template< typename T > virtual void VectorAdd< T >::generateCode() throw (OpenCLError) {
	long long unsigned int ops = static_cast< long long unsigned int >(nrThreads);
	long long unsigned int memOps = ops * 12;

	this->arInt = ops / static_cast< double >(memOps);
	this->gflop = giga(ops);
	this->gb = giga(memOps);
	
	if ( this->code != 0 ) {
		delete this->code;
	}
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(__global " + this->dataType + " *A, __global " +  this->dataType + " *B, __global " + this->dataType + " *C) {\n"
		"unsigned int id = (get_group_id(1) * get_num_groups(0) * get_local_size(0)) + (get_group_id(0) * get_local_size(0)) + get_local_id(0);\n"
		"C[id] = A[id] + B[id];\n"
		"}";

	this->compile();
}


template< typename T > void VectorAdd< T >::operator()(GPUData< T > *a, GPUData< T > *b, GPUData< T > *c) throw (OpenCLError) {
	cl::NDRange globalSize(nrThreads / nrRows, nrRows);
	cl::NDRange localSize(nrThreadsPerBlock, 1);

	this->setArgument(0, *(a->getDeviceData()));
	this->setArgument(1, *(b->getDeviceData()));
	this->setArgument(2, *(c->getDeviceData()));

	this->run(globalSize, localSize);
}


template< typename T > inline void VectorAdd< T >::setNrThreadsPerBlock(unsigned int threads) {
	nrThreadsPerBlock = threads;
}


template< typename T > inline void VectorAdd< T >::setNrThreads(unsigned int threads) {
	nrThreads = threads;
}


template< typename T > inline void VectorAdd< T >::setNrRows(unsigned int rows) {
	nrRows = rows;
}

} // OpenCL
} // isa

#endif // VECTOR_ADD_HPP

