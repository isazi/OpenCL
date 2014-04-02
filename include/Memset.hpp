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
#include <string>
#include <utility>

using std::string;
using std::make_pair;

#include <Kernel.hpp>
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

