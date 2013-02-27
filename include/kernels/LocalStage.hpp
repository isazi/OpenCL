/*
 * Copyright (C) 2013
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
using isa::utils::toStringValue;


#ifndef LOCAL_STAGE_HPP
#define LOCAL_STAGE_HPP

namespace isa {

namespace OpenCL {

template < typename T > class LocalStage : public Kernel< T > {
public:
	LocalStage(string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(GPUData< T > * a, GPUData< T > * b) throw (OpenCLError);

	inline void setNrThreadsPerBlock(unsigned int threads);
	inline void setNrThreads(unsigned int threads);
	inline void setNrRows(unsigned int rows);

	inline void setStripe(bool stripe);

private:
	unsigned int nrThreadsPerBlock;
	unsigned int nrThreads;
	unsigned int nrRows;

	bool stripe;
};


// Implementation

template< typename T > LocalStage< T >::LocalStage(string dataType) : Kernel< T >("LocalStage", dataType), nrThreadsPerBlock(0), nrThreads(0), nrRows(0), stripe(false) {}


template< typename T > void LocalStage< T >::generateCode() throw (OpenCLError) {
	long long unsigned int ops = static_cast< long long unsigned int >(nrThreads);
	long long unsigned int memOps = ops * 8;

	this->arInt = ops / static_cast< double >(memOps);
	this->gflop = giga(ops);
	this->gb = giga(memOps);
	
	if ( this->code != 0 ) {
		delete this->code;
	}
	this->code = new string();
	if ( stripe ) {
		*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict input, __global " +  this->dataType + " * const restrict output) {\n"
			"unsigned int id = (get_group_id(1) * get_num_groups(0) * get_local_size(0)) + (get_group_id(0) * get_local_size(0)) + get_local_id(0);\n"
			"__local " + this->dataType + " stage[" + toStringValue< unsigned int >(nrThreadsPerBlock * 2) + "];\n"
			"\n"
			"stage[get_local_id(0) * 2] = input[id];\n"
			"output[id] = stage[get_local_id(0) * 2] * 2;\n"
			"}";
	}
	else {
		*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict input, __global " +  this->dataType + " * const restrict output) {\n"
			"unsigned int id = (get_group_id(1) * get_num_groups(0) * get_local_size(0)) + (get_group_id(0) * get_local_size(0)) + get_local_id(0);\n"
			"__local " + this->dataType + " stage[get_local_size(0) * 2];\n"
			"\n"
			"stage[get_local_id(0) * 1] = input[id];\n"
			"output[id] = stage[get_local_id(0) * 1] * 2;\n"
			"}";
	}

	this->compile();
}


template< typename T > void LocalStage< T >::operator()(GPUData< T > * a, GPUData< T > * b) throw (OpenCLError) {
	cl::NDRange globalSize(nrThreads / nrRows, nrRows);
	cl::NDRange localSize(nrThreadsPerBlock, 1);

	this->setArgument(0, *(a->getDeviceData()));
	this->setArgument(1, *(b->getDeviceData()));

	this->run(globalSize, localSize);
}


template< typename T > inline void LocalStage< T >::setNrThreadsPerBlock(unsigned int threads) {
	nrThreadsPerBlock = threads;
}


template< typename T > inline void LocalStage< T >::setNrThreads(unsigned int threads) {
	nrThreads = threads;
}


template< typename T > inline void LocalStage< T >::setNrRows(unsigned int rows) {
	nrRows = rows;
}


template< typename T > inline void LocalStage< T >::setStripe(bool stripe) {
	this->stripe = stripe;
}

} // OpenCL
} // isa

#endif // LOCAL_STAGE_HPP
