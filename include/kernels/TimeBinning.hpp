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
using isa::utils::toString;


#ifndef TIME_BINNING_HPP
#define TIME_BINNING_HPP

namespace isa {

namespace OpenCL {

template < typename T > class TimeBinning : public Kernel< T > {
public:
	TimeBinning(string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(GPUData< T > *a, GPUData< T > *b) throw (OpenCLError);

	inline void setNrThreadsPerBlock(unsigned int threads);
	inline void setNrThreads(unsigned int threads);
	inline void setNrRows(unsigned int rows);

	inline void setBinFactor(unsigned int bin);

private:
	unsigned int nrThreadsPerBlock;
	unsigned int nrThreads;
	unsigned int nrRows;

	unsigned int binFactor;
};


// Implementation

template< typename T > TimeBinning< T >::TimeBinning(string dataType) : Kernel< T >("TimeBinning", dataType), nrThreadsPerBlock(0), nrThreads(0), nrRows(0), binFactor(0) {}


template< typename T > void TimeBinning< T >::generateCode() throw (OpenCLError) {
	long long unsigned int ops = static_cast< long long unsigned int >(nrThreads * binFactor);
	long long unsigned int memOps = (ops * 4) + (nrThreads * 4);

	this->arInt = ops / static_cast< double >(memOps);
	this->gflop = giga(ops);
	this->gb = giga(memOps);

	string *binFactor_s = toString< unsigned int >(binFactor);

	if ( this->code != 0 ) {
		delete this->code;
	}
	this->code = new string();
	if ( binFactor == 4 ) {
		*(this->code) = "__kernel void " + this->name + "(__global " + this->dataType + "4 *A, __global " +  this->dataType + " *B) {\n"
			"unsigned int id = (get_group_id(1) * get_num_groups(0) * get_local_size(0)) + (get_group_id(0) * get_local_size(0)) + get_local_id(0);\n"
			+ this->dataType + "4 sample = 0;\n"
			+ this->dataType + " acc = 0;\n"
			""
			"sample = A[id];\n"
			"acc += sample.x;\n"
			"acc += sample.y;\n"
			"acc += sample.z;\n"
			"acc += sample.w;\n"
			"B[id] = acc;\n"
			"}";
	}
	else {
		*(this->code) = "__kernel void " + this->name + "(__global " + this->dataType + " *A, __global " +  this->dataType + " *B) {\n"
			"unsigned int idIn = (get_group_id(1) * get_num_groups(0) * get_local_size(0) * " + *binFactor_s + ") + (get_group_id(0) * get_local_size(0) * " + *binFactor_s + ") + (get_local_id(0) * " + *binFactor_s + ");\n"
			"unsigned int idOut = (get_group_id(1) * get_num_groups(0) * get_local_size(0)) + (get_group_id(0) * get_local_size(0)) + get_local_id(0);\n"
			+ this->dataType + " acc = 0;\n"
			""
			"for ( unsigned int sample = 0; sample < " + *binFactor_s +  "; sample++ ) {\n"
			"acc += A[idIn + sample];\n"
			"}\n"
			"B[idOut] = acc;\n"
			"}";
	}

	delete binFactor_s;

	this->compile();
}


template< typename T > void TimeBinning< T >::operator()(GPUData< T > *a, GPUData< T > *b) throw (OpenCLError) {
	cl::NDRange globalSize(nrThreads / nrRows, nrRows);
	cl::NDRange localSize(nrThreadsPerBlock, 1);

	this->setArgument(0, *(a->getDeviceData()));
	this->setArgument(1, *(b->getDeviceData()));

	this->run(globalSize, localSize);
}


template< typename T > inline void TimeBinning< T >::setNrThreadsPerBlock(unsigned int threads) {
	nrThreadsPerBlock = threads;
}


template< typename T > inline void TimeBinning< T >::setNrThreads(unsigned int threads) {
	nrThreads = threads;
}


template< typename T > inline void TimeBinning< T >::setNrRows(unsigned int rows) {
	nrRows = rows;
}


template< typename T > inline void TimeBinning< T >::setBinFactor(unsigned int bin) {
	binFactor = bin;
}

} // OpenCL
} // isa

#endif // TIME_BINNING_HPP

