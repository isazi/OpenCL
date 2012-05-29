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
#include <cstring>

#include <utils.hpp>
using isa::utils::toString;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <Timer.hpp>
using LOFAR::NSTimer;


#ifndef GPU_DATA_HPP
#define GPU_DATA_HPP

namespace isa {

namespace OpenCL {

template< typename T > class GPUData {
public:
	GPUData(string name, bool deletePolicy = false);
	~GPUData();

	void allocateHostData(T *data, size_t size);
	void allocateHostData(long long unsigned int nrElements);
	void deleteHostData();
	void allocateDeviceData(cl::Buffer *data, size_t size);
	void allocateDeviceData(cl::Context *clContext) throw (OpenCLError);
	void allocateDeviceData(cl::Context *clContext, long long unsigned int nrElements) throw (OpenCLError);
	void deleteDeviceData();

	void copyHostToDevice(cl::CommandQueue *clQueue) throw (OpenCLError);
	void copyHostToDevice(cl::CommandQueue *clQueue, cl::Event *clEvent) throw (OpenCLError);
	void copyDeviceToHost(cl::CommandQueue *clQueue) throw (OpenCLError);
	void copyDeviceToHost(cl::CommandQueue *clQueue, cl::Event *clEvent) throw (OpenCLError);

	inline T *getHostData();
	inline void *getRawHostData();
	inline size_t getHostDataSize() const;
	inline cl::Buffer *getDeviceData();
	inline size_t getDeviceDataSize() const;
	inline string getName() const;

	NSTimer		timerH2D;
	NSTimer		timerD2H;

private:
	bool 		deleteHost;
	T 		*hostData;
	size_t 		hostDataSize;
	cl::Buffer 	*deviceData;
	size_t 		deviceDataSize;

	string 		name;
};


// Implementations

template< typename T > GPUData< T >::GPUData(string name, bool deletePolicy) : name(name), deleteHost(deletePolicy), hostData(0), hostDataSize(0), deviceData(0), deviceDataSize(0), timerH2D(NSTimer(name + "_H2D", false, false)), timerD2H(NSTimer(name + "_D2H", false, false)) {}


template< typename T > GPUData< T >::~GPUData() {
	deleteHostData();
	deleteDeviceData();
}


template< typename T > void GPUData< T >::allocateHostData(T *data, size_t size) {
	deleteHostData();

	hostData = data;
	hostDataSize = size;
}


template< typename T > void GPUData< T >::allocateHostData(long long unsigned int nrElements) {
	size_t newSize = nrElements * sizeof(T);

	if ( newSize != hostDataSize ) {
		deleteHostData();

		hostData = new T [nrElements];
		hostDataSize = newSize;
	}
	memset(getRawHostData(), 0, newSize);
}


template< typename T > void GPUData< T >::deleteHostData() {
	if ( deleteHost && hostDataSize != 0 ) {
		delete [] hostData;
		hostData = 0;
		hostDataSize = 0;
	}
}


template< typename T > void GPUData< T >::allocateDeviceData(cl::Buffer *data, size_t size) {
	deleteDeviceData();

	deviceData = data;
	deviceDataSize = size;
}


template< typename T > void GPUData< T >::allocateDeviceData(cl::Context *clContext) throw (OpenCLError) {
	deleteDeviceData();
	
	try {
		deviceData = new cl::Buffer(*clContext, CL_MEM_READ_WRITE, hostDataSize, NULL, NULL);
	}
	catch ( cl::Error err ) {
		throw  OpenCLError("Impossible to allocate " + name + " device memory: " + *(toString< cl_int >(err.err())));
	}
	deviceDataSize = hostDataSize;
}


template< typename T > void GPUData< T >::allocateDeviceData(cl::Context *clContext, long long unsigned int nrElements) throw (OpenCLError) {
	size_t newSize = nrElements * sizeof(T);
	if ( newSize != deviceDataSize ) {
		deleteDeviceData();

		try {
			deviceData = new cl::Buffer(*clContext, CL_MEM_READ_WRITE, newSize, NULL, NULL);
		}
		catch ( cl::Error err ) {
			throw  OpenCLError("Impossible to allocate " + name + " device memory: " + *(toString< cl_int >(err.err())));
		}
		deviceDataSize = newSize;
	}
}


template< typename T > void GPUData< T >::deleteDeviceData() {
	if ( deviceDataSize != 0 ) {
		delete deviceData;
		deviceData = 0;
		deviceDataSize = 0;
	}
}


template< typename T > void GPUData< T >::copyHostToDevice(cl::CommandQueue *clQueue) throw (OpenCLError) {
	cl::Event clEvent;
	
	if ( hostDataSize != deviceDataSize ) {
		throw OpenCLError("Impossible to copy " + name + ": different memory sizes.");
	}

	timerH2D.start();
	try {
		clQueue->enqueueWriteBuffer(*deviceData, CL_TRUE, 0, deviceDataSize, getRawHostData(), NULL, &clEvent);
		clEvent.wait();
	}
	catch ( cl::Error err ) {
		timerH2D.stop();
		throw OpenCLError("Impossible to copy " + name + " to device: " + *(toString< cl_int >(err.err())));
	}
	timerH2D.stop();
}


template< typename T > void GPUData< T >::copyHostToDevice(cl::CommandQueue *clQueue, cl::Event *clEvent) throw (OpenCLError) {
	if ( hostDataSize != deviceDataSize ) {
		throw OpenCLError("Impossible to copy " + name + ": different memory sizes.");
	}

	try {
		clQueue->enqueueWriteBuffer(*deviceData, CL_FALSE, 0, deviceDataSize, getRawHostData(), NULL, clEvent);
	}
	catch ( cl::Error err ) {
		throw OpenCLError("Impossible to copy " + name + " to device: " + *(toString< cl_int >(err.err())));
	}
}


template< typename T > void GPUData< T >::copyDeviceToHost(cl::CommandQueue *clQueue) throw (OpenCLError) {
	cl::Event clEvent;

	if ( hostDataSize != deviceDataSize ) {
		throw OpenCLError("Impossible to copy " + name + ": different memory sizes.");
	}

	timerD2H.start();
	try {
		clQueue->enqueueReadBuffer(*deviceData, CL_TRUE, 0, hostDataSize, getRawHostData(), NULL, &clEvent);
		clEvent.wait();
	}
	catch ( cl::Error err ) {
		timerD2H.stop();
		throw OpenCLError("Impossible to copy " + name + " to host: " + *(toString< cl_int >(err.err())));
	}
	timerD2H.stop();
}


template< typename T > void GPUData< T >::copyDeviceToHost(cl::CommandQueue *clQueue, cl::Event *clEvent) throw (OpenCLError) {
	if ( hostDataSize != deviceDataSize ) {
		throw OpenCLError("Impossible to copy " + name + ": different memory sizes.");
	}

	try {
		clQueue->enqueueReadBuffer(*deviceData, CL_FALSE, 0, hostDataSize, getRawHostData(), NULL, clEvent);
	}
	catch ( cl::Error err ) {
		throw OpenCLError("Impossible to copy " + name + " to host: " + *(toString< cl_int >(err.err())));
	}
}


template< typename T > inline T *GPUData< T >::getHostData() {
	return hostData;
}


template< typename T > inline void *GPUData< T >::getRawHostData() {
	return reinterpret_cast< void * >(hostData);
}


template< typename T > inline size_t GPUData< T >::getHostDataSize() const {
	return hostDataSize;
}


template< typename T > inline cl::Buffer *GPUData< T >::getDeviceData() {
	return deviceData;
}


template< typename T > inline size_t GPUData< T >::getDeviceDataSize() const {
	return deviceDataSize;
}


template< typename T > inline string GPUData< T >::getName() const {
	return name;
}

} // OpenCL
} // isa

#endif // GPU_DATA_HPP

