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
#include <cstring>
#include <fstream>
using std::string;
using std::ofstream;

#include <utils.hpp>
#include <Exceptions.hpp>
#include <Timer.hpp>
using isa::utils::toStringValue;
using isa::Exceptions::OpenCLError;
using LOFAR::NSTimer;


#ifndef GPU_DATA_HPP
#define GPU_DATA_HPP

namespace isa {

namespace OpenCL {

template< typename T > class GPUData {
public:
	GPUData(string name, bool deletePolicy = false, bool devRO = false);
	~GPUData();
	
	inline string getName() const;

	// Allocation of host data
	void allocateHostData(T *data, size_t size);
	void allocateHostData(long long unsigned int nrElements);
	void deleteHostData();
	
	// Allocation of device data
	void allocateDeviceData(cl::Buffer *data, size_t size);
	void allocateDeviceData(long long unsigned int nrElements) throw (OpenCLError);
	void allocateDeviceData() throw (OpenCLError);
	void deleteDeviceData();

	// Memory transfers
	void copyHostToDevice(bool async = false) throw (OpenCLError);
	void copyDeviceToHost(bool async = false) throw (OpenCLError);
	void dumpDeviceToDisk() throw (OpenCLError);

	// OpenCL
	inline void setCLContext(cl::Context *context);
	inline void setCLQueue(cl::CommandQueue *queue);
	
	// Access host data
	inline T *getHostData();
	inline T *getHostDataAt(long long unsigned int startintPoint);
	inline void *getRawHostData();
	inline size_t getHostDataSize() const;
	inline T operator[](long long unsigned int item);

	// Access device data
	inline cl::Buffer *getDeviceData();
	inline size_t getDeviceDataSize() const;
	
	
	// Timers
	inline double getH2DTime() const;
	inline double getD2HTime() const;

private:
	cl::Context *clContext;
	cl::CommandQueue *clQueue;
	NSTimer timerH2D;
	NSTimer	timerD2H;

	bool deleteHost;
	bool deviceReadOnly;
	T *hostData;
	size_t hostDataSize;
	cl::Buffer *deviceData;
	size_t deviceDataSize;

	string name;
};


// Implementations

template< typename T > GPUData< T >::GPUData(string name, bool deletePolicy, bool devRO) : clContext(0), clQueue(0), timerH2D(NSTimer(name + "_H2D", false, false)), timerD2H(NSTimer(name + "_D2H", false, false)), deleteHost(deletePolicy), deviceReadOnly(devRO), hostData(0), hostDataSize(0), deviceData(0), deviceDataSize(0), name(name) {}


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


template< typename T > void GPUData< T >::allocateDeviceData(long long unsigned int nrElements) throw (OpenCLError) {
	size_t newSize = nrElements * sizeof(T);
	if ( newSize != deviceDataSize ) {
		deleteDeviceData();

		try {
			if ( deviceReadOnly ) {
				deviceData = new cl::Buffer(*clContext, CL_MEM_READ_ONLY, newSize, NULL, NULL);
			}
			else {
				deviceData = new cl::Buffer(*clContext, CL_MEM_READ_WRITE, newSize, NULL, NULL);
			}
		}
		catch ( cl::Error err ) {
			throw  OpenCLError("Impossible to allocate " + name + " device memory: " + toStringValue< cl_int >(err.err()));
		}
		deviceDataSize = newSize;
	}
}


template< typename T > void GPUData< T >::allocateDeviceData() throw (OpenCLError) {
	deleteDeviceData();
	
	try {
		if ( deviceReadOnly ) {
			deviceData = new cl::Buffer(*clContext, CL_MEM_READ_ONLY, hostDataSize, NULL, NULL);
		}
		else {
			deviceData = new cl::Buffer(*clContext, CL_MEM_READ_WRITE, hostDataSize, NULL, NULL);
		}
	}
	catch ( cl::Error err ) {
		throw  OpenCLError("Impossible to allocate " + name + " device memory: " + toStringValue< cl_int >(err.err()));
	}
	deviceDataSize = hostDataSize;
}


template< typename T > void GPUData< T >::deleteDeviceData() {
	if ( deviceDataSize != 0 ) {
		delete deviceData;
		deviceData = 0;
		deviceDataSize = 0;
	}
}


template< typename T > void GPUData< T >::copyHostToDevice(bool async) throw (OpenCLError) {
	if ( hostDataSize != deviceDataSize ) {
		throw OpenCLError("Impossible to copy " + name + ": different memory sizes.");
	}

	if ( async ) {
		try {
			clQueue->enqueueWriteBuffer(*deviceData, CL_FALSE, 0, deviceDataSize, getRawHostData(), NULL, NULL);
		}
		catch ( cl::Error err ) {
			throw OpenCLError("Impossible to copy " + name + " to device: " + toStringValue< cl_int >(err.err()));
		}
	}
	else {
		cl::Event clEvent;

		try {
			timerH2D.start();
			clQueue->enqueueWriteBuffer(*deviceData, CL_TRUE, 0, deviceDataSize, getRawHostData(), NULL, &clEvent);
			clEvent.wait();
			timerH2D.stop();
		}
		catch ( cl::Error err ) {
			timerH2D.reset();
			throw OpenCLError("Impossible to copy " + name + " to device: " + toStringValue< cl_int >(err.err()));
		}
	}
}


template< typename T > void GPUData< T >::copyDeviceToHost(bool async) throw (OpenCLError) {
	if ( hostDataSize != deviceDataSize ) {
		throw OpenCLError("Impossible to copy " + name + ": different memory sizes.");
	}

	if ( async ) {
		try {
			clQueue->enqueueReadBuffer(*deviceData, CL_FALSE, 0, hostDataSize, getRawHostData(), NULL, NULL);
		}
		catch ( cl::Error err ) {
			throw OpenCLError("Impossible to copy " + name + " to host: " + toStringValue< cl_int >(err.err()));
		}
	}
	else {
		cl::Event clEvent;

		try {
			timerD2H.start();
			clQueue->enqueueReadBuffer(*deviceData, CL_TRUE, 0, hostDataSize, getRawHostData(), NULL, &clEvent);
			clEvent.wait();
			timerD2H.stop();
		}
		catch ( cl::Error err ) {
			timerD2H.reset();
			throw OpenCLError("Impossible to copy " + name + " to host: " + toStringValue< cl_int >(err.err()));
		}
	}

}


template< typename T > void GPUData< T >::dumpDeviceToDisk() throw (OpenCLError) {
	GPUData< T > temp = GPUData< T >("temp", true);

	temp.setCLContext(clContext);
	temp.setCLQueue(clQueue);
	temp.allocateHostData(hostDataSize / sizeof(T));
	temp.allocateDeviceData(deviceData, deviceDataSize);
	temp.copyDeviceToHost();

	ofstream oFile(("./" + name + ".bin").c_str(), ofstream::binary);
	oFile.write(reinterpret_cast< char * >(temp.getRawHostData()), temp.getHostDataSize());
	oFile.close();
}

template< typename T > inline void GPUData< T >::setCLContext(cl::Context *context) {
	clContext = context;
}

	
template< typename T > inline void GPUData< T >::setCLQueue(cl::CommandQueue *queue) {
	clQueue = queue;
}


template< typename T > inline T *GPUData< T >::getHostData() {
	return hostData;
}


template< typename T > inline T *GPUData< T >::getHostDataAt(long long unsigned int startingPoint) {
	return &(hostData[startingPoint])
}


template< typename T > inline void *GPUData< T >::getRawHostData() {
	return reinterpret_cast< void * >(hostData);
}


template< typename T > inline size_t GPUData< T >::getHostDataSize() const {
	return hostDataSize;
}


template< typename T > inline T GPUData< T >::operator[](long long unsigned int item) {
	return hostData[item];
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

	
template< typename T > inline double GPUData< T >::getH2DTime() const {
	return timerH2D.getElapsed();
}


template< typename T > inline double GPUData< T >::getD2HTime() const {
	return timerD2H.getElapsed();
}

} // OpenCL
} // isa

#endif // GPU_DATA_HPP

