/*
 * Copyright (C) 2012
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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstring>

using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using std::ofstream;
using std::ceil;
using std::pow;

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <CLData.hpp>
#include <Exceptions.hpp>
#include <kernels/VectorAdd.hpp>
#include <utils.hpp>

using isa::utils::ArgumentList;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::CLData;
using isa::Exceptions::OpenCLError;
using isa::OpenCL::VectorAdd;
using isa::utils::same;


int main(int argc, char *argv[]) {
	bool vector4 = false;
	unsigned int nrIterations = 10;
	unsigned int oclPlatformID = 0;
	unsigned int device = 0;
	unsigned int arrayDim = 0;
	unsigned int nrThreads = 0;
	unsigned int nrRows = 0;

	// Parse command line
	if ( argc != 11 ) {
		cerr << "Usage: " << argv[0] << " -p <opencl_platform> -d <opencl_device> -n <dim> -t <threads> -r <rows>" << endl;
		return 1;
	}

	ArgumentList commandLine(argc, argv);
	try {
		oclPlatformID = commandLine.getSwitchArgument< unsigned int >("-p");
		device = commandLine.getSwitchArgument< unsigned int >("-d");
		arrayDim = commandLine.getSwitchArgument< unsigned int >("-n");
		nrThreads = commandLine.getSwitchArgument< unsigned int >("-t");
		nrRows = commandLine.getSwitchArgument< unsigned int >("-r");
	}
	catch ( exception &err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// Initialize OpenCL
	vector< cl::Platform > *oclPlatforms = new vector< cl::Platform >();
	cl::Context *oclContext = new cl::Context();
	vector< cl::Device > *oclDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > *oclQueues = new vector< vector< cl::CommandQueue > >();
	try {
		initializeOpenCL(oclPlatformID, 1, oclPlatforms, oclContext, oclDevices, oclQueues);
	}
	catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	CLData< float > *A = new CLData< float >("A", true);
	CLData< float > *B = new CLData< float >("B", true);
	CLData< float > *C = new CLData< float >("C", true);

	A->setCLContext(oclContext);
	A->setCLQueue(&(oclQueues->at(device)[0]));
	A->allocateHostData(arrayDim);
	B->setCLContext(oclContext);
	B->setCLQueue(&(oclQueues->at(device)[0]));
	B->allocateHostData(arrayDim);
	C->setCLContext(oclContext);
	C->setCLQueue(&(oclQueues->at(device)[0]));
	C->allocateHostData(arrayDim);
	try {
		A->setDeviceWriteOnly();
		A->allocateDeviceData();
		B->setDeviceWriteOnly();
		B->allocateDeviceData();
		C->setDeviceReadOnly();
		C->allocateDeviceData();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	VectorAdd< float > *vectorAdd = new VectorAdd< float >("float");
	try {
		vectorAdd->bindOpenCL(oclContext, &(oclDevices->at(device)), &(oclQueues->at(device)[0]));
		vectorAdd->setNrThreadsPerBlock(nrThreads);
		vectorAdd->setNrThreads(arrayDim);
		vectorAdd->setNrRows(nrRows);
		vectorAdd->generateCode();

		A->copyHostToDevice(true);
		B->copyHostToDevice(true);
		for ( unsigned int iter = 0; iter < nrIterations; iter++ ) {
			(*vectorAdd)(A, B, C);
		}
		C->copyDeviceToHost();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cout << endl;
	cout << "Time \t\t" << (vectorAdd->getTimer()).getAverageTime() << endl;
	cout << "GFLOP/s \t" << vectorAdd->getGFLOP() / (vectorAdd->getTimer()).getAverageTime() << endl;
	cout << "GB/s \t\t" << vectorAdd->getGB() / (vectorAdd->getTimer()).getAverageTime() << endl;
	cout << endl;

	for ( unsigned int item = 0; item < arrayDim; item++ ) {
		float value = (A->getHostData())[item] + (B->getHostData())[item];

		if ( ! same(value, (C->getHostData())[item]) ) {
			cerr << "Error at item " << item << "." << endl;
			return 1;
		}
	}

	cout << endl << "Test passed." << endl << endl;

	return 0;
}
