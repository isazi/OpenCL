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
#include <GPUData.hpp>
#include <Exceptions.hpp>
#include <kernels/VectorAdd.hpp>
#include <utils.hpp>

using isa::utils::ArgumentList;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::GPUData;
using isa::Exceptions::OpenCLError;
using isa::OpenCL::VectorAdd;
using isa::utils::same;


int main(int argc, char *argv[]) {
	bool stripe = false;
	unsigned int nrIterations = 10;
	unsigned int oclPlatformID = 0;
	unsigned int device = 0;
	unsigned int arrayDim = 0;
	unsigned int nrThreads = 0;
	unsigned int nrRows = 0;

	// Parse command line
	if ( ! ((argc == 11) || (argc == 12)) ) {
		cerr << "Usage: " << argv[0] << " [-st] -p <opencl_platform> -d <opencl_device> -n <dim> -t <threads> -r <rows>" << endl;
		return 1;
	}

	ArgumentList commandLine(argc, argv);
	try {
		stripe = commandLine.getSwitch("-st");
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

	GPUData< float > *A = new GPUData< float >("A", true);
	GPUData< float > *B = new GPUData< float >("B", true);

	A->setCLContext(oclContext);
	A->setCLQueue(&(oclQueues->at(device)[0]));
	A->allocateHostData(arrayDim);
	B->setCLContext(oclContext);
	B->setCLQueue(&(oclQueues->at(device)[0]));
	B->allocateHostData(arrayDim);
	try {
		A->allocateDeviceData();
		B->allocateDeviceData();
	}
	catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	VectorAdd< float > *vectorAdd = new VectorAdd< float >("float");
	try {
		vectorAdd->bindOpenCL(oclContext, &(oclDevices->at(device)), &(oclQueues->at(device)[0]));
		vectorAdd->setNrThreadsPerBlock(nrThreads);
		vectorAdd->setNrThreads(arrayDim);
		vectorAdd->setNrRows(nrRows);
		vectorAdd->setStripe(stripe);
		vectorAdd->generateCode();

		A->copyHostToDevice(true);
		for ( unsigned int iter = 0; iter < nrIterations; iter++ ) {
			(*vectorAdd)(A, B);
		}
		B->copyDeviceToHost();
	}
	catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cout << endl;
	cout << "Time \t\t" << (vectorAdd->getTimer()).getAverageTime() << endl;
	cout << "GFLOP/s \t" << vectorAdd->getGFLOP() / (vectorAdd->getTimer()).getAverageTime() << endl;
	cout << "GB/s \t\t" << vectorAdd->getGB() / (vectorAdd->getTimer()).getAverageTime() << endl;
	cout << endl;

	for ( unsigned int item = 0; item < arrayDim; item++ ) {
		float value = (A->getHostData())[item] * 2;

		if ( ! same(value, (B->getHostData())[item]) ) {
			cerr << "Error at item " << item << "." << endl;
			return 1;
		}
	}

	cout << endl << "Test passed." << endl << endl;

	return 0;
}
