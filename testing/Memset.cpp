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
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
#include <iomanip>
using std::setprecision;
#include <fstream>
using std::ofstream;
#include <cmath>
using std::ceil;
using std::pow;
#include <cstring>

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <GPUData.hpp>
using isa::OpenCL::GPUData;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <kernels/Memset.hpp>
using isa::OpenCL::Memset;


int main(int argc, char *argv[]) {
	int value = 0;
	unsigned int oclPlatformID = 0;
	unsigned int device = 0;
	unsigned int arrayDim = 0;
	unsigned int nrThreads = 0;
	unsigned int nrRows = 0;
	GPUData< int > *data = new GPUData< int >("data", true);
	Memset< int > *memset = new Memset< int >("int");

	// Parse command line
	if ( argc != 13 ) {
		cerr << "Usage: " << argv[0] << " -p <opencl_platform> -d <opencl_device> -v <value> -n <dim> -t <threads< -r <rows>" << endl;
		return 1;
	}

	ArgumentList commandLine(argc, argv);
	try {
		oclPlatformID = commandLine.getSwitchArgument< unsigned int >("-p");
		device = commandLine.getSwitchArgument< unsigned int >("-d");
		value = commandLine.getSwitchArgument< int >("-v");
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

	data->setCLContext(oclContext);
	data->setCLQueue(&(oclQueues->at(device)[0]));
	data->allocateHostData(arrayDim);
	try {
		data->allocateDeviceData();
	}
	catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	try {
		memset->compile(*oclContext, oclDevices->at(device), &(oclQueues->at(device)[0]));
		memset->setNrThreads(nrThreads);
		memset->setNrBlocks(arrayDim / nrThreads);
		memset->setNrRows(nrRows);

		data->copyHostToDevice(true);
		memset->run(value, data);
		data->copyDeviceToHost();
	}
	catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	for ( unsigned int item = 0; item < arrayDim; item++ ) {
		if ( (data->getHostData())[item] != value ) {
			cerr << "Error at item " << item << "." << endl;
			return 1;
		}
	}

	cout << endl << "Test passed." << endl << endl;

	return 0;
}

