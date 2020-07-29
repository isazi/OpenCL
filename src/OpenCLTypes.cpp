// Copyright 2019 Netherlands eScience Center
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

#include <OpenCLTypes.hpp>

namespace isa
{
namespace OpenCL
{

OpenCLError::OpenCLError(const std::string & message) : message(message) {}
	
OpenCLError::~OpenCLError() {}

const char * OpenCLError::what() const noexcept {
  return (this->message).c_str();
}

KernelConf::KernelConf() : nrThreadsD0(1), nrThreadsD1(1), nrThreadsD2(1), nrItemsD0(1), nrItemsD1(1), nrItemsD2(1), intType(0)
{}

KernelConf::~KernelConf()
{}

TuningParameters::TuningParameters() : bestMode(false), nrIterations(1), minThreads(1), maxThreads(1), maxItems(1)
{}

TuningParameters::~TuningParameters()
{}

} // OpenCL
} // isa