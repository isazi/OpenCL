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

KernelConf::KernelConf() : nrThreadsD0(1), nrThreadsD1(1), nrThreadsD2(1), nrItemsD0(1), nrItemsD1(1), nrItemsD2(1), intType(0) {}

KernelConf::~KernelConf() {}

std::string KernelConf::print() const
{
    return std::to_string(nrThreadsD0) + " " + std::to_string(nrThreadsD1) + " " + std::to_string(nrThreadsD2) + " " + std::to_string(nrItemsD0) + " " + std::to_string(nrItemsD1) + " " + std::to_string(nrItemsD2) + " " + std::to_string(intType);
}

inline std::string KernelConf::getIntType() const
{
    if ( intType == 0 )
    {
        return "int";
    }
    else if ( intType == 1 )
    {
        return "unsigned int";
    }
    return "int";
}

inline unsigned int KernelConf::getNrThreadsD0() const
{
    return nrThreadsD0;
}

inline unsigned int KernelConf::getNrThreadsD1() const
{
    return nrThreadsD1;
}

inline unsigned int KernelConf::getNrThreadsD2() const
{
    return nrThreadsD2;
}

inline unsigned int KernelConf::getNrItemsD0() const
{
    return nrItemsD0;
}

inline unsigned int KernelConf::getNrItemsD1() const
{
    return nrItemsD1;
}

inline unsigned int KernelConf::getNrItemsD2() const
{
    return nrItemsD2;
}

inline void KernelConf::setIntType(unsigned int type)
{
    intType = type;
}

inline void KernelConf::setNrThreadsD0(unsigned int threads)
{
    nrThreadsD0 = threads;
}

inline void KernelConf::setNrThreadsD1(unsigned int threads)
{
    nrThreadsD1 = threads;
}

inline void KernelConf::setNrThreadsD2(unsigned int threads)
{
    nrThreadsD2 = threads;
}

inline void KernelConf::setNrItemsD0(unsigned int items)
{
    nrItemsD0 = items;
}

inline void KernelConf::setNrItemsD1(unsigned int items)
{
    nrItemsD1 = items;
}

inline void KernelConf::setNrItemsD2(unsigned int items)
{
    nrItemsD2 = items;
}
} // OpenCL
} // isa