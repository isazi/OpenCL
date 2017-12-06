// Copyright 2012 Alessio Sclocco <a.sclocco@vu.nl>
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
#include <vector>
#include <utils.hpp>

#include "Exceptions.hpp"


#pragma once

namespace isa {
namespace OpenCL {

// Class that represents the configuration of an OpenCL kernel
class KernelConf {
public:
  KernelConf();
  ~KernelConf();
  // Get
  inline unsigned int getNrThreadsD0() const;
  inline unsigned int getNrThreadsD1() const;
  inline unsigned int getNrThreadsD2() const;
  inline unsigned int getNrItemsD0() const;
  inline unsigned int getNrItemsD1() const;
  inline unsigned int getNrItemsD2() const;
  // Set
  inline void setNrThreadsD0(unsigned int threads);
  inline void setNrThreadsD1(unsigned int threads);
  inline void setNrThreadsD2(unsigned int threads);
  inline void setNrItemsD0(unsigned int items);
  inline void setNrItemsD1(unsigned int items);
  inline void setNrItemsD2(unsigned int items);
  // utils
  std::string print() const;

private:
  unsigned int nrThreadsD0, nrThreadsD1, nrThreadsD2;
  unsigned int nrItemsD0, nrItemsD1, nrItemsD2;
};

cl::Kernel * compile(const std::string & name, const std::string & code, const std::string & flags, cl::Context & clContext, cl::Device & clDevice);


// Implementations
inline unsigned int KernelConf::getNrThreadsD0() const {
  return nrThreadsD0;
}

inline unsigned int KernelConf::getNrThreadsD1() const {
  return nrThreadsD1;
}

inline unsigned int KernelConf::getNrThreadsD2() const {
  return nrThreadsD2;
}

inline unsigned int KernelConf::getNrItemsD0() const {
  return nrItemsD0;
}

inline unsigned int KernelConf::getNrItemsD1() const {
  return nrItemsD1;
}

inline unsigned int KernelConf::getNrItemsD2() const {
  return nrItemsD2;
}

inline void KernelConf::setNrThreadsD0(unsigned int threads) {
  nrThreadsD0 = threads;
}

inline void KernelConf::setNrThreadsD1(unsigned int threads) {
  nrThreadsD1 = threads;
}

inline void KernelConf::setNrThreadsD2(unsigned int threads) {
  nrThreadsD2 = threads;
}

inline void KernelConf::setNrItemsD0(unsigned int items) {
  nrItemsD0 = items;
}

inline void KernelConf::setNrItemsD1(unsigned int items) {
  nrItemsD1 = items;
}

inline void KernelConf::setNrItemsD2(unsigned int items) {
  nrItemsD2 = items;
}

} // OpenCL
} // isa

