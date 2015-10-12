// Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <string>

#ifndef BITS_HPP
#define BITS_HPP

namespace isa {
namespace OpenCL {

inline std::string getBit(const std::string & value, const std::string & bit);
inline std::string setBit(const std::string & value, const std::string & newBit, const std::string & bit);


// Implementations

inline std::string getBit(const std::string & value, const std::string & bit) {
  return "convert_uchar((" + value + " >> (" + bit + ")) & 1)";
}

inline std::string setBit(const std::string & value, const std::string & newBit, const std::string & bit) {
  return value + " ^= (-(" + newBit + ") ^ " + value + ") & (1 << (" + bit + "));\n";
}

} // OpenCL
} // isa

#endif // BITS_HPP

