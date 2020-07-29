// Copyright 2015 Alessio Sclocco <alessio@sclocco.eu>
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


#pragma once

namespace isa
{
namespace OpenCL
{
/**
 ** @brief Return the OpenCL code to read a single bit.
 **
 ** @param value The name of the variable containing the bit to read.
 ** @param bit A number representing the position of the bit to read.
 **
 ** @return A string containing the OpenCL code.
 */
std::string getBit(const std::string & value, const std::string & bit);
/**
 ** @brief Return the OpenCL code to write a single bit.
 **
 ** @param value The name of the variable containing the bit to write.
 ** @param newBit The new value for the bit.
 ** @param bit A number representing the position of the bit to write.
 **
 ** @return A string containing the OpenCL code.
 */
std::string setBit(const std::string & value, const std::string & newBit, const std::string & bit);

inline std::string getBit(const std::string & value, const std::string & bit) {
    return "((" + value + " >> (" + bit + ")) & 1)";
}

inline std::string setBit(const std::string & value, const std::string & newBit, const std::string & bit) {
    return value + " ^= (-(" + newBit + ") ^ " + value + ") & (1 << (" + bit + "));\n";
}
} // OpenCL
} // isa

