// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
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
#include <exception>
#include <typeinfo>

#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

namespace isa {
namespace OpenCL {

// Generic OpenCL exception to incapsulate the error message
class OpenCLError : public std::exception {
public:
	OpenCLError(std::string message);

	const char *what() const throw ();

private:
  std::string message;
};


// Implementations
OpenCLError::OpenCLError(std::string message) : message(message) {}

const char * OpenCLError::what() const throw () {
  return (this->message).c_str();
}

} // OpenCL
} // isa
#endif

