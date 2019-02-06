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

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <vector>
#include <string>


#pragma once

namespace isa
{
namespace OpenCL
{

/**
 ** @brief Data structure containing all the necessary OpenCL runtime objects.
 */
struct OpenCLRunTime
{
    cl::Context * context = nullptr;
    std::vector<cl::Platform> * platforms = nullptr;
    std::vector<cl::Device> * devices = nullptr;
    std::vector<std::vector<cl::CommandQueue>> * queues = nullptr;
};

/**
 ** @brief OpenCL error.
 */
class OpenCLError : public std::exception
{
public:
	explicit OpenCLError(const std::string & message);
	~OpenCLError();
	const char *what() const noexcept;

private:
    std::string message;
};

/**
 ** @brief Basic OpenCL kernel configuration.
 */
class KernelConf
{
  public:
    KernelConf();
    ~KernelConf();
    /**
     ** @brief Get the integer type used in the kernel.
     ** This type is not the one associated with data, but the one used for internal variables.
     **
     ** @return 0 for "int", 1 for "unsigned int".
     */
    std::string getIntType() const;
    unsigned int getNrThreadsD0() const;
    unsigned int getNrThreadsD1() const;
    unsigned int getNrThreadsD2() const;
    unsigned int getNrItemsD0() const;
    unsigned int getNrItemsD1() const;
    unsigned int getNrItemsD2() const;
    /**
     ** @brief Set the integer type to use in the kernel.
     ** This type is not the one associated with data, but the one used for internal variables.
     **
     ** @param Integer representing the type: 0 is "int", 1 is "unsigned int".
     */
    void setIntType(const unsigned int type);
    void setNrThreadsD0(const unsigned int threads);
    void setNrThreadsD1(const unsigned int threads);
    void setNrThreadsD2(const unsigned int threads);
    void setNrItemsD0(const unsigned int items);
    void setNrItemsD1(const unsigned int items);
    void setNrItemsD2(const unsigned int items);
    // utils
    std::string print() const;

  private:
    unsigned int nrThreadsD0, nrThreadsD1, nrThreadsD2;
    unsigned int nrItemsD0, nrItemsD1, nrItemsD2;
    unsigned int intType;
};

/**
 ** @brief Basic kernel tuning constraints.
 */
class TuningParameters
{
public:
    TuningParameters();
    ~TuningParameters();
    /**
     ** @brief Set the best mode to true or false.
     ** The best mode is a tuning mode in which only the best configuration is returned by the tuner.
     **
     ** @param mode The value for the best mode, either true or false.
     */
    void setBestMode(const bool mode);
    /**
     ** @brief Retrieve the status of the best mode.
     ** The best mode is a tuning mode in which only the best configuration is returned by the tuner.
     **
     ** @return The status of the best mode.
     */
    bool getBestMode() const;
    /**
     ** @brief Set the number of iterations for each kernel configuration.
     **
     ** @param iterations the number of iterations.
     */
    void setNrIterations(const unsigned int iterations);
    /**
     ** @brief Retrieve the number of iterations for each kernel configuration.
     **
     ** @return The number of iterations.
     */
    unsigned int getNrIterations() const;
    /**
     ** @brief Set the minimum number of threads to use.
     **
     ** @param threads The minimum number of threads.
     */
    void setMinThreads(const unsigned int threads);
    /**
     ** @brief Retrieve the minimum number of threads to use.
     **
     ** @return The minimum number of threads.
     */
    unsigned int getMinThreads() const;
    /**
     ** @brief Set the maximum number of threads to use.
     **
     ** @param threads The maximum number of threads.
     */
    void setMaxThreads(const unsigned int threads);
    /**
     ** @brief Retrieve the maximum number of threads to use.
     **
     ** @return The maximum number of threads.
     */
    unsigned int getMaxThreads() const;
    /**
     ** @brief Set the maximum number of items to declare per kernel.
     ** Any variable allocated in the OpenCL code is one item.
     **
     ** @param items The maximum number of items.
     */
    void setMaxItems(const unsigned int items);
    /**
     ** @brief Retrieve the maximum number of items to use.
     ** Any variable allocated in the OpenCL code is one item.
     **
     ** @return The maximum number of items.
     */
    unsigned int getMaxItems() const;
private:
    bool bestMode;
    unsigned int nrIterations;
    unsigned int minThreads;
    unsigned int maxThreads;
    unsigned int maxItems;
};

inline std::string KernelConf::print() const
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

inline void KernelConf::setIntType(const unsigned int type)
{
    intType = type;
}

inline void KernelConf::setNrThreadsD0(const unsigned int threads)
{
    nrThreadsD0 = threads;
}

inline void KernelConf::setNrThreadsD1(const unsigned int threads)
{
    nrThreadsD1 = threads;
}

inline void KernelConf::setNrThreadsD2(const unsigned int threads)
{
    nrThreadsD2 = threads;
}

inline void KernelConf::setNrItemsD0(const unsigned int items)
{
    nrItemsD0 = items;
}

inline void KernelConf::setNrItemsD1(const unsigned int items)
{
    nrItemsD1 = items;
}

inline void KernelConf::setNrItemsD2(const unsigned int items)
{
    nrItemsD2 = items;
}

inline void TuningParameters::setBestMode(const bool mode)
{
    bestMode = mode;
}

inline bool TuningParameters::getBestMode() const
{
    return bestMode;
}

inline void TuningParameters::setNrIterations(const unsigned int iterations)
{
    nrIterations = iterations;
}


inline unsigned int TuningParameters::getNrIterations() const
{
    return nrIterations;
}

inline void TuningParameters::setMinThreads(const unsigned int threads)
{
    minThreads = threads;
}

inline unsigned int TuningParameters::getMinThreads() const
{
    return minThreads;
}

inline void TuningParameters::setMaxThreads(const unsigned int threads)
{
    maxThreads = threads;
}

inline unsigned int TuningParameters::getMaxThreads() const
{
    return maxThreads;
}

inline void TuningParameters::setMaxItems(const unsigned int items)
{
    maxItems = items;
}

inline unsigned int TuningParameters::getMaxItems() const
{
    return maxItems;
}

} // OpenCL
} // isa