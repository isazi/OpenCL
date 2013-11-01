
INCLUDES 	:= -I"${OPENCL_INCLUDE}" -I"/home/alessio/src/OpenCL/include" -I"/home/alessio/src/utils/include"
LIBS		:= -L"${OPENCL_LIB}"

ifneq ($(debug), 1)
	CFLAGS		:= -std=c++0x -O3 -g0 -w -msse3 -fopenmp
else
	CFLAGS		:= -std=c++0x -O0 -g3 -Wall -msse3 -fopenmp
endif

LDFLAGS         := -lm -lOpenCL

CC		:= g++
LINKER		:= g++

all: clean memset

memset: Memset.cpp
	$(LINKER) -o ~/bin/Memset $(INCLUDES) $(LIBS) Memset.cpp $(CFLAGS) $(LDFLAGS)

clean:
	rm -f ~/bin/Memset


