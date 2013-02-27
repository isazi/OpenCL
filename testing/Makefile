
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

all: clean memset vectoradd timebinning vectorcos localstage

memset: Memset.cpp
	$(LINKER) -o ~/bin/Memset $(INCLUDES) $(LIBS) Memset.cpp $(CFLAGS) $(LDFLAGS)

vectoradd: VectorAdd.cpp
	$(LINKER) -o ~/bin/VectorAdd $(INCLUDES) $(LIBS) VectorAdd.cpp $(CFLAGS) $(LDFLAGS)

timebinning: TimeBinning.cpp
	$(LINKER) -o ~/bin/TimeBinning $(INCLUDES) $(LIBS) TimeBinning.cpp $(CFLAGS) $(LDFLAGS)

vectorcos: VectorCos.cpp
	$(LINKER) -o ~/bin/VectorCos $(INCLUDES) $(LIBS) VectorCos.cpp $(CFLAGS) $(LDFLAGS)

localstage: LocalStage.cpp
	$(LINKER) -o ~/bin/LocalStage $(INCLUDES) $(LIBS) LocalStage.cpp $(CFLAGS) $(LDFLAGS)

clean:
	rm -f ~/bin/Memset
	rm -f ~/bin/VectorAdd
	rm -f ~/bin/TimeBinning
	rm -f ~/bin/VectorCos

