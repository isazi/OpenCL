
SOURCE_ROOT ?= $(HOME)

UTILS := $(SOURCE_ROOT)/src/utils

CC := g++
CFLAGS := -std=c++11 -Wall
ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -O3 -g0
endif


all: bin/Exceptions.o bin/InitializeOpenCL.o bin/Kernel.o

bin/Exceptions.o: include/Exceptions.hpp src/Exceptions.cpp
	-@mkdir -p bin
	$(CC) -o bin/Exceptions.o -c src/Exceptions.cpp -I"include" $(CFLAGS)

bin/InitializeOpenCL.o: $(UTILS)/include/utils.hpp include/Exceptions.hpp include/InitializeOpenCL.hpp src/InitializeOpenCL.cpp
	-@mkdir -p bin
	$(CC) -o bin/InitializeOpenCL.o -c src/InitializeOpenCL.cpp -I"include" -I"$(UTILS)/include" $(CFLAGS)

bin/Kernel.o: $(UTILS)/include/utils.hpp include/Exceptions.hpp include/Kernel.hpp src/Kernel.cpp
	-@mkdir -p bin
	$(CC) -o bin/Kernel.o -c src/Kernel.cpp -I"include" -I"$(UTILS)/include" $(CFLAGS)

clean:
	-@rm bin/*.o

