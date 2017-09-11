
SOURCE_ROOT ?= $(HOME)

CC := g++
CFLAGS := -std=c++11 -Wall
ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -O3 -g0
endif


all: bin/Exceptions.o bin/InitializeOpenCL.o bin/Kernel.o
	-@mkdir -p lib
	$(CC) -o lib/libisaOpenCL.so -shared -Wl,-soname,libisaOpenCL.so bin/Exceptions.so bin/InitializeOpenCL.o bin/Kernel.o $(CFLAGS)

bin/Exceptions.o: include/Exceptions.hpp src/Exceptions.cpp
	-@mkdir -p bin
	$(CC) -o bin/Exceptions.o -c -fpic src/Exceptions.cpp -I"include" $(CFLAGS)

bin/InitializeOpenCL.o: $(SOURCE_ROOT)/include/utils.hpp include/Exceptions.hpp include/InitializeOpenCL.hpp src/InitializeOpenCL.cpp
	-@mkdir -p bin
	$(CC) -o bin/InitializeOpenCL.o -c -fpic src/InitializeOpenCL.cpp -I"include" -I"$(SOURCE_ROOT)/include" $(CFLAGS)

bin/Kernel.o: $(SOURCE_ROOT)/include/utils.hpp include/Exceptions.hpp include/Kernel.hpp src/Kernel.cpp
	-@mkdir -p bin
	$(CC) -o bin/Kernel.o -c -fpic src/Kernel.cpp -I"include" -I"$(SOURCE_ROOT)/include" $(CFLAGS)

clean:
	-@rm bin/*.o
	-@rm lib/*

install: all
	-@cp include/* $(SOURCE_ROOT)/include
	-@cp lib/* $(SOURCE_ROOT)/lib
