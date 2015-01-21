
UTILS := $(HOME)/src/utils

CC := g++
CFLAGS := -std=c++11 -Wall
ifneq ($(DEBUG), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif


all: Exceptions.o InitializeOpenCL.o Kernel.o

Exceptions.o: include/Exceptions.hpp src/Exceptions.cpp
	$(CC) -o bin/Exceptions.o -c src/Exceptions.cpp -I"include" $(CFLAGS)

InitializeOpenCL.o: $(UTILS)/include/utils.hpp include/Exceptions.hpp include/InitializeOpenCL.hpp src/InitializeOpenCL.cpp
	$(CC) -o bin/InitializeOpenCL.o -c src/InitializeOpenCL.cpp -I"include" -I"$(UTILS)/include" $(CFLAGS)

Kernel.o: $(UTILS)/include/utils.hpp include/Exceptions.hpp include/Kernel.hpp src/Kernel.cpp
	$(CC) -o bin/Kernel.o -c src/Kernel.cpp -I"include" -I"$(UTILS)/include" $(CFLAGS)

clean:
	rm bin/*.o

