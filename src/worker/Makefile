CC=g++
CFLAGS=-Wall -g -c -I dep/dist/include/openbabel-2.0/ -O3 -fopenmp
LDFLAGS=-L dep/dist/lib/ -lgomp -lopenbabel

all: worker

worker: worker.o
	$(CC) $(LDFLAGS) worker.o -o worker

worker.o: main.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) main.cpp

clean:
	rm -rf worker.o worker
