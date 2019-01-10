CC=g++
OPT= -O3
CFLAGS=-Wall -g -I ../dep/dist/include/openbabel-2.0/ -I ../dep/dist/include/ -fopenmp -std=c++11
LDFLAGS=-L ../dep/dist/lib/ -lgomp -lopenbabel
NETWORK=-lhiredis
LDPOSTFLAGS=-lcblas -llapacke -llapack -lblas -lgfortran
MATHFLAGS=

all: worker

worker: main.cpp
	$(CC) $(CFLAGS) $(MATHFLAGS) $(LDFLAGS) main.cpp -o worker $(LDPOSTFLAGS) $(NETWORK)

test: poc_rmsd.o poc_obmol.o poc_conformations.o

poc_rmsd.o: molecule.cpp
poc_obmol.o: molecule.cpp
poc_conformations.o: molecule.cpp

%.o: %.cpp
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $(MATHFLAGS) $< $(LDPOSTFLAGS)

clean:
	rm -rf *.o worker
