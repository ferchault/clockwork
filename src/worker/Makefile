CC=g++
OPT= -O3
CFLAGS=-Wall -g -I ../dep/dist/include/openbabel-2.0/ -I ../dep/dist/include/ -fopenmp
LDFLAGS=-L ../dep/dist/lib/ -lgomp -lopenbabel
LDPOSTFLAGS=-lhiredis
MATHFLAGS=-lblas -llapack -llapacke

all: worker

worker: main.cpp
	$(CC) $(CFLAGS) $(MATHFLAGS) $(LDFLAGS) main.cpp -o worker $(LDPOSTFLAGS)

test: poc_rmsd.o poc_conformations.o

%.o: %.cpp
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $(MATHFLAGS) $<

clean:
	rm -rf *.o worker
