CC=g++
OPT= -O3
CFLAGS=-Wall -g -I ../dep/dist/include/openbabel-2.0/ -I ../dep/dist/include/ -fopenmp
LDFLAGS=-L ../dep/dist/lib/ -lgomp -lopenbabel
LDPOSTFLAGS=-lhiredis
JCKFLAGS=-lblas -llapack -llapacke

all: worker

worker: main.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) main.cpp -o worker $(LDPOSTFLAGS)

test.o: rmsd_poc.cpp kabsch.h
	$(CC) -o test.o $(CFLAGS) $(LDFLAGS) $(JCKFLAGS) $<

clean:
	rm -rf main.o worker
