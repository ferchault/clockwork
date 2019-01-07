CC=g++
OPT= -O3
CFLAGS=-Wall -g -I ../dep/dist/include/openbabel-2.0/ -fopenmp
LDFLAGS=-L dep/dist/lib/ -lgomp -lopenbabel
JCKFLAGS=-lblas -llapack -llapacke

all: worker

worker: worker.o
	$(CC) $(LDFLAGS) worker.o -o worker

worker.o: main.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) main.cpp

test.o: rmsd_poc.cpp kabsch.h
	$(CC) -o test.o $(CFLAGS) $(LDFLAGS) $(JCKFLAGS) $<

clean:
	rm -rf worker.o worker
