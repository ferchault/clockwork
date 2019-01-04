# Components

## Generator
Takes a set of xyz files, writes a file in the following format:

| Field | Count | Type
|-------|------|-----
| molecule ID | 1 | int32
| number of atoms (N) | 1 | int8
| element number | N | int8
| nuclear coordinates | 3(N-1) | float32
| number of dihedrals (D) | 1 | int8
| atom indices of dihedrals | 4D| int8

The first nuclear coordinates are set to be the origin and not explicitly written to the file. This file needs to be read by the worker. For 130k molecules, this is about 15MB.

## Worker
Takes an input specification, generates geometries using a constrained geometry optimisation first, and a full relaxation second. Writes output specification for every unique confomer found.

| Field | Count | Type
|-------|------|-----
| molecule ID | 1 | int32
| dihedral ID to scan | 1 | int8
| min clockwork n | 1 | int8
| max clockwork n | 1 | int8
| number of frozen dihedrals (F) | 1 | int8
| indices of frozen dihedrals | F| int8
| clockwork n of frozen dihedrals | F | int8
| clockwork i of frozen dihedrals | F | int16

This is about 20 bytes per input specification. The output is stored as follows:

| Field | Count | Type
|-------|------|-----
| number of initialised dihedrals (D) | 1 | int8
| indices of initialised dihedrals | D | int8
| clockwork n of initialised dihedrals | D | int8
| clockwork i of initialised dihedrals | D | int16
| energy | 1 | float64
| nuclear coordinates | 3(N-1) | float32

Here, the molecule ID and the level of theory are implicitly stored in the filename. The first nuclear coordinates are set to be the origin and not explicitly written to the file. This results in about 250 bytes per conformer.
