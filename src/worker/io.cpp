#include <iostream>
#include <vector>
#include <map>

#define DEBUG

class Archive {
	std::vector<unsigned int> molecule_ids;
	std::map<unsigned int, unsigned int> number_of_atoms;
	std::map<unsigned int, unsigned int*> element_numbers;
	std::map<unsigned int, double**> coordinates;
	std::map<unsigned int, unsigned int> number_of_bonds;
	std::map<unsigned int, unsigned int**> bonds;
	std::map<unsigned int, unsigned int*> bond_orders;
	std::map<unsigned int, unsigned int> number_of_dihedrals;
	std::map<unsigned int, unsigned int**> dihedrals;
public:
	int read_archive(const char * filename);
};

int Archive::read_archive(const char * filename) {
	std::ifstream fh;
	fh.open(filename, std::ios::in | std::ios::binary);

	if (!fh.is_open()) {
		std::cout << "Unable to open file " << filename << std::endl;
		return 1;
	}

	char * buffer1 = new char[1];
	char * buffer4 = new char[4];
	char * buffer8 = new char[8];
	unsigned int atomcount, elementid;
	unsigned int molecule_id;
	while (!fh.eof()) {
		// read molecule id
		fh.read(buffer4, 4);
		molecule_id = *(unsigned int*)buffer4;
		#ifdef DEBUG
			std::cout << "Reading molecule " << molecule_id << std::endl;
		#endif
		this->molecule_ids.push_back(molecule_id);

		// read number of atoms
		fh.read(buffer1, 1);
		atomcount = *(unsigned int*)buffer1;
		#ifdef DEBUG
			std::cout << " > has " << atomcount << " atoms" << std::endl;
		#endif
		number_of_atoms.insert(std::pair<unsigned int, unsigned int>(molecule_id, atomcount));

		// read element numbers
		unsigned int * numbers = new unsigned int[atomcount];
		#ifdef DEBUG
			std::cout << " > atoms have these atomic numbers:" << std::endl << " >> ";
		#endif
		for (unsigned int atom = 0; atom < atomcount; atom++) {
			fh.read(buffer1, 1);
			elementid = *(unsigned int*)buffer1;
			#ifdef DEBUG
				std::cout << elementid << ", ";
			#endif
			numbers[atom] = elementid;
		}
		#ifdef DEBUG
			std::cout << std::endl;
		#endif
		element_numbers.insert(std::pair<unsigned int, unsigned int*>(molecule_id, numbers));

		// read nuclear coordinates
		double * coordinates = new double[atomcount*3];
		coordinates[0] = 0;
		coordinates[1] = 0;
		coordinates[2] = 0;
		#ifdef DEBUG
			std::cout << " > atoms have these coordinates: " << std::endl;
		#endif
		for (unsigned int atom = 1; atom < atomcount; atom++) {
			for (unsigned int dimension = 0; dimension < 3; dimension++) {
				fh.read(buffer8, 8);
				coordinates[atom * 3 + dimension] = *(double*)buffer8;
			}
			#ifdef DEBUG
				std::cout << " > atom " << atom << ": " << coordinates[atom * 3 + 0] << ", " << coordinates[atom * 3 + 1]<< ", " << coordinates[atom * 3 + 2] << std::endl;
			#endif
		}

		


	}
	std::cout << this->molecule_ids[0] << " " << this->molecule_ids[1] << std::endl;

	delete[] buffer1;

	fh.close();
	return 0;
}