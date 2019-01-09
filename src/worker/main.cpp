#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <omp.h>
#include <ctime>
#include <cstring>
//#include <openbabel/mol.h>
//#include <openbabel/forcefield.h>
//#include <openbabel/obconversion.h>
extern "C" {
	#include "hiredis/hiredis.h"
};
#include "io.cpp"

class Workpackage {
public:
	unsigned int molecule_id;
	unsigned int scan_dihedral;
	unsigned int clockwork_min;
	unsigned int clockwork_max;
	unsigned int frozen_count;
	unsigned int * frozen_dihedrals;
	unsigned int * frozen_dihedrals_n;
	unsigned int * frozen_dihedrals_i;

	void read_binary(unsigned char *){};
};

redisContext * redis_connect() {
	redisContext * context = redisConnect(getenv("CONFORMERREDIS"), 6379);
	if (context != NULL && context->err) {
		printf("Error: %s\n", context->errstr);
	}
	redisReply *reply;
	reply = (redisReply*)redisCommand(context, "AUTH chemspacelab");
	freeReplyObject(reply);

	return context;
}

// reads a new work package from redis securely (i.e. without chance of dropping the package)
int redis_fetch(redisContext * context, unsigned char * payload, size_t * payloadsize) {
	redisReply *reply;
	reply = (redisReply*)redisCommand(context, "RPOPLPUSH WPQ WPR");
	if (reply == NULL || reply->type != REDIS_REPLY_STRING) {
		// invalid reply || no further work or server-side issue
		freeReplyObject(reply);
		return 2;
	}

	// valid reply with more work
	*payloadsize = reply->len;
	unsigned char * chunk = new unsigned char[reply->len];
	std::memcpy(chunk, reply->str, reply->len);
	payload = chunk;
	freeReplyObject(reply);

	// store start time 
	reply = (redisReply*)redisCommand(context, "HSET WPRstarted %b %d", payload, payloadsize, (int)std::time(NULL));
	if (reply == NULL || reply->type != REDIS_REPLY_INTEGER) {
		// unexpected result, terminating to be on the safe side
		freeReplyObject(reply);	
		return 3;
	}
	freeReplyObject(reply);
	return 0;
}

// clears up the redis queues with running work packages
int redis_notify(redisContext * context, unsigned char * payload, size_t payloadsize, time_t packagestart) {
	redisReply *reply;
	
	// delete entry for running work packages
	reply = (redisReply*)redisCommand(context, "LREM WPR 1 %b", payload, payloadsize);
	if (reply == NULL) {
		std::cout << "Lost connection to REDIS in redis_notify LREM." << std::endl;
		return 2;
	}
	freeReplyObject(reply);

	// delete age entry
	reply = (redisReply*)redisCommand(context, "HDEL WPRstarted %b", payload, payloadsize);
	if (reply == NULL) {
		std::cout << "Lost connection to REDIS in redis_notify HDEL." << std::endl;
		return 2;
	}

	// update stats
	time_t now = std::time(NULL);
	reply = (redisReply*)redisCommand(context, "RPUSH WPRstats:%d %d", now % 60, now - packagestart);
	if (reply == NULL) {
		std::cout << "Lost connection to REDIS in redis_notify INCR." << std::endl;
		return 2;
	}	
	reply = (redisReply*)redisCommand(context, "EXPIRE WPRstats:%d 3600", now % 60);
	if (reply == NULL) {
		std::cout << "Lost connection to REDIS in redis_notify EXPIRE." << std::endl;
		return 2;
	}	

	return 0;	
}

int main(int argc,char **argv)
{
	Archive archive;
	archive.read_archive("../../fixtures/sample.archive");
	
	redisContext *context = redis_connect();

	Workpackage *wp = new Workpackage();
	unsigned char * payload = 0;
	size_t payloadsize;
	time_t packagestart;
	while (redis_fetch(context, payload, &payloadsize)) {
		packagestart = std::time(NULL);
		wp->read_binary(payload);

		
/*
	// Needed such that openbabel does not try to parallelise
	omp_set_num_threads(1);

	OpenBabel::OBForceField * ff = OpenBabel::OBForceField::FindForceField("MMFF94");


	OpenBabel::OBMol mol;
	for (atom = wp->atoms.begin(); atom != wp->atoms.end(); ++atom) {
		OpenBabel::OBAtom obatom;
		obatom.SetAtomicNum(OpenBabel::OBElements::GetAtomicNum(atom->number);
		obatom.SetVector(atom->x, atom->y, atom->z);
		mol->AddAtom(obatom);
	}
	
	for (bond = wp->bonds.begin(); bond != wp->bonds.end(); ++bond) {
		mol->AddBond(bond->begin, bond->end, bond->order);
	}
	
	// constraints
	constraints = OpenBabel::OBForceField::OBFFConstraints()
	ff->SetConstraints(constraints);
	for (frozen = wp->frozen_dihedrals.begin(); frozen != wp->frozen_dihedrals.end(); ++frozen) {
		constraints->AddTorsionConstraint(frozen->i, frozen->j, frozen->k, frozen->l, frozen->value)
	}

	for (scan in wp->scans.begin(); scan != wp->scans.end(); ++scan) {
		constraints->AddTorsionConstraint(scan->i, scan->j, scan->k, scan->l, scan->value);
		ff->ConjugateGradients(numsteps, threshold);
		constraints->DeleteConstraint(constraints->Size());
		ff->ConjugateGradients(numsteps, threshold);
		
		// Use mol.GetCoordinates() for rmsd check
	}
  */
		if (!redis_notify(context, payload, payloadsize, packagestart)) {
			std::cout << "Problems in communicating to REDIS. Aborting." << std::endl;
		}

		delete[] payload;
	}

	redisFree(context);
	return 0;
}
