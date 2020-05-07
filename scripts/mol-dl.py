#!/usr/bin/env python
import json
import redis
import os
import sys

redis = redis.Redis.from_url("redis://" + os.environ.get("EXECUTOR_CONSTR", "127.0.0.1:6379/0"))
class Downloader():
	def _do_workpackage(self, molname):
		self._connection = redis
		self._sdfstr = self._connection.get(f'clockwork:{molname}:sdf').decode("ascii")
		self._torsions = json.loads(self._connection.get(f'clockwork:{molname}:dihedrals').decode("ascii"))
		self._bonds = set([tuple(_) for _ in json.loads(self._connection.get(f'clockwork:{molname}:bonds').decode("ascii"))])

dl = Downloader()
dl._do_workpackage(sys.argv[1])
print (dl._torsions)
print (dl._sdfstr)
print (dl._bonds)
