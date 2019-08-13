#!/usr/bin/env python
import time
import os
import redis
import sys
import subprocess
import shlex
import gzip
import hashlib

class Taskqueue(object):
	def __init__(self, connectionstring, projectprefix):
		""" Connection string: password@host:6380/1 """
		self._prefix = projectprefix
		password, rest = connectionstring.split('@')
		hostname, rest = rest.split(':')
		port, db = rest.split('/')
		try:
			self._con = redis.Redis(host=hostname, port=int(port), db=int(db), password=password)
		except:
			raise ValueError('Unable to connect to this redis instance.')

	def _start_task(self):
		""" Fetches a new task from redis and returns the associated string."""
		task = self._con.rpoplpush(self._prefix + '_Queue', self._prefix + '_Running')
		if task is None:
			return None

		self._taskid = hashlib.md5(task).hexdigest()
		self._starttime = time.time()
		self._con.hset(self._prefix + '_Started', self._taskid, self._starttime)
		return task
		
	def _store_result(self, taskstring, resultstring, logmessage=None, storestring="Results"):
		""" Signals task completion to redis and stores the results."""
		# remove server-side backup
		pipeline = self._con.pipeline()
		pipeline.lrem(self._prefix + '_Running', 1, taskstring)
		pipeline.hdel(self._prefix + '_Started', self._taskid)

		# Store results and logs
		pipeline.lpush(self._prefix + "_" + storestring, resultstring)
		if logmessage is not None:
			pipeline.hset(self._prefix + '_Log', taskstring, logmessage)
		pipeline.execute()

		# Update stats
		unixminute = int(time.time() / 60)
		duration = time.time() - self._starttime
		statskey = self._prefix + '_Stats:%d' % unixminute
		pipeline = self._con.pipeline()
		pipeline.rpush(statskey, duration)
		pipeline.expire(statskey, 3600)
		pipeline.execute()
	
	def _get_deadline(self):
		""" Calculates the deadline of job termination. Returns a unix time stamp. When run outside SLURM, returns 0. """
		jobid = os.getenv('SLURM_JOB_ID')
		if jobid is None:
			return 0

		cmd = 'squeue -h -j %s -o "%%L"' % jobid
		try:
			p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE)
		except:
			return 0
		remaining = p.stdout.decode('utf8').strip()

		# parse time format
		days = 0
		if '-' in remaining:
			days, rest = remaining.split('-')
		else:
			rest = remaining
		try:
			hours, minutes, seconds = rest.split(':')
		except:
			return 0

		return time.time() + ((int(days)*24 + int(hours))*60 + int(minutes))*60 + int(seconds)

	def main_loop(self, callback):
		"""
		callback should return (result_str, log_str, store_str)
		which is the result, the status and where to put the result.
		"""
		deadline = self._get_deadline()
		while True:
			if deadline > 0 and deadline - time.time() < 120:
				break
			gztaskstring = self._start_task()
			if gztaskstring is None:
				break
			taskstring = gzip.decompress(gztaskstring).decode('utf8')
			result_str, log_str, store_str = callback(taskstring)
			gzresultstring = gzip.compress(result_str.encode())
			self._store_result(gztaskstring, gzresultstring, log_str, storestring=store_str)

	def insert(self, taskstring, iscompressed=False):
		""" Enqueues a task. """
		if not iscompressed:
			gztaskstring = gzip.compress(taskstring.encode())
		else:
			gztaskstring = taskstring
		self._con.lpush(self._prefix + '_Queue', gztaskstring)

	def insert_batch(self, taskstrings, iscompressed=False):
		""" Enqueues multiple tasks. """
		if not iscompressed:
			gztaskstrings = [gzip.compress(_.encode()) for _ in taskstrings]
		else:
			gztaskstrings = taskstrings
		pipe = self._con.pipeline()
		batch_length = 0
		for i in range(len(gztaskstrings)):
			batch_length += 1
			pipe.lpush(self._prefix + '_Queue', gztaskstrings[i])
			if batch_length % 500 == 0:
				pipe.execute()
				pipe = self._con.pipeline()
		pipe.execute()

	def get_results(self, purge_after=False):
		""" Fetches and optionally deletes the results."""
		results = self._con.lrange(self._prefix + '_Results', 0, -1)
		results = [gzip.decompress(_).decode('utf8') for _ in results]
		if purge_after:
			self._con.delete(self._prefix + '_Results')
		return results

	def discover_projects(self):
		""" Finds valid projects using the selected database."""
		projects = []
		for kind in 'Results Queue Log Running Started'.split():
			keys = self._con.keys('*_%s' % kind)
			projects += [_.decode('utf8')[:-(len(kind)+1)] for _ in keys]
		return list(set(projects))

	def get_orphaned(self, projectname=None):
		""" Returns a list of orphaned task ids. """
		now = time.time()
		orphaned = []
		if projectname is None:
			projectname = self._prefix
		for taskid, starttime in self._con.hscan_iter('%s_Started' % projectname):
			if now - float(starttime) > 20*60:
				orphaned.append(taskid)
		return orphaned

	def requeue_orphaned(self):
		orphaned = [_.decode('utf8') for _ in self.get_orphaned()]

		running = set(self._con.lrange(self._prefix + '_Running', 0, -1))
		tobeinserted = []
		for workpackage in running:
			taskid = hashlib.md5(workpackage).hexdigest()
			if taskid in orphaned:
				self._con.hdel('%s_Started' % self._prefix, taskid)
				self._con.lrem('%s_Running' % self._prefix, 0, workpackage)
				tobeinserted.append(workpackage)

		# remove orphans without workpackage in running state
		for orphan in orphaned:
			self._con.hdel('%s_Started' % self._prefix, orphan)

		# insert
		for workpackage in tobeinserted:
			self.insert(workpackage, iscompressed=True)

	def print_stats(self, projectname):
		print ('Summary for project %s' % projectname)
		print ('  Waiting:       ', self._con.llen('%s_Queue' % projectname))
		print ('  Results:       ', self._con.llen('%s_Results' % projectname))
		errors = self._con.hlen('%s_Log' % projectname)
		errorattn = '!' if errors > 0 else ' '
		print ('%s Errors in log: ' % errorattn, errors)
		orphaned = len(self.get_orphaned(projectname))
		orphanattn = '!' if orphaned > 0 else ' '
		print ('%s Orphaned:      ' % orphanattn, orphaned)
		running = self._con.llen('%s_Running' % projectname) - orphaned
		print ('  Running:       ', running)

		# get rate info
		statskeys = sorted(self._con.keys('%s_Stats:*' % projectname))
		if running > 0 and len(statskeys) > 0:
			statsentries = sum([self._con.llen(_) for _ in statskeys])
			print ('    Packages / h:', int(statsentries / len(statskeys) * 60))

		# get workpackage durations
		durations = []
		for statskey in statskeys:
			durations += map(float, self._con.lrange(statskey, 0, -1))
		if len(durations) > 0:
			import numpy as np
			durations = np.array(durations)
			print ('    Min:         ', int(np.min(durations)))
			print ('    Mean:        ', int(np.average(durations)))
			print ('    Max:         ', int(np.max(durations)))
			print ('    Last:        ', list(map(int, durations[-10:][::-1])))

	def has_work(self):
		return self._con.llen('%s_Queue' % self._prefix) > 0

def do_work(task):
	""" Sample task evaluation. Returns a result as string and an optional log message."""
	time.sleep(2)
	return "result", "Another information"

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--status', action="store_true", help='')
	parser.add_argument('--haswork', type=str, help='Requires a project name.', metavar='project')
	parser.add_argument('--requeue_orphaned', type=str, help='Requires a project name.', metavar='project')

	args = parser.parse_args()

	if args.haswork:
		tasks = Taskqueue(os.getenv('CHEMSPACELAB_REDIS_CONNECTION'), args.haswork)
		sys.exit(not tasks.has_work())

	if args.status:
		tasks = Taskqueue(os.getenv('CHEMSPACELAB_REDIS_CONNECTION'), 'DEBUG')
		for project in sorted(tasks.discover_projects()):
			tasks.print_stats(project)

	if args.requeue_orphaned:
		tasks = Taskqueue(os.getenv('CHEMSPACELAB_REDIS_CONNECTION'), args.requeue_orphaned)
		tasks.requeue_orphaned()
