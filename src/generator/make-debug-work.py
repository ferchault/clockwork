#!/usr/bin/env python
""" Usage: make-debug-work.py molecule-id"""
import time
import sys
import os
import struct
import redis as rd

def worker_steps():
	# pull work create a copy in WPR
	wp = con.rpoplpush('WPQ', 'WPR')
	# label time
	con.hset('WPRstarted', wp, time.time())
	# do work
	pass
	# cleanup
	con.lrem('WPR', 1, wp)
	con.hdel('WPRstarted', sample_workpackage)

def make_work(molecule_id):
	sample_workpackage = struct.pack('=IBBBBBBBB', molecule_id, 1, 0, 1, 5, 1, 1, 1, 1)
	con = rd.Redis(host=os.getenv('CONFORMERREDIS'), password='chemspacelab')
	# submit work
	for i in range(100):
		con.rpush('WPQ', sample_workpackage)
	
if __name__ == '__main__':
	make_work(int(sys.argv[1]))
