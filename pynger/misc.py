import os
import random
from time import time
from itertools import zip_longest, combinations

		
def testTime(fun, rep=10, msg="Function executed in {} seconds on average"):
	t = time()
	for _ in range(rep-1): fun()
	out = fun()
	print(msg.format((time()-t)/rep))
	return out

def recursively_scan_dir(path, ending):
	"""
	Recursively scan the folder.
	
	Args:
		- path, path to recursevily analyze;
		- ending, list of possible extensions.
		
	Returns: a pair with the list of directories and the list of files (no clue on folder tree structure).
	"""
	file_list = []
	dir_list = []
	for curr_dir, _, local_files in os.walk(path):
		# filter local files
		local_files = [os.path.join(curr_dir, x) for x in local_files if any(map(lambda ext: x.endswith(ext), ending))]
		# append to global list
		file_list += local_files
		if local_files:
			dir_list.append(curr_dir)
	return dir_list, file_list


def recursively_scan_dir_gen(path, ending):
	"""
	Recursively scan the folder, but returns a generator.

	Return:
		This function yields the absolute path of the files.
	
	See Also:
		`.recursively_scan_dir'
	"""
	for curr_dir, _, local_files in os.walk(path):
		for x in filter(lambda x: x.endswith(ending), local_files):
			yield os.path.join(curr_dir, x)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)