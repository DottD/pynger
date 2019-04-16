import numpy as np
from collections import Iterable


def flatten(items):
	""" Yield items from any nested iterable; see REF. """
	for x in items:
		if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
			yield from flatten(x)
		else:
			yield x

def cartesian(arrays, out=None):
	""" Generate a cartesian product of input arrays.

	Parameters:
		arrays (list of array-like): 1-D arrays to form the cartesian product of.
		out (ndarray): Array to place the cartesian product in.

	Returns:
		out (ndarray): 2-D array of shape ``(M, len(arrays))`` containing cartesian products formed of input arrays.

	Examples:
		>>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
		array([[1, 4, 6],
			[1, 4, 7],
			[1, 5, 6],
			[1, 5, 7],
			[2, 4, 6],
			[2, 4, 7],
			[2, 5, 6],
			[2, 5, 7],
			[3, 4, 6],
			[3, 4, 7],
			[3, 5, 6],
			[3, 5, 7]])
	"""
	arrays = [np.asarray(x) for x in arrays]
	dtype = arrays[0].dtype

	n = np.prod([x.size for x in arrays])
	if out is None:
		out = np.zeros([n, len(arrays)], dtype=dtype)

	m = int(n / arrays[0].size)
	out[:,0] = np.repeat(arrays[0], m)
	if arrays[1:]:
		cartesian(arrays[1:], out=out[0:m,1:])
		for j in range(1, arrays[0].size):
			out[j*m:(j+1)*m,1:] = out[0:m,1:]
	return out
	
def combinations(iterable):
	""" Makes combinations out of the input iterable.

	Args:
		iterable (iterable): The iterable to process

	Transforms a dictionary ({name_i: values_i}) into a list of dictionaries structured as:

	- each dictionary.values() is an element of the carthesian product of values_i (computed accross every i)
	- each dictionary entry is a parameter name-value pair
		
	Transforms a list of values_i into a list of array, where each is an element of the carthesian product of values_i (computed accross every i).
	
	Note:
		The array type is inferred from the first array in list!
	"""
	if isinstance(iterable, list):
		return cartesian(iterable)
	elif isinstance(iterable, dict):
		values = cartesian(iterable.values())
		return list(dict(zip(iterable.keys(), val)) for val in values)
	else:
		raise ValueError("iterable must be a list or a dictionary")
	
	
if __name__ == '__main__':
	N = np.random.randint(3,5)
	A = list(np.random.randint(0,10,np.random.randint(1,4)).tolist() for _ in range(N))
	B = dict((str(n), np.random.randint(0,10,np.random.randint(1,4)).tolist()) for n in range(N))
	print('A')
	for a in A: print(a)
	print()
	print('B')
	for b1, b2 in B.items(): print(b1, b2)
	print('combinations(A)')
	for ca in combinations(A): print(ca)
	print('combinations(B)')
	for d in combinations(B): print(d)
	