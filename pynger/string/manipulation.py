from typing import *


def remove_prefix(s: str, pref: Union[str, Iterable[str]]) -> str:
	""" Remove a prefix from the input string.
		
	Args:
	    s: String to modify
	    pref: Prefix to remove from the string
		
	Returns:
	    String without the prefix
	"""
	if isinstance(pref, str):
		return s[len(pref):] if s.startswith(pref) else s
	elif isinstance(pref, Iterable[str]):
		try:
			idx = [s.startswith(p) for p in pref].index(True)
		except ValueError:
			return s
		else:
			return s[len(pref[idx]):]
	else:
		raise TypeError("Incorrect prefix type")
	
def remove_suffix(s: str, suff: Union[str, Iterable[str]]) -> str:
	""" Remove a suffix from the input string.
		
	Args:
	    s: String to modify
	    suff: Suffix to remove from the string
		
	Returns:
	    String without the suffix
	"""
	if isinstance(suff, str):
		return s[:-len(suff)] if s.endswith(suff) else s
	elif isinstance(suff, Iterable[str]):
		try:
			idx = [s.endswith(p) for p in suff].index(True)
		except ValueError:
			return s
		else:
			return s[:-len(suff[idx])]
	else:
		raise TypeError("Incorrect suffix type")