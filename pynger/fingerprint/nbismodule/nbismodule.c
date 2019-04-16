#include <Python.h>
#include <patchlevel.h>
#define PY_ARRAY_UNIQUE_SYMBOL NBIS_NUMPY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
// NBIS library
#include "sgmnt.h"
#include "enhnc.h"
#include "rors.h"
#include "mindtct.h"

static PyMethodDef NBISmethods[] = {
	{"segment", (PyCFunction)nbis_sgmnt, METH_VARARGS|METH_KEYWORDS, "Segments the input image"},
	{"enhance", (PyCFunction)nbis_enhnc, METH_VARARGS|METH_KEYWORDS, "Enhances the input image"},
	{"compute_lro", (PyCFunction)nbis_rors, METH_VARARGS|METH_KEYWORDS, "Compute the local ridge orientation of the input fingerprint"},
	{"mindtct", (PyCFunction)nbis_mindtct, METH_VARARGS|METH_KEYWORDS, "Perform minutiae detection using the given function as orientation extractor"},
	{NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef NBISmodule = {
	PyModuleDef_HEAD_INIT,
	"nbis",
	"Wrapper to NBIS software suite tools",
	-1,
	NBISmethods
};

PyMODINIT_FUNC PyInit_nbis(void)
{
	PyObject* obj = PyModule_Create(&NBISmodule);
	import_array();
	return obj;
}