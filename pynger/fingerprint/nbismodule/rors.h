#include <Python.h>
#include <patchlevel.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL NBIS_NUMPY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <pca.h>

#include "utils.h"

PyObject* nbis_rors(PyObject *self, PyObject *args, PyObject *kwargs);