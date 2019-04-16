#include <Python.h>
#include <patchlevel.h>
#define PY_ARRAY_UNIQUE_SYMBOL NBIS_NUMPY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "panigauss.h"


PyObject* panigauss(PyObject *self, PyObject *args)
{
	// Declarations
	PyObject *input, *array, *output;
	double sigma_v, sigma_u, angle=0.0;
	int der_v=0, der_u=0;
	npy_intp *dim;
	int w, h;
	npy_double *data, *outdata;

	// Parse input arguments
	if (!PyArg_ParseTuple(args, "Odd|dii", &input,
		&sigma_v, &sigma_u, &angle, &der_v, &der_u))
		return NULL;

	// Input check
	if (der_v < 0 || der_u < 0){
    	PyErr_SetString(PyExc_ValueError, "Derivative order cannot be negative");
		return NULL;
	}
	if (sigma_u <= 0.0 || sigma_v <= 0.0){
    	PyErr_SetString(PyExc_ValueError, "Sigma must be positive");
		return NULL;
	}
		
	// Convert to a Numpy array
	array = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
	if (array == NULL) return NULL;
		
	// Read the dimension
	dim = PyArray_DIMS( (PyArrayObject*)array );
	h = dim[0];
	w = dim[1];
		
	// Get a pointer to the underline data of the array
	data = (npy_double*)PyArray_DATA( (PyArrayObject*)array );
	
	// Allocate memory for the output array
	output = PyArray_SimpleNew(2, dim, NPY_DOUBLE);
	outdata = (npy_double*)PyArray_DATA( (PyArrayObject*)output );
	
	// Execute anigauss
	if(anigauss(data, outdata, w, h, sigma_v, sigma_u, angle, der_v, der_u)){
		Py_DECREF(array);
    	PyErr_SetString(PyExc_RuntimeError, "Error running anigauss");
		return NULL;
	}
		
	// Free memory, return values
	Py_DECREF(array);
	return Py_BuildValue("N", output);
}

static PyMethodDef panimethods[] = {
	{
		"fast_ani_gauss_filter", (PyCFunction)panigauss, METH_VARARGS, 
		"Apply an anisotropic gaussian filter to the given image.\n\n"
		":param input: Image the filter will be applied to\n"
		":type input: numpy array\n"
		":param sigma_v: Standard deviation for Gaussian kernel along the short axis\n"
		"type sigma_v: float\n"
		":param sigma_u: Standard deviation for Gaussian kernel along the long axis\n"
		"type sigma_u: float\n"
		":param angle: Orientation angle in degrees\n"
		"type angle: float\n"
		":param der_v: Derivation order along the short axis\n"
		"type der_v: int\n"
		":param der_u: Derivation order along the short axis\n"
		"type der_u: int\n\n"
		"Example\n"
		"-------\n\n"
		"- For anisotropic data smoothing:\n"
		"``fast_ani_gauss_filter(image, 3.0, 7.0, 30.0, 0, 0)``\n"
		"- For anisotropic edge detection:\n"
		"``fast_ani_gauss_filter(image, 3.0, 7.0, 30.0, 1, 0)``\n"
		"- For anisotropic line detection:\n"
		"``fast_ani_gauss_filter(image, 3.0, 7.0, 30.0, 2, 0)``\n"
		":note: Check the paper:\n"
		"J. M. Geusebroek, A. W. M. Smeulders, and J. van de Weijer."
		"Fast anisotropic gauss filtering. IEEE Trans. Image Processing,"
		"vol. 12, no. 8, pp. 938-943, 2003."
	},
	{NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef panimodule = {
	PyModuleDef_HEAD_INIT,
	"panimodule",
	"Wrapper to anigauss tool by Geusebroek et al.",
	-1,
	panimethods
};

PyMODINIT_FUNC PyInit_pani(void)
{
	PyObject* obj = PyModule_Create(&panimodule);
	import_array();
	return obj;
}