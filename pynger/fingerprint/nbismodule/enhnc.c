#include "enhnc.h"

PyObject* nbis_enhnc(PyObject *self, PyObject *args, PyObject *kwargs)
{
	// Declarations
	ENHNC_PRS enhnc_prs;
	PyObject *input, *array, *output;
	npy_intp *dim;
	int sw, sh;
	npy_ubyte *data, *outdata;
	npy_ubyte **segras, **ehras;

	// Set up default parameters
	// int params
	enhnc_prs.rr1 = 150;
	enhnc_prs.rr2 = 449;
	// float params
	enhnc_prs.pow = 0.3;
		
	// Parse input arguments
	static char *kwlist[] = 
	{
		"input", // not a kwarg actually
		"rr1", "rr2", // int
		"pow", // float
		NULL
	};
	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "O|$iif", kwlist, // specifications
		&input, // argument
		&enhnc_prs.rr1, // list of variables
		&enhnc_prs.rr2,
		&enhnc_prs.pow
		))
		return NULL;
			
	// Convert to a Numpy array
	array = PyArray_FROM_OTF(input, NPY_UBYTE, NPY_ARRAY_C_CONTIGUOUS);
	if (array == NULL) return NULL;
			
	// Read the dimension
	dim = PyArray_DIMS( (PyArrayObject*)array );
	sh = dim[0];
	sw = dim[1];
	
	// Get a pointer to the underline data of the array
	data = (npy_ubyte*)PyArray_DATA( (PyArrayObject*)array );
	
	// Convert data to 2D array
	build_2d_array_b(&segras, data, sw, sh, true);
	
	// Apply the NBIS function
	enhnc(segras, &enhnc_prs, &ehras, sw, sh);
	free_dbl_char((char **)segras, sh);
	
	// Allocate memory for the output array
	output = PyArray_SimpleNew(2, dim, NPY_UBYTE);
	outdata = (npy_ubyte*)PyArray_DATA( (PyArrayObject*)output );
		
	// Make enhanced image pointer contiguous
	flatten_2d_array_b(&outdata, ehras, sw, sh, false);
	free_dbl_char((char **)ehras, sh);
	
	// Free memory, return values
	Py_DECREF(array);
	return Py_BuildValue("N", output);
}