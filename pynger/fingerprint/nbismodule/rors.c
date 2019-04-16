#include "rors.h"

/* Extracts from a raster the pixelwise ridge-orientation indices
(pixelrors) and also averages them in non-overlapping WSxWS-pixel
squares to make average ridge-orientation vectors (avrors_x,avrors_y).
Each vector (avrors_x[i][j],avrors_y[i][j]) is the average, over
square (i,j), of unit vectors pointing in the directions of the
doubled pixelwise-orientation angles; it will therefore have a short
length if the pixelwise orientations vary widely within its square,
because of cancellation.  Outer squares get pixelwise orientations
computed only for those of their pixels that are centers of slits that
fit entirely on the raster, and the average-orientations of outer
squares are not computed. */
PyObject* nbis_rors(PyObject *self, PyObject *args, PyObject *kwargs)
{
	// Declarations
	PyObject *input, *array, *out_pixelrors, *out_avrors_x, *out_avrors_y;
	npy_intp *dim;
	int sw, sh;
	int rors_slit_range_thresh;
	npy_ubyte *data;
	npy_ubyte **ehras;
	npy_byte **pixelrors;
	npy_float **avrors_x, **avrors_y;
	int aw, ah;
	npy_byte *out_pixelrors_data;
	npy_float *out_avrors_x_data, *out_avrors_y_data;

	// Set up default parameters
	/*
	rors_slit_range_thresh:
	if the difference between the maximum and minimum slit-sums 
	at a pixel is less than this, then this pixel makes no contribution
	to the histogram used to make the local average orientation
	*/
	rors_slit_range_thresh = 10;
		
	// Parse input arguments
	static char *kwlist[] = 
	{
		"input", // not a kwarg actually
		"slit_range_thresh",
		NULL
	};
	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "O|$i", kwlist, // specifications
		&input, // argument
		&rors_slit_range_thresh
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
	build_2d_array_b(&ehras, data, sw, sh, true);
	
	// Apply the NBIS function
	rors(ehras, sw, sh, rors_slit_range_thresh, &pixelrors, &avrors_x, &avrors_y, &aw, &ah);
	free_dbl_char((char **)ehras, sh);
	
	// Allocate memory for the output arrays
	out_pixelrors = PyArray_SimpleNew(2, dim, NPY_BYTE);
	out_pixelrors_data = (npy_byte*)PyArray_DATA( (PyArrayObject*)out_pixelrors );
	npy_intp out_avrors_dim[] = {ah, aw};
	out_avrors_x = PyArray_SimpleNew(2, out_avrors_dim, NPY_FLOAT);
	out_avrors_x_data = (npy_float*)PyArray_DATA( (PyArrayObject*)out_avrors_x );
	out_avrors_y = PyArray_SimpleNew(2, out_avrors_dim, NPY_FLOAT);
	out_avrors_y_data = (npy_float*)PyArray_DATA( (PyArrayObject*)out_avrors_y );
		
	// Flatten enhanced image pointer
	flatten_2d_array_c(&out_pixelrors_data, pixelrors, sw, sh, false);
	flatten_2d_array_f(&out_avrors_x_data, avrors_x, aw, ah, false);
	flatten_2d_array_f(&out_avrors_y_data, avrors_y, aw, ah, false);
	free_dbl_char((char **)pixelrors, ah);
	free_dbl_flt(avrors_x, ah);
	free_dbl_flt(avrors_y, ah);
		
	// Free memory, return values
	Py_DECREF(array);
	return Py_BuildValue("NNN", out_pixelrors, out_avrors_x, out_avrors_y);
}