#include "sgmnt.h"

PyObject* nbis_sgmnt(PyObject *self, PyObject *args, PyObject *kwargs)
{
	// Declarations
	SGMNT_PRS sgmnt_prs;
	PyObject *input, *array, *seg_out, *seg_fg_out;
	npy_intp *dim;
	int w, h;
	npy_ubyte* data;
	npy_ubyte **segras, **segras_fg; // segmented raster image and foreground
	int sfgw, sfgh;
	npy_ubyte *seg_out_data, *seg_fg_out_data;
	
	// Set up default parameters
	// int params
	sgmnt_prs.fac_n = 5;
	sgmnt_prs.min_fg = 2000;
	sgmnt_prs.max_fg = 8000;
	sgmnt_prs.nerode = 3;
	sgmnt_prs.rsblobs = 1;
	sgmnt_prs.fill = 1;
	sgmnt_prs.min_n = 25;
	sgmnt_prs.hist_thresh = 20;
	sgmnt_prs.origras_wmax = 2000;
	sgmnt_prs.origras_hmax = 2000;
	// float params
	sgmnt_prs.fac_min = 0.75;
	sgmnt_prs.fac_del = 0.05;
	sgmnt_prs.slope_thresh = 0.90;
		
	// Parse input arguments
	static char *kwlist[] = 
	{
		"input", // not a kwarg actually
		"fac_n", "min_fg", "max_fg", "nerode", // int
		"rsblobs", "fill", "min_n", "hist_thresh",
		"origras_wmax", "origras_hmax", 
		"fac_min", "fac_del", "slope_thresh", // float
		NULL
	};
	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "O|$iiiiiiiiiifff", kwlist, // specifications
		&input, // argument
		&sgmnt_prs.fac_n, // list of variables
		&sgmnt_prs.min_fg,
		&sgmnt_prs.max_fg,
		&sgmnt_prs.nerode,
		&sgmnt_prs.rsblobs,
		&sgmnt_prs.fill,
		&sgmnt_prs.min_n,
		&sgmnt_prs.hist_thresh,
		&sgmnt_prs.origras_wmax,
		&sgmnt_prs.origras_hmax,
		&sgmnt_prs.fac_min,
		&sgmnt_prs.fac_del,
		&sgmnt_prs.slope_thresh
		))
		return NULL;
			
	// Convert to a Numpy array
	array = PyArray_FROM_OTF(input, NPY_UBYTE, NPY_ARRAY_C_CONTIGUOUS);
	if (array == NULL) return NULL;
			
	// Read the dimension
	dim = PyArray_DIMS( (PyArrayObject*)array );
	h = dim[0];
	w = dim[1];
	if(h != 512 || w != 512)
	{
		char msg[50];
		sprintf(msg, "The input image should be %d x %d\n", 512, 512);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}
	
	// Get a pointer to the underline data of the array
	data = (npy_ubyte*)PyArray_DATA( (PyArrayObject*)array );
	
	// Apply the NBIS function
	sgmnt(data, w, h,
		&sgmnt_prs,
		&segras, WIDTH, HEIGHT,
		&segras_fg, &sfgw, &sfgh);
	
	// Allocate memory for the output array
	npy_intp seg_out_dim[] = {HEIGHT, WIDTH};
	seg_out = PyArray_SimpleNew(2, seg_out_dim, NPY_UBYTE);
	seg_out_data = (npy_ubyte*)PyArray_DATA( (PyArrayObject*)seg_out );
	npy_intp seg_fg_out_dim[] = {sfgh, sfgw};
	seg_fg_out = PyArray_SimpleNew(2, seg_fg_out_dim, NPY_UBYTE);
	seg_fg_out_data = (npy_ubyte*)PyArray_DATA( (PyArrayObject*)seg_fg_out );
		
	// Flatten segmented image pointer
	flatten_2d_array_b(&seg_out_data, segras, WIDTH, HEIGHT, false);
	free_dbl_char((char **)segras, HEIGHT);
		
	// Flatten segmented foreground pointer
	flatten_2d_array_b(&seg_fg_out_data, segras_fg, sfgw, sfgh, false);
	free_dbl_char((char **)segras_fg, sfgh);
	
	// Free memory, return values
	Py_DECREF(array);
	return Py_BuildValue("NN", seg_out, seg_fg_out);
}