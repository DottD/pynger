extern "C" {
	#include <Python.h>
	#include <patchlevel.h>
	#define PY_ARRAY_UNIQUE_SYMBOL NBIS_NUMPY_API
	#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
	#include <numpy/arrayobject.h>
}

#include "Headers/ang_seg_wrapper.hpp"

PyObject* sgmnt_enh(PyObject *self, PyObject *args, PyObject *kwargs)
{
	// Declarations
	PyObject *input, *array; // input
	npy_intp *dim;
	unsigned char* data;
	PyObject *enh_img, *fg_mask; // output
	unsigned char *enh_img_data, *fg_mask_data;

	// Parameters declarations and defaults
	_ImageBimodalize ImageBimodalize = {
		.brightness = 0.35,
		.leftCut = 0.25,
		.rightCut = 0.5,
		.histSmooth = 25,
		.reparSmooth = 10
	};
	_ImageCroppingSimple ImageCroppingSimple = {
		.minVariation = 0.01,
		.marg = 5
	};
	_TopMask TopMask = {
		.scanAreaAmount = 0.1,
		.gradientFilterWidth = 0.25,
		.gaussianFilterSide = 5,
		.binarizationLevel = 0.2,
		.f = 3.5,
		.slopeAngle = 1.5,
		.lev = 0.95,
		.marg = 5
	};
	_ImageSignificantMask ImageSignificantMask = {
		.medFilterSide = 2,
		.gaussFilterSide = 3,
		.minFilterSide = 5,
		.binLevVarMask = 0.45,
		.dilate1RadiusVarMask = 5,
		.erodeRadiusVarMask = 35,
		.dilate2RadiusVarMask = 20,
		.maxCompNumVarMask = 2,
		.minCompThickVarMask = 75,
		.maxHolesNumVarMask = -1,
		.minHolesThickVarMask = 18,
		.histeresisThreshold1Gmask = 30,
		.histeresisThreshold2Gmask = 70,
		.radiusGaussFilterGmask = 10,
		.minMeanIntensityGmask = 0.2,
		.dilate1RadiusGmask = 10,
		.erodeRadiusGmask = 25,
		.dilate2RadiusGmask = 10,
		.maxCompNumGmask = 2,
		.minCompThickGmask = 75,
		.maxHolesNumGmask = -1,
		.minHolesThickGmask = 15,
		.histeresisThreshold3Gmask = 25,
		.histeresisThreshold4Gmask = 50,
		.dilate3RadiusGmask = 10,
		.erode2RadiusGmask = 5,
		.histeresisThreshold5Gmask = 45,
		.histeresisThreshold6Gmask = 90,
		.dilate4RadiusGmask = 4,
		.radiusGaussFilterComp = 30,
		.meanIntensityCompThreshold = 0.6,
		.dilateFinalRadius = 10,
		.erodeFinalRadius = 20,
		.smoothFinalRadius = 10,
		.maxCompNumFinal = 2,
		.minCompThickFinal = 75,
		.maxHolesNumFinal = 4,
		.minHolesThickFinal = 30,
		.fixedFrameWidth = 20,
		.smooth2FinalRadius = 20
	};
	_ImageEqualize ImageEqualize = {
		.minMaxFilter = 5,
		.mincp1 = 0.75,
		.mincp2 = 0.9,
		.maxcp1 = 0.0,
		.maxcp2 = 0.25
	};
		
	// Parse input arguments
	static char *kwlist[] = 
	{
		(char*)"image",
		(char*)"brightness",
		(char*)"leftCut",
		(char*)"rightCut",
		(char*)"histSmooth",
		(char*)"reparSmooth",
		(char*)"minVariation",
		(char*)"cropSimpleMarg",
		(char*)"scanAreaAmount",
		(char*)"gradientFilterWidth",
		(char*)"gaussianFilterSide",
		(char*)"binarizationLevel",
		(char*)"f",
		(char*)"slopeAngle",
		(char*)"lev",
		(char*)"topMaskMarg",
		(char*)"medFilterSide",
		(char*)"gaussFilterSide",
		(char*)"minFilterSide",
		(char*)"binLevVarMask",
		(char*)"dilate1RadiusVarMask",
		(char*)"erodeRadiusVarMask",
		(char*)"dilate2RadiusVarMask",
		(char*)"maxCompNumVarMask",
		(char*)"minCompThickVarMask",
		(char*)"maxHolesNumVarMask",
		(char*)"minHolesThickVarMask",
		(char*)"histeresisThreshold1Gmask",
		(char*)"histeresisThreshold2Gmask",
		(char*)"radiusGaussFilterGmask",
		(char*)"minMeanIntensityGmask",
		(char*)"dilate1RadiusGmask",
		(char*)"erodeRadiusGmask",
		(char*)"dilate2RadiusGmask",
		(char*)"maxCompNumGmask",
		(char*)"minCompThickGmask",
		(char*)"maxHolesNumGmask",
		(char*)"minHolesThickGmask",
		(char*)"histeresisThreshold3Gmask",
		(char*)"histeresisThreshold4Gmask",
		(char*)"dilate3RadiusGmask",
		(char*)"erode2RadiusGmask",
		(char*)"histeresisThreshold5Gmask",
		(char*)"histeresisThreshold6Gmask",
		(char*)"dilate4RadiusGmask",
		(char*)"radiusGaussFilterComp",
		(char*)"meanIntensityCompThreshold",
		(char*)"dilateFinalRadius",
		(char*)"erodeFinalRadius",
		(char*)"smoothFinalRadius",
		(char*)"maxCompNumFinal",
		(char*)"minCompThickFinal",
		(char*)"maxHolesNumFinal",
		(char*)"minHolesThickFinal",
		(char*)"fixedFrameWidth",
		(char*)"smooth2FinalRadius",
		(char*)"minMaxFilter",
		(char*)"mincp1",
		(char*)"mincp2",
		(char*)"maxcp1",
		(char*)"maxcp2",
		NULL
	};
		
	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "O|$fffiififfiffffiiiifiiiiiiiiiifiiiiiiiiiiiiiiifiiiiiiiiiiffff", kwlist, // specifications
		&input, // argument
		&ImageBimodalize.brightness,
		&ImageBimodalize.leftCut,
		&ImageBimodalize.rightCut,
		&ImageBimodalize.histSmooth,
		&ImageBimodalize.reparSmooth,
		&ImageCroppingSimple.minVariation,
		&ImageCroppingSimple.marg,
		&TopMask.scanAreaAmount,
		&TopMask.gradientFilterWidth,
		&TopMask.gaussianFilterSide,
		&TopMask.binarizationLevel,
		&TopMask.f,
		&TopMask.slopeAngle,
		&TopMask.lev,
		&TopMask.marg,
		&ImageSignificantMask.medFilterSide,
		&ImageSignificantMask.gaussFilterSide,
		&ImageSignificantMask.minFilterSide,
		&ImageSignificantMask.binLevVarMask,
		&ImageSignificantMask.dilate1RadiusVarMask,
		&ImageSignificantMask.erodeRadiusVarMask,
		&ImageSignificantMask.dilate2RadiusVarMask,
		&ImageSignificantMask.maxCompNumVarMask,
		&ImageSignificantMask.minCompThickVarMask,
		&ImageSignificantMask.maxHolesNumVarMask,
		&ImageSignificantMask.minHolesThickVarMask,
		&ImageSignificantMask.histeresisThreshold1Gmask,
		&ImageSignificantMask.histeresisThreshold2Gmask,
		&ImageSignificantMask.radiusGaussFilterGmask,
		&ImageSignificantMask.minMeanIntensityGmask,
		&ImageSignificantMask.dilate1RadiusGmask,
		&ImageSignificantMask.erodeRadiusGmask,
		&ImageSignificantMask.dilate2RadiusGmask,
		&ImageSignificantMask.maxCompNumGmask,
		&ImageSignificantMask.minCompThickGmask,
		&ImageSignificantMask.maxHolesNumGmask,
		&ImageSignificantMask.minHolesThickGmask,
		&ImageSignificantMask.histeresisThreshold3Gmask,
		&ImageSignificantMask.histeresisThreshold4Gmask,
		&ImageSignificantMask.dilate3RadiusGmask,
		&ImageSignificantMask.erode2RadiusGmask,
		&ImageSignificantMask.histeresisThreshold5Gmask,
		&ImageSignificantMask.histeresisThreshold6Gmask,
		&ImageSignificantMask.dilate4RadiusGmask,
		&ImageSignificantMask.radiusGaussFilterComp,
		&ImageSignificantMask.meanIntensityCompThreshold,
		&ImageSignificantMask.dilateFinalRadius,
		&ImageSignificantMask.erodeFinalRadius,
		&ImageSignificantMask.smoothFinalRadius,
		&ImageSignificantMask.maxCompNumFinal,
		&ImageSignificantMask.minCompThickFinal,
		&ImageSignificantMask.maxHolesNumFinal,
		&ImageSignificantMask.minHolesThickFinal,
		&ImageSignificantMask.fixedFrameWidth,
		&ImageSignificantMask.smooth2FinalRadius,
		&ImageEqualize.minMaxFilter,
		&ImageEqualize.mincp1,
		&ImageEqualize.mincp2,
		&ImageEqualize.maxcp1,
		&ImageEqualize.maxcp2
		))
		return NULL;
			
	// Convert to a Numpy array
	array = PyArray_FROM_OTF(input, NPY_UBYTE, NPY_ARRAY_C_CONTIGUOUS);
	if (array == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "Unable to convert input to np.array");
		return NULL;
	}
			
	// Read dimension and data
	dim = PyArray_DIMS( (PyArrayObject*)array );
	data = (npy_ubyte*)PyArray_DATA( (PyArrayObject*)array );

	// Allocate memory for the output arrays and assign their values
	enh_img = PyArray_SimpleNew(2, dim, NPY_UBYTE);
	enh_img_data = (unsigned char*)PyArray_DATA( (PyArrayObject*)enh_img );

	fg_mask = PyArray_SimpleNew(2, dim, NPY_UBYTE);
	fg_mask_data = (unsigned char*)PyArray_DATA( (PyArrayObject*)fg_mask );
	
	// Segment the image
	char* seg_error = segmentation(data, dim, ImageBimodalize, ImageCroppingSimple, TopMask, ImageSignificantMask, ImageEqualize, enh_img_data, fg_mask_data);
	if (seg_error != NULL) {
		PyErr_SetString(PyExc_RuntimeError, seg_error);
		return NULL;
	}
	
	// Free memory, return values
	Py_DECREF(array);
	return Py_BuildValue("NN", enh_img, fg_mask);
}

static PyMethodDef methods[] = {
	{"segment_enhance", (PyCFunction)sgmnt_enh, METH_VARARGS|METH_KEYWORDS, "Segments and enhances the input image"},
	{NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef module = {
	PyModuleDef_HEAD_INIT,
	"cangafris",
	"Anisotropic Gaussian Filtering and Regularization through Iterative Smoothing",
	-1,
	methods
};

PyMODINIT_FUNC PyInit_cangafris(void)
{
	PyObject* obj = PyModule_Create(&module);
	import_array();
	return obj;
}
