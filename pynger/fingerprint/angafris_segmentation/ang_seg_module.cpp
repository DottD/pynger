#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Headers/ang_seg_wrapper.hpp"

static PyObject* sgmnt_enh(PyObject *self, PyObject *args, PyObject *kwargs)
{
	// Declarations
	PyObject *input, *array; // input
	npy_intp *dim;
	unsigned char* data;
	PyObject *enh_img, *fg_mask; // output
	unsigned char *enh_img_data, *fg_mask_data;
	
	_ImageBimodalize ImageBimodalize;
	_ImageCroppingSimple ImageCroppingSimple;
	_TopMask TopMask;
	_ImageSignificantMask ImageSignificantMask;
	_ImageEqualize ImageEqualize;
	int enhance_only = 0; // do everything by default
		
	// Parse input arguments
	static const char *kwlist[] = 
	{
		"image",
		"brightness",
		"leftCut",
		"rightCut",
		"histSmooth",
		"reparSmooth",
		"minVariation",
		"cropSimpleMarg",
		"scanAreaAmount",
		"gradientFilterWidth",
		"gaussianFilterSide",
		"binarizationLevel",
		"f",
		"slopeAngle",
		"lev",
		"topMaskMarg",
		"medFilterSide",
		"gaussFilterSide",
		"minFilterSide",
		"binLevVarMask",
		"dilate1RadiusVarMask",
		"erodeRadiusVarMask",
		"dilate2RadiusVarMask",
		"maxCompNumVarMask",
		"minCompThickVarMask",
		"maxHolesNumVarMask",
		"minHolesThickVarMask",
		"histeresisThreshold1Gmask",
		"histeresisThreshold2Gmask",
		"radiusGaussFilterGmask",
		"minMeanIntensityGmask",
		"dilate1RadiusGmask",
		"erodeRadiusGmask",
		"dilate2RadiusGmask",
		"maxCompNumGmask",
		"minCompThickGmask",
		"maxHolesNumGmask",
		"minHolesThickGmask",
		"histeresisThreshold3Gmask",
		"histeresisThreshold4Gmask",
		"dilate3RadiusGmask",
		"erode2RadiusGmask",
		"histeresisThreshold5Gmask",
		"histeresisThreshold6Gmask",
		"dilate4RadiusGmask",
		"radiusGaussFilterComp",
		"meanIntensityCompThreshold",
		"dilateFinalRadius",
		"erodeFinalRadius",
		"smoothFinalRadius",
		"maxCompNumFinal",
		"minCompThickFinal",
		"maxHolesNumFinal",
		"minHolesThickFinal",
		"fixedFrameWidth",
		"smooth2FinalRadius",
		"minMaxFilter",
		"mincp1",
		"mincp2",
		"maxcp1",
		"maxcp2",
		"enhanceOnly",
		NULL
	};
	
	// Default values
	ImageBimodalize.brightness = 0.35;
	ImageBimodalize.leftCut = 0.25;
	ImageBimodalize.rightCut = 0.5;
	ImageBimodalize.histSmooth = 25;
	ImageBimodalize.reparSmooth = 10;
	ImageCroppingSimple.minVariation = 0.01;
	ImageCroppingSimple.marg = 5;
	TopMask.scanAreaAmount = 0.1;
	TopMask.gradientFilterWidth = 0.25;
	TopMask.gaussianFilterSide = 5;
	TopMask.binarizationLevel = 0.2;
	TopMask.f = 3.5;
	TopMask.slopeAngle = 1.5;
	TopMask.lev = 0.95;
	TopMask.marg = 5;
	ImageSignificantMask.medFilterSide = 2;
	ImageSignificantMask.gaussFilterSide = 3;
	ImageSignificantMask.minFilterSide = 5;
	ImageSignificantMask.binLevVarMask = 0.45;
	ImageSignificantMask.dilate1RadiusVarMask = 5;
	ImageSignificantMask.erodeRadiusVarMask = 35;
	ImageSignificantMask.dilate2RadiusVarMask = 20;
	ImageSignificantMask.maxCompNumVarMask = 2;
	ImageSignificantMask.minCompThickVarMask = 75;
	ImageSignificantMask.maxHolesNumVarMask = -1;
	ImageSignificantMask.minHolesThickVarMask = 18;
	ImageSignificantMask.histeresisThreshold1Gmask = 30;
	ImageSignificantMask.histeresisThreshold2Gmask = 70;
	ImageSignificantMask.radiusGaussFilterGmask = 10;
	ImageSignificantMask.minMeanIntensityGmask = 0.2;
	ImageSignificantMask.dilate1RadiusGmask = 10;
	ImageSignificantMask.erodeRadiusGmask = 25;
	ImageSignificantMask.dilate2RadiusGmask = 10;
	ImageSignificantMask.maxCompNumGmask = 2;
	ImageSignificantMask.minCompThickGmask = 75;
	ImageSignificantMask.maxHolesNumGmask = -1;
	ImageSignificantMask.minHolesThickGmask = 15;
	ImageSignificantMask.histeresisThreshold3Gmask = 25;
	ImageSignificantMask.histeresisThreshold4Gmask = 50;
	ImageSignificantMask.dilate3RadiusGmask = 10;
	ImageSignificantMask.erode2RadiusGmask = 5;
	ImageSignificantMask.histeresisThreshold5Gmask = 45;
	ImageSignificantMask.histeresisThreshold6Gmask = 90;
	ImageSignificantMask.dilate4RadiusGmask = 4;
	ImageSignificantMask.radiusGaussFilterComp = 30;
	ImageSignificantMask.meanIntensityCompThreshold = 0.6;
	ImageSignificantMask.dilateFinalRadius = 10;
	ImageSignificantMask.erodeFinalRadius = 20;
	ImageSignificantMask.smoothFinalRadius = 10;
	ImageSignificantMask.maxCompNumFinal = 2;
	ImageSignificantMask.minCompThickFinal = 75;
	ImageSignificantMask.maxHolesNumFinal = 4;
	ImageSignificantMask.minHolesThickFinal = 30;
	ImageSignificantMask.fixedFrameWidth = 20;
	ImageSignificantMask.smooth2FinalRadius = 20;
	ImageEqualize.minMaxFilter = 5;
	ImageEqualize.mincp1 = 0.75;
	ImageEqualize.mincp2 = 0.9;
	ImageEqualize.maxcp1 = 0.0;
	ImageEqualize.maxcp2 = 0.25;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs,
		"O|$dddiididdiddddiiiidiiiiiiiiiidiiiiiiiiiiiiiiidiiiiiiiiiiddddp",
		const_cast<char**>(kwlist), // specifications
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
		&ImageEqualize.maxcp2,
		&enhance_only
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
#ifdef _WIN32
	char* seg_error = segmentation(data, (long*)dim, ImageBimodalize, ImageCroppingSimple, TopMask, ImageSignificantMask, ImageEqualize, enh_img_data, fg_mask_data, enhance_only);
#else
	char* seg_error = segmentation(data, dim, ImageBimodalize, ImageCroppingSimple, TopMask, ImageSignificantMask, ImageEqualize, enh_img_data, fg_mask_data, enhance_only);
#endif
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
