#include <Python.h>
#include <patchlevel.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL NBIS_NUMPY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdbool.h>

#include <lfs.h>
#include <morph.h>
#include <mytime.h>
#include <log.h>
#include <imgboost.h>


int custom_gen_image_maps(
    int (*custom_dirmap)(int**, int, int, unsigned char*, const int, const int, int*, const int),
    int **odmap, int **olcmap, int **olfmap, int **ohcmap,
    int *omw, int *omh,
    unsigned char *pdata, const int pw, const int ph,
    const DIR2RAD *dir2rad, const DFTWAVES *dftwaves,
    const ROTGRIDS *dftgrids, const LFSPARMS *lfsparms);

int custom_lfs_detect_minutiae_V2(
    int (*custom_dirmap)(int**, int, int, unsigned char*, const int, const int, int*, const int),
    MINUTIAE **ominutiae,
    int **odmap, int **olcmap, int **olfmap, int **ohcmap,
    int *omw, int *omh,
    unsigned char **obdata, int *obw, int *obh,
    unsigned char *idata, const int iw, const int ih,
    const LFSPARMS *lfsparms);

int custom_get_minutiae(
    int (*custom_dirmap)(int**, int, int, unsigned char*, const int, const int, int*, const int),
    MINUTIAE **ominutiae, int **oquality_map,
    int **odirection_map, int **olow_contrast_map,
    int **olow_flow_map, int **ohigh_curve_map,
    int *omap_w, int *omap_h,
    unsigned char **obdata, int *obw, int *obh, int *obd,
    unsigned char *idata, const int iw, const int ih,
    const int id, const double ppmm, const LFSPARMS *lfsparms);

int custom_dirmap(
    int** direction_map, int mw, int mh, 
    unsigned char* pdata, const int pw, const int ph,
    int*, const int);

int minutiae_to_python(PyObject **min_list, MINUTIAE *minutiae);

PyObject* nbis_mindtct(PyObject *self, PyObject *args, PyObject *kwargs);
