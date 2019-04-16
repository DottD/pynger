#include "mindtct.h"

/*************************************************************************
**************************************************************************
#cat: gen_image_maps - Computes a set of image maps based on Version 2
#cat:            of the NIST LFS System.  The first map is a Direction Map
#cat:            which is a 2D vector of integer directions, where each
#cat:            direction represents the dominant ridge flow in a block of
#cat:            the input grayscale image.  The Low Contrast Map flags
#cat:            blocks with insufficient contrast.  The Low Flow Map flags
#cat:            blocks with insufficient ridge flow.  The High Curve Map
#cat:            flags blocks containing high curvature. This routine will
#cat:            generate maps for an arbitrarily sized, non-square, image.

   Input:
      pdata     - padded input image data (8 bits [0..256) grayscale)
      pw        - padded width (in pixels) of the input image
      ph        - padded height (in pixels) of the input image
      dir2rad   - lookup table for converting integer directions
      dftwaves  - structure containing the DFT wave forms
      dftgrids  - structure containing the rotated pixel grid offsets
      lfsparms  - parameters and thresholds for controlling LFS
   Output:
      odmap     - points to the created Direction Map
      olcmap    - points to the created Low Contrast Map
      olfmap    - points to the Low Ridge Flow Map
      ohcmap    - points to the High Curvature Map
      omw       - width (in blocks) of the maps
      omh       - height (in blocks) of the maps
   Return Code:
      Zero     - successful completion
      Negative - system error
**************************************************************************/
int custom_gen_image_maps(
    int (*custom_dirmap)(int**, int, int, unsigned char*, const int, const int, int*, const int),
    int **odmap, int **olcmap, int **olfmap, int **ohcmap,
    int *omw, int *omh,
    unsigned char *pdata, const int pw, const int ph,
    const DIR2RAD *dir2rad, const DFTWAVES *dftwaves,
    const ROTGRIDS *dftgrids, const LFSPARMS *lfsparms)
{
   int *direction_map, *low_contrast_map, *low_flow_map, *high_curve_map;
   int mw, mh, iw, ih;
   int *blkoffs;
   int ret; /* return code */

   /* 1. Compute block offsets for the entire image, accounting for pad */
   /* Block_offsets() assumes square block (grid), so ERROR otherwise. */
   if(dftgrids->grid_w != dftgrids->grid_h){
      fprintf(stderr,
              "ERROR : gen_image_maps : DFT grids must be square\n");
      return(-540);
   }
   /* Compute unpadded image dimensions. */
   iw = pw - (dftgrids->pad<<1);
   ih = ph - (dftgrids->pad<<1);
   if((ret = block_offsets(&blkoffs, &mw, &mh, iw, ih,
                        dftgrids->pad, lfsparms->blocksize))){
      return(ret);
   }

   /* 2. Generate initial Direction Map and Low Contrast Map*/
   if((ret = gen_initial_maps(&direction_map, &low_contrast_map,
                              &low_flow_map, blkoffs, mw, mh,
                              pdata, pw, ph, dftwaves, dftgrids, lfsparms))){
      /* Free memory allocated to this point. */
      free(blkoffs);
      return(ret);
   }

   /* 2.5 Overwrite the direction map */
   if((ret = custom_dirmap(&direction_map, mw, mh, pdata, pw, ph, blkoffs, lfsparms->num_directions))){
      free(blkoffs);
      free(direction_map);
      return (ret);
   }

   if((ret = morph_TF_map(low_flow_map, mw, mh, lfsparms))){
      return(ret);
   }

   /* 3. Remove directions that are inconsistent with neighbors */
   remove_incon_dirs(direction_map, mw, mh, dir2rad, lfsparms);


   /* 4. Smooth Direction Map values with their neighbors */
   smooth_direction_map(direction_map, low_contrast_map, mw, mh,
                           dir2rad, lfsparms);

   /* 5. Interpolate INVALID direction blocks with their valid neighbors. */
   if((ret = interpolate_direction_map(direction_map, low_contrast_map,
                                       mw, mh, lfsparms))){
      return(ret);
   }

   /* May be able to skip steps 6 and/or 7 if computation time */
   /* is a critical factor.                                    */

   /* 6. Remove directions that are inconsistent with neighbors */
   remove_incon_dirs(direction_map, mw, mh, dir2rad, lfsparms);

   /* 7. Smooth Direction Map values with their neighbors. */
   smooth_direction_map(direction_map, low_contrast_map, mw, mh,
                           dir2rad, lfsparms);

   /* 8. Set the Direction Map values in the image margin to INVALID. */
   set_margin_blocks(direction_map, mw, mh, INVALID_DIR);

   /* 9. Generate High Curvature Map from interpolated Direction Map. */
   if((ret = gen_high_curve_map(&high_curve_map, direction_map, mw, mh,
                                lfsparms))){
      return(ret);
   }

   /* Deallocate working memory. */
   free(blkoffs);

   *odmap = direction_map;
   *olcmap = low_contrast_map;
   *olfmap = low_flow_map;
   *ohcmap = high_curve_map;
   *omw = mw;
   *omh = mh;
   return(0);
}

/*************************************************************************
#cat: lfs_detect_minutiae_V2 - Takes a grayscale fingerprint image (of
#cat:          arbitrary size), and returns a set of image block maps,
#cat:          a binarized image designating ridges from valleys,
#cat:          and a list of minutiae (including position, reliability,
#cat:          type, direction, neighbors, and ridge counts to neighbors).
#cat:          The image maps include a ridge flow directional map,
#cat:          a map of low contrast blocks, a map of low ridge flow blocks.
#cat:          and a map of high-curvature blocks.

   Input:
      idata     - input 8-bit grayscale fingerprint image data
      iw        - width (in pixels) of the image
      ih        - height (in pixels) of the image
      lfsparms  - parameters and thresholds for controlling LFS

   Output:
      ominutiae - resulting list of minutiae
      odmap     - resulting Direction Map
                  {invalid (-1) or valid ridge directions}
      olcmap    - resulting Low Contrast Map
                  {low contrast (TRUE), high contrast (FALSE)}
      olfmap    - resulting Low Ridge Flow Map
                  {low ridge flow (TRUE), high ridge flow (FALSE)}
      ohcmap    - resulting High Curvature Map
                  {high curvature (TRUE), low curvature (FALSE)}
      omw       - width (in blocks) of image maps
      omh       - height (in blocks) of image maps
      obdata    - resulting binarized image
                  {0 = black pixel (ridge) and 255 = white pixel (valley)}
      obw       - width (in pixels) of the binary image
      obh       - height (in pixels) of the binary image
   Return Code:
      Zero      - successful completion
      Negative  - system error
**************************************************************************/
int custom_lfs_detect_minutiae_V2(
                        int (*custom_dirmap)(int**, int, int, unsigned char*, const int, const int, int*, const int),
                        MINUTIAE **ominutiae,
                        int **odmap, int **olcmap, int **olfmap, int **ohcmap,
                        int *omw, int *omh,
                        unsigned char **obdata, int *obw, int *obh,
                        unsigned char *idata, const int iw, const int ih,
                        const LFSPARMS *lfsparms)
{
   unsigned char *pdata, *bdata;
   int pw, ph, bw, bh;
   DIR2RAD *dir2rad;
   DFTWAVES *dftwaves;
   ROTGRIDS *dftgrids;
   ROTGRIDS *dirbingrids;
   int *direction_map, *low_contrast_map, *low_flow_map, *high_curve_map;
   int mw, mh;
   int ret, maxpad;
   MINUTIAE *minutiae;

   set_timer(total_timer);

   /******************/
   /* INITIALIZATION */
   /******************/

   /* If LOG_REPORT defined, open log report file. */
   if((ret = open_logfile()))
      /* If system error, exit with error code. */
      return(ret);

   /* Determine the maximum amount of image padding required to support */
   /* LFS processes.                                                    */
   maxpad = get_max_padding_V2(lfsparms->windowsize, lfsparms->windowoffset,
                          lfsparms->dirbin_grid_w, lfsparms->dirbin_grid_h);

   /* Initialize lookup table for converting integer directions */
   /* to angles in radians.                                     */
   if((ret = init_dir2rad(&dir2rad, lfsparms->num_directions))){
      /* Free memory allocated to this point. */
      return(ret);
   }

   /* Initialize wave form lookup tables for DFT analyses. */
   /* used for direction binarization.                             */
   if((ret = init_dftwaves(&dftwaves, dft_coefs, lfsparms->num_dft_waves,
                        lfsparms->windowsize))){
      /* Free memory allocated to this point. */
      free_dir2rad(dir2rad);
      return(ret);
   }

   /* Initialize lookup table for pixel offsets to rotated grids */
   /* used for DFT analyses.                                     */
   if((ret = init_rotgrids(&dftgrids, iw, ih, maxpad,
                        lfsparms->start_dir_angle, lfsparms->num_directions,
                        lfsparms->windowsize, lfsparms->windowsize,
                        RELATIVE2ORIGIN))){
      /* Free memory allocated to this point. */
      free_dir2rad(dir2rad);
      free_dftwaves(dftwaves);
      return(ret);
   }

   /* Pad input image based on max padding. */
   if(maxpad > 0){   /* May not need to pad at all */
      if((ret = pad_uchar_image(&pdata, &pw, &ph, idata, iw, ih,
                             maxpad, lfsparms->pad_value))){
         /* Free memory allocated to this point. */
         free_dir2rad(dir2rad);
         free_dftwaves(dftwaves);
         free_rotgrids(dftgrids);
         return(ret);
      }
   }
   else{
      /* If padding is unnecessary, then copy the input image. */
      pdata = (unsigned char *)malloc(iw*ih);
      if(pdata == (unsigned char *)NULL){
         /* Free memory allocated to this point. */
         free_dir2rad(dir2rad);
         free_dftwaves(dftwaves);
         free_rotgrids(dftgrids);
         fprintf(stderr, "ERROR : lfs_detect_minutiae_V2 : malloc : pdata\n");
         return(-580);
      }
      memcpy(pdata, idata, iw*ih);
      pw = iw;
      ph = ih;
   }

   /* Scale input image to 6 bits [0..63] */
   /* !!! Would like to remove this dependency eventualy !!!     */
   /* But, the DFT computations will need to be changed, and     */
   /* could not get this work upon first attempt. Also, if not   */
   /* careful, I think accumulated power magnitudes may overflow */
   /* doubles.                                                   */
   bits_8to6(pdata, pw, ph);

   print2log("\nINITIALIZATION AND PADDING DONE\n");

   /******************/
   /*      MAPS      */
   /******************/
   set_timer(imap_timer);

   /* Generate block maps from the input image. */
   if((ret = custom_gen_image_maps(
                     custom_dirmap,
                     &direction_map, &low_contrast_map,
                    &low_flow_map, &high_curve_map, &mw, &mh,
                    pdata, pw, ph, dir2rad, dftwaves, dftgrids, lfsparms))){
      /* Free memory allocated to this point. */
      free_dir2rad(dir2rad);
      free_dftwaves(dftwaves);
      free_rotgrids(dftgrids);
      free(pdata);
      return(ret);
   }
   /* Deallocate working memories. */
   free_dir2rad(dir2rad);
   free_dftwaves(dftwaves);
   free_rotgrids(dftgrids);

   print2log("\nMAPS DONE\n");

   time_accum(imap_timer, imap_time);

   /******************/
   /* BINARIZARION   */
   /******************/
   set_timer(bin_timer);

   /* Initialize lookup table for pixel offsets to rotated grids */
   /* used for directional binarization.                         */
   if((ret = init_rotgrids(&dirbingrids, iw, ih, maxpad,
                        lfsparms->start_dir_angle, lfsparms->num_directions,
                        lfsparms->dirbin_grid_w, lfsparms->dirbin_grid_h,
                        RELATIVE2CENTER))){
      /* Free memory allocated to this point. */
      free(pdata);
      free(direction_map);
      free(low_contrast_map);
      free(low_flow_map);
      free(high_curve_map);
      return(ret);
   }

   /* Binarize input image based on NMAP information. */
   if((ret = binarize_V2(&bdata, &bw, &bh,
                      pdata, pw, ph, direction_map, mw, mh,
                      dirbingrids, lfsparms))){
      /* Free memory allocated to this point. */
      free(pdata);
      free(direction_map);
      free(low_contrast_map);
      free(low_flow_map);
      free(high_curve_map);
      free_rotgrids(dirbingrids);
      return(ret);
   }

   /* Deallocate working memory. */
   free_rotgrids(dirbingrids);

   /* Check dimension of binary image.  If they are different from */
   /* the input image, then ERROR.                                 */
   if((iw != bw) || (ih != bh)){
      /* Free memory allocated to this point. */
      free(pdata);
      free(direction_map);
      free(low_contrast_map);
      free(low_flow_map);
      free(high_curve_map);
      free(bdata);
      fprintf(stderr, "ERROR : lfs_detect_minutiae_V2 :");
      fprintf(stderr,"binary image has bad dimensions : %d, %d\n",
              bw, bh);
      return(-581);
   }

   print2log("\nBINARIZATION DONE\n");

   time_accum(bin_timer, bin_time);

   /******************/
   /*   DETECTION    */
   /******************/
   set_timer(minutia_timer);

   /* Convert 8-bit grayscale binary image [0,255] to */
   /* 8-bit binary image [0,1].                       */
   gray2bin(1, 1, 0, bdata, iw, ih);

   /* Allocate initial list of minutia pointers. */
   if((ret = alloc_minutiae(&minutiae, MAX_MINUTIAE))){
      return(ret);
   }

   /* Detect the minutiae in the binarized image. */
   if((ret = detect_minutiae_V2(minutiae, bdata, iw, ih,
                             direction_map, low_flow_map, high_curve_map,
                             mw, mh, lfsparms))){
      /* Free memory allocated to this point. */
      free(pdata);
      free(direction_map);
      free(low_contrast_map);
      free(low_flow_map);
      free(high_curve_map);
      free(bdata);
      return(ret);
   }

   time_accum(minutia_timer, minutia_time);

   set_timer(rm_minutia_timer);

   if((ret = remove_false_minutia_V2(minutiae, bdata, iw, ih,
                       direction_map, low_flow_map, high_curve_map, mw, mh,
                       lfsparms))){
      /* Free memory allocated to this point. */
      free(pdata);
      free(direction_map);
      free(low_contrast_map);
      free(low_flow_map);
      free(high_curve_map);
      free(bdata);
      free_minutiae(minutiae);
      return(ret);
   }

   print2log("\nMINUTIA DETECTION DONE\n");

   time_accum(rm_minutia_timer, rm_minutia_time);

   /******************/
   /*  RIDGE COUNTS  */
   /******************/
   set_timer(ridge_count_timer);

   if((ret = count_minutiae_ridges(minutiae, bdata, iw, ih, lfsparms))){
      /* Free memory allocated to this point. */
      free(pdata);
      free(direction_map);
      free(low_contrast_map);
      free(low_flow_map);
      free(high_curve_map);
      free_minutiae(minutiae);
      return(ret);
   }


   print2log("\nNEIGHBOR RIDGE COUNT DONE\n");

   time_accum(ridge_count_timer, ridge_count_time);

   /******************/
   /*    WRAP-UP     */
   /******************/

   /* Convert 8-bit binary image [0,1] to 8-bit */
   /* grayscale binary image [0,255].           */
   gray2bin(1, 255, 0, bdata, iw, ih);

   /* Deallocate working memory. */
   free(pdata);

   /* Assign results to output pointers. */
   *odmap = direction_map;
   *olcmap = low_contrast_map;
   *olfmap = low_flow_map;
   *ohcmap = high_curve_map;
   *omw = mw;
   *omh = mh;
   *obdata = bdata;
   *obw = bw;
   *obh = bh;
   *ominutiae = minutiae;

   time_accum(total_timer, total_time);

   /******************/
   /* PRINT TIMINGS  */
   /******************/
   /* These Timings will print when TIMER is defined. */
   /* print MAP generation timing statistics */
   print_time(stderr, "TIMER: MAPS time   = %f (secs)\n", imap_time);
   /* print binarization timing statistics */
   print_time(stderr, "TIMER: Binarization time   = %f (secs)\n", bin_time);
   /* print minutia detection timing statistics */
   print_time(stderr, "TIMER: Minutia Detection time   = %f (secs)\n",
              minutia_time);
   /* print minutia removal timing statistics */
   print_time(stderr, "TIMER: Minutia Removal time   = %f (secs)\n",
              rm_minutia_time);
   /* print neighbor ridge count timing statistics */
   print_time(stderr, "TIMER: Neighbor Ridge Counting time   = %f (secs)\n",
              ridge_count_time);
   /* print total timing statistics */
   print_time(stderr, "TIMER: Total time   = %f (secs)\n", total_time);

   /* If LOG_REPORT defined, close log report file. */
   if((ret = close_logfile()))
      return(ret);

   return(0);
}

/*************************************************************************
**************************************************************************
#cat:   get_minutiae - Takes a grayscale fingerprint image, binarizes the input
#cat:                image, and detects minutiae points using LFS Version 2.
#cat:                The routine passes back the detected minutiae, the
#cat:                binarized image, and a set of image quality maps.

   Input:
      idata    - grayscale fingerprint image data
      iw       - width (in pixels) of the grayscale image
      ih       - height (in pixels) of the grayscale image
      id       - pixel depth (in bits) of the grayscale image
      ppmm     - the scan resolution (in pixels/mm) of the grayscale image
      lfsparms - parameters and thresholds for controlling LFS
   Output:
      ominutiae         - points to a structure containing the
                          detected minutiae
      oquality_map      - resulting integrated image quality map
      odirection_map    - resulting direction map
      olow_contrast_map - resulting low contrast map
      olow_flow_map     - resulting low ridge flow map
      ohigh_curve_map   - resulting high curvature map
      omap_w   - width (in blocks) of image maps
      omap_h   - height (in blocks) of image maps
      obdata   - points to binarized image data
      obw      - width (in pixels) of binarized image
      obh      - height (in pixels) of binarized image
      obd      - pixel depth (in bits) of binarized image
   Return Code:
      Zero     - successful completion
      Negative - system error
**************************************************************************/
int custom_get_minutiae(
                  int (*custom_dirmap)(int**, int, int, unsigned char*, const int, const int, int*, const int),
                  MINUTIAE **ominutiae, int **oquality_map,
                 int **odirection_map, int **olow_contrast_map,
                 int **olow_flow_map, int **ohigh_curve_map,
                 int *omap_w, int *omap_h,
                 unsigned char **obdata, int *obw, int *obh, int *obd,
                 unsigned char *idata, const int iw, const int ih,
                 const int id, const double ppmm, const LFSPARMS *lfsparms)
{
   int ret;
   MINUTIAE *minutiae;
   int *direction_map, *low_contrast_map, *low_flow_map;
   int *high_curve_map, *quality_map;
   int map_w, map_h;
   unsigned char *bdata;
   int bw, bh;

   /* If input image is not 8-bit grayscale ... */
   if(id != 8){
      fprintf(stderr, "ERROR : get_minutiae : input image pixel ");
      fprintf(stderr, "depth = %d != 8.\n", id);
      return(-2);
   }

   /* Detect minutiae in grayscale fingerpeint image. */
   if((ret = custom_lfs_detect_minutiae_V2(
                                    custom_dirmap,
                                    &minutiae,
                                   &direction_map, &low_contrast_map,
                                   &low_flow_map, &high_curve_map,
                                   &map_w, &map_h,
                                   &bdata, &bw, &bh,
                                   idata, iw, ih, lfsparms))){
      return(ret);
   }

   /* Build integrated quality map. */
   if((ret = gen_quality_map(&quality_map,
                            direction_map, low_contrast_map,
                            low_flow_map, high_curve_map, map_w, map_h))){
      free_minutiae(minutiae);
      free(direction_map);
      free(low_contrast_map);
      free(low_flow_map);
      free(high_curve_map);
      free(bdata);
      return(ret);
   }

   /* Assign reliability from quality map. */
   if((ret = combined_minutia_quality(minutiae, quality_map, map_w, map_h,
                                     lfsparms->blocksize,
                                     idata, iw, ih, id, ppmm))){
      free_minutiae(minutiae);
      free(direction_map);
      free(low_contrast_map);
      free(low_flow_map);
      free(high_curve_map);
      free(quality_map);
      free(bdata);
      return(ret);
   }

   /* Set output pointers. */
   *ominutiae = minutiae;
   *oquality_map = quality_map;
   *odirection_map = direction_map;
   *olow_contrast_map = low_contrast_map;
   *olow_flow_map = low_flow_map;
   *ohigh_curve_map = high_curve_map;
   *omap_w = map_w;
   *omap_h = map_h;
   *obdata = bdata;
   *obw = bw;
   *obh = bh;
   *obd = id;

   /* Return normally. */
   return(0);
}

/*************************************************************************
**************************************************************************/
static PyObject *py_in_func = NULL;

int custom_dirmap(int** direction_map, int mw, int mh, unsigned char* pdata, const int pw, const int ph, int* blkoffs, const int num_dir)
{
   // Variables
   //  PyObject *dir_array;
   PyObject *img_array, *blk_array, *result, *new_dir = NULL;
   npy_intp *dim;

   // Wrap the input image into an ndarray
   // Note: no need to decref the newly created PyObject, it does not own its data
   npy_intp img_dim[] = {ph, pw};
   img_array = PyArray_SimpleNewFromData(2, img_dim, NPY_UBYTE, pdata);

   // Create array of block offsets
   npy_intp blk_dim[] = {mh, mw};
   blk_array = PyArray_SimpleNewFromData(2, blk_dim, NPY_INT, blkoffs);

   // Call external Python function
   result = PyObject_CallFunctionObjArgs(py_in_func, img_array, blk_array, Py_BuildValue("i", num_dir), NULL);
   if (result == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "Error calling the given function on input");
      goto err;
   }
   // Convert result to numpy array
   new_dir = PyArray_FROM_OTF(result, NPY_INT, NPY_ARRAY_C_CONTIGUOUS);
   if (new_dir == NULL){
      PyErr_SetString(PyExc_RuntimeError, "Error converting the function's result to numpy array");
      goto err;
   }
   // Dimensional check
   dim = PyArray_DIMS( (PyArrayObject*)new_dir );
   if (dim[0]!=mh || dim[1]!=mw){
      PyErr_SetString(PyExc_RuntimeError, "Result dimensions mismatch");
      goto err;
   }
   // Replace the values of direction_map with the new ones
   free(*direction_map);
   *direction_map = (npy_int*)PyArray_DATA( (PyArrayObject*)new_dir );
   PyArray_CLEARFLAGS( (PyArrayObject*)new_dir, NPY_ARRAY_OWNDATA );

   goto out;
err:
	Py_XDECREF(new_dir);
   Py_XDECREF(result);
	Py_XDECREF(img_array);
	Py_XDECREF(blk_array);
   return 1;
out:
	Py_XDECREF(new_dir);
   Py_XDECREF(result);
	Py_XDECREF(img_array);
	Py_XDECREF(blk_array);
   return 0;
}

int minutiae_to_python(PyObject **min_list, MINUTIAE *minutiae)
{
   PyObject *list, *element, *nbrs;
   MINUTIA *minutia;
   // Create the list of minutiae
   if ((list = PyList_New(minutiae->num)) == NULL){
      PyErr_SetString(PyExc_RuntimeError, "Cannot create minutiae list");
      goto err;
   }
   
   for(int i = 0; i < PyList_Size(list); ++i){
      minutia = minutiae->list[i];
      // Create new dictionary with neighbouring minutiae
      if ((nbrs = PyDict_New()) == NULL){
         PyErr_SetString(PyExc_RuntimeError, "Cannot create neighbouring minutiae dictionary");
         goto err;
      }
      // Add neighbours ids as keys and ridge_counts as values
      for (int j = 0; j < minutia->num_nbrs; ++j){
         if (PyDict_SetItem(nbrs, Py_BuildValue("i", minutia->nbrs[j]), Py_BuildValue("i", minutia->ridge_counts[j]))){
            PyErr_SetString(PyExc_RuntimeError, "Cannot set item in neighbouring minutiae dictionary");
            goto err;
         }
      }
      
      // Create new dictionary to hold minutia's data
      if ((element = Py_BuildValue("{sisisisisisfsisisisN}",
            "x", minutia->x,
            "y", minutia->y,
            "ex", minutia->ex,
            "ey", minutia->ey,
            "direction", minutia->direction,
            "reliability", minutia->reliability,
            "type", minutia->type,
            "appearing", minutia->appearing,
            "feature_id", minutia->feature_id,
            "nbrs", nbrs
            )) == NULL){
         PyErr_SetString(PyExc_RuntimeError, "Cannot create a new dictionary for the minutia");
         goto err;
      }

      // Assign the newly create list element to the list
      if (PyList_SetItem(list, i, element)) {
         PyErr_SetString(PyExc_RuntimeError, "Cannot assign new element to the list of minutiae");
         goto err;
      }
   }

   goto out;
err:
   return 1;
out:
   *min_list = list;
   return 0;
}

PyObject* nbis_mindtct(PyObject *self, PyObject *args, PyObject *kwargs)
{
   PyObject *input, *pyfun, *in_array = NULL;
   npy_intp *idim;
   npy_ubyte *idata;
   int ih, iw, id; // input image specs
   double ippmm; // the scan resolution (in pixels/mm) of the grayscale image

   int bw, bh, bd; // binarized image data specs
   unsigned char *bdata; // output binarized image data
   int map_w, map_h; // dimensions of output maps
   int *direction_map, *low_contrast_map, *low_flow_map, *high_curve_map, *quality_map; // output maps
   MINUTIAE *minutiae = NULL; // list of minutiae
   PyObject *out_bdata, *out_direction_map, *out_low_contrast_map, *out_low_flow_map, *out_high_curve_map, *out_quality_map, *list_min;

   bool boostflag = 0;

   // Define possible keyword arguments
	static char *kwlist[] = 
	{
		"input", // input fingerprint image
		"fun", // function for LRO computation
		"contrast_boost", // whether to enhance the image contrast before extracting the orientation field
		NULL
	};

   // Parse arguments
   if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, 
      "OO|$p", kwlist, 
      &input, &pyfun,
      &boostflag)) {
      goto err;
   }

   // Convert the first argument to a numpy array
	if ((in_array = PyArray_FROM_OTF(input, NPY_UBYTE, NPY_ARRAY_C_CONTIGUOUS)) == NULL){
      PyErr_SetString(PyExc_TypeError, "Need a numpy array as 'input' argument!");
      goto err;
   }

   // Make sure second argument is a function
   if (!PyCallable_Check(pyfun)) {
      PyErr_SetString(PyExc_TypeError, "Need a callable object as 'fun' argument!");
      goto err;
   }
   py_in_func = pyfun;

   /* 1. Take data from input fingerprint */
   idim = PyArray_DIMS( (PyArrayObject*)in_array );
   ih = idim[0];
   iw = idim[1];
   id = 8; // set image depth to 8 bit (it is guaranteed by the array structure with NPY_UBYTE)
   ippmm = DEFAULT_PPI / (double)MM_PER_INCH;
   idata = (npy_ubyte*)PyArray_DATA( (PyArrayObject*)in_array );

   /* 2. ENHANCE IMAGE CONTRAST IF REQUESTED */
   if(boostflag) {
      trim_histtails_contrast_boost(idata, iw, ih); 
   }

   /* 3. GET MINUTIAE & BINARIZED IMAGE. */
   if(custom_get_minutiae(
      custom_dirmap,
      &minutiae, &quality_map, &direction_map,
      &low_contrast_map, &low_flow_map, &high_curve_map,
      &map_w, &map_h, &bdata, &bw, &bh, &bd,
      idata, iw, ih, id, ippmm, &lfsparms_V2)){
         if (!PyErr_Occurred()){
            PyErr_SetString(PyExc_RuntimeError, "Cannot extract minutiae from input fingerprint");
         }
      goto err;
   }

   /* 4. Convert the maps to numpy arrays */

   // Assemble the binarized image as a numpy array
   npy_intp bin_dim[] = {bh, bw};
   out_bdata = PyArray_SimpleNewFromData(2, bin_dim, NPY_UBYTE, bdata);
   if (out_bdata == NULL){
      PyErr_SetString(PyExc_RuntimeError, "Problems with out_bdata");
      goto err;
   }
   PyArray_ENABLEFLAGS( (PyArrayObject*)out_bdata, NPY_ARRAY_OWNDATA );
   Py_XINCREF(out_bdata);

   // Assemble the direction map as a numpy array
   npy_intp map_dim[] = {map_h, map_w};
   out_direction_map = PyArray_SimpleNewFromData(2, map_dim, NPY_INT, direction_map);
   if (out_direction_map == NULL){
      PyErr_SetString(PyExc_RuntimeError, "Problems with out_direction_map");
      goto err;
   }
   PyArray_ENABLEFLAGS( (PyArrayObject*)out_direction_map, NPY_ARRAY_OWNDATA );
   Py_XINCREF(out_direction_map);

   // Assemble the low contrast map as a numpy array
   out_low_contrast_map = PyArray_SimpleNewFromData(2, map_dim, NPY_INT, low_contrast_map);
   if (out_low_contrast_map == NULL){
      PyErr_SetString(PyExc_RuntimeError, "Problems with out_low_contrast_map");
      goto err;
   }
   PyArray_ENABLEFLAGS( (PyArrayObject*)out_low_contrast_map, NPY_ARRAY_OWNDATA );
   Py_XINCREF(out_low_contrast_map);

   // Assemble the low flow map as a numpy array
   out_low_flow_map = PyArray_SimpleNewFromData(2, map_dim, NPY_INT, low_flow_map);
   if (out_low_flow_map == NULL){
      PyErr_SetString(PyExc_RuntimeError, "Problems with out_low_flow_map");
      goto err;
   }
   PyArray_ENABLEFLAGS( (PyArrayObject*)out_low_flow_map, NPY_ARRAY_OWNDATA );
   Py_XINCREF(out_low_flow_map);

   // Assemble the high curvature map as a numpy array
   out_high_curve_map = PyArray_SimpleNewFromData(2, map_dim, NPY_INT, high_curve_map);
   if (out_high_curve_map == NULL){
      PyErr_SetString(PyExc_RuntimeError, "Problems with out_high_curve_map");
      goto err;
   }
   PyArray_ENABLEFLAGS( (PyArrayObject*)out_high_curve_map, NPY_ARRAY_OWNDATA );
   Py_XINCREF(out_high_curve_map);

   // Assemble the quality map as a numpy array
   out_quality_map = PyArray_SimpleNewFromData(2, map_dim, NPY_INT, quality_map);
   if (out_quality_map == NULL){
      PyErr_SetString(PyExc_RuntimeError, "Problems with out_quality_map");
      goto err;
   }
   PyArray_ENABLEFLAGS( (PyArrayObject*)out_quality_map, NPY_ARRAY_OWNDATA );
   Py_XINCREF(out_quality_map);

   // Assemble the minutiae list
   if (minutiae_to_python(&list_min, minutiae)){
         if (!PyErr_Occurred()){
            PyErr_SetString(PyExc_RuntimeError, "Cannot convert minutiae to Python list of dictionary");
         }
      goto err;
   }

   goto out;
err:
   Py_XDECREF(in_array);
   if(minutiae) free_minutiae(minutiae);
   return NULL;
out:
   Py_XDECREF(in_array);
   if(minutiae) free_minutiae(minutiae);
	return Py_BuildValue("NNNNNNN", out_bdata, out_direction_map, out_low_contrast_map, out_low_flow_map, out_high_curve_map, out_quality_map, list_min);
}
