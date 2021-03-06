#include "../Headers/ang_seg_wrapper.hpp"


char* segmentation(unsigned char* data, const intptr_t* dim,
	const _ImageBimodalize ImageBimodalize,
	const _ImageCroppingSimple ImageCroppingSimple,
	const _TopMask TopMask,
	const _ImageSignificantMask ImageSignificantMask,
	const _ImageEqualize ImageEqualize,
    unsigned char* enh_img_data, unsigned char* fg_mask_data,
	const int enhance_only)
{
	// Declasrations
	char *msg = NULL;
	fm::Mask__ imageb, mask; // working variables
	fm::Image__ imaged;

	try {
		// Convert to OpenCV matrix
		imageb = fm::Mask__(dim[0], dim[1]);
		memcpy(imageb.data, data, imageb.elemSize() * dim[0] * dim[1]);
		imageb.convertTo(imaged, CV_64F);
		fm::rescale(imaged);
		
		// Set the initial mask
		mask = fm::Mask__::ones(imaged.size()) * MASK_TRUE;
		// Processing
		fm::imageBimodalize(imaged, imaged,
							ImageBimodalize.brightness,
							ImageBimodalize.leftCut,
							ImageBimodalize.rightCut,
							ImageBimodalize.histSmooth,
							ImageBimodalize.reparSmooth);
		if (!enhance_only) {
			fm::ImageCroppingSimple(imaged, mask,
								ImageCroppingSimple.minVariation,
								ImageCroppingSimple.marg);
			fm::ImageCroppingLines(imaged, mask,
								TopMask.scanAreaAmount,
								TopMask.gradientFilterWidth,
								TopMask.gaussianFilterSide,
								TopMask.binarizationLevel,
								TopMask.f,
								TopMask.slopeAngle,
								TopMask.lev,
								TopMask.marg);
			fm::ImageSignificantMask(imaged, mask,
								ImageSignificantMask.medFilterSide,
								ImageSignificantMask.gaussFilterSide,
								ImageSignificantMask.minFilterSide,
								ImageSignificantMask.binLevVarMask,
								ImageSignificantMask.dilate1RadiusVarMask,
								ImageSignificantMask.erodeRadiusVarMask,
								ImageSignificantMask.dilate2RadiusVarMask,
								ImageSignificantMask.maxCompNumVarMask,
								ImageSignificantMask.minCompThickVarMask,
								ImageSignificantMask.maxHolesNumVarMask,
								ImageSignificantMask.minHolesThickVarMask,
								ImageSignificantMask.histeresisThreshold1Gmask,
								ImageSignificantMask.histeresisThreshold2Gmask,
								ImageSignificantMask.radiusGaussFilterGmask,
								ImageSignificantMask.minMeanIntensityGmask,
								ImageSignificantMask.dilate1RadiusGmask,
								ImageSignificantMask.erodeRadiusGmask,
								ImageSignificantMask.dilate2RadiusGmask,
								ImageSignificantMask.maxCompNumGmask,
								ImageSignificantMask.minCompThickGmask,
								ImageSignificantMask.maxHolesNumGmask,
								ImageSignificantMask.minHolesThickGmask,
								ImageSignificantMask.histeresisThreshold3Gmask,
								ImageSignificantMask.histeresisThreshold4Gmask,
								ImageSignificantMask.dilate3RadiusGmask,
								ImageSignificantMask.erode2RadiusGmask,
								ImageSignificantMask.histeresisThreshold5Gmask,
								ImageSignificantMask.histeresisThreshold6Gmask,
								ImageSignificantMask.dilate4RadiusGmask,
								ImageSignificantMask.radiusGaussFilterComp,
								ImageSignificantMask.meanIntensityCompThreshold,
								ImageSignificantMask.dilateFinalRadius,
								ImageSignificantMask.erodeFinalRadius,
								ImageSignificantMask.smoothFinalRadius,
								ImageSignificantMask.maxCompNumFinal,
								ImageSignificantMask.minCompThickFinal,
								ImageSignificantMask.maxHolesNumFinal,
								ImageSignificantMask.minHolesThickFinal,
								ImageSignificantMask.fixedFrameWidth,
								ImageSignificantMask.smooth2FinalRadius);
			mask = mask > 0; // Make sure mask is 0,255 valued
		}
		fm::ImageEqualize(imaged, imaged,
						ImageEqualize.minMaxFilter,
						ImageEqualize.mincp1,
						ImageEqualize.mincp2,
						ImageEqualize.maxcp1,
						ImageEqualize.maxcp2,
						mask);

		// Convert bask to unsigned char matrix
		fm::rescale(imaged, 255.0);
		imaged.convertTo(imageb, CV_8U);

		// Copy to destination pointers
		memcpy(enh_img_data, imageb.data, sizeof(unsigned char)*dim[0]*dim[1]);
		memcpy(fg_mask_data, mask.data, sizeof(unsigned char)*dim[0]*dim[1]);
	} catch(const std::exception& ex) {
		msg = (char*)malloc(sizeof(char)*300);
		sprintf(msg, "Exception caught while processing the input image: %s\n", ex.what());
	} catch(...) {
		msg = (char*)malloc(sizeof(char)*300);
		sprintf(msg, "Generic exception caught while processing the input image\n");
	}

	return msg;
}