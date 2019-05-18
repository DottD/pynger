#include "../Headers/ang_seg_wrapper.hpp"


char* segmentation(unsigned char* data, const long* dim,
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
	cv::Mat1b imageb, mask; // working variables
	cv::Mat1d imaged;

	try {
		// Convert to OpenCV matrix
		imageb = cv::Mat1b(dim[0], dim[1]);
		memcpy(imageb.data, data, imageb.elemSize() * dim[0] * dim[1]);
		imageb.convertTo(imaged, CV_64F, 1, 0);
		
		// Set the initial mask
		mask = cv::Mat1b::ones(imaged.size()) * MASK_TRUE;
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
		imaged.convertTo(imageb, CV_8U, 255, 0);

		// Copy to destination pointers
		memcpy(enh_img_data, imageb.data, sizeof(unsigned char)*dim[0]*dim[1]);
		memcpy(fg_mask_data, mask.data, sizeof(unsigned char)*dim[0]*dim[1]);
	} catch(const std::exception& ex) {
		msg = (char*)malloc(sizeof(char)*300);
		sprintf(msg, "Exception catched while processing the input image: %s\n", ex.what());
	} catch(...) {
		msg = (char*)malloc(sizeof(char)*300);
		sprintf(msg, "Generic exception catched while processing the input image\n");
	}

	return msg;
}