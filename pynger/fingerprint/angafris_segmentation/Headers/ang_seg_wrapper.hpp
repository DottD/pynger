#ifdef __cplusplus
    #include "myMathFunc.hpp"
    #include "AdaptiveThreshold.hpp"
    #include "ImageSignificantMask.hpp"
    #include "ImageRescale.hpp"
    #include "ImageNormalization.hpp"
    #include "ImageMaskSimplify.hpp"
    #include "ImageCropping.hpp"

    #include <exception>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	double brightness;
	double leftCut;
	double rightCut;
	int histSmooth;
	int reparSmooth;
} _ImageBimodalize;
typedef struct {
	double minVariation;
	int marg;
} _ImageCroppingSimple;
typedef struct {
	double scanAreaAmount;
	double gradientFilterWidth;
	int gaussianFilterSide;
	double binarizationLevel;
	double f;
	double slopeAngle;
	double lev;
	int marg;
} _TopMask;
typedef struct {
	int medFilterSide;
	int gaussFilterSide;
	int minFilterSide;
	double binLevVarMask;
	int dilate1RadiusVarMask;
	int erodeRadiusVarMask;
	int dilate2RadiusVarMask;
	int maxCompNumVarMask;
	int minCompThickVarMask;
	int maxHolesNumVarMask;
	int minHolesThickVarMask;
	int histeresisThreshold1Gmask;
	int histeresisThreshold2Gmask;
	int radiusGaussFilterGmask;
	double minMeanIntensityGmask;
	int dilate1RadiusGmask;
	int erodeRadiusGmask;
	int dilate2RadiusGmask;
	int maxCompNumGmask;
	int minCompThickGmask;
	int maxHolesNumGmask;
	int minHolesThickGmask;
	int histeresisThreshold3Gmask;
	int histeresisThreshold4Gmask;
	int dilate3RadiusGmask;
	int erode2RadiusGmask;
	int histeresisThreshold5Gmask;
	int histeresisThreshold6Gmask;
	int dilate4RadiusGmask;
	int radiusGaussFilterComp;
	double meanIntensityCompThreshold;
	int dilateFinalRadius;
	int erodeFinalRadius;
	int smoothFinalRadius;
	int maxCompNumFinal;
	int minCompThickFinal;
	int maxHolesNumFinal;
	int minHolesThickFinal;
	int fixedFrameWidth;
	int smooth2FinalRadius;
} _ImageSignificantMask;
typedef struct {
	int minMaxFilter;
	double mincp1;
	double mincp2;
	double maxcp1;
	double maxcp2;
} _ImageEqualize;

char* segmentation(const unsigned char* data, const long* dim,
	const _ImageBimodalize ImageBimodalize,
	const _ImageCroppingSimple ImageCroppingSimple,
	const _TopMask TopMask,
	const _ImageSignificantMask ImageSignificantMask,
	const _ImageEqualize ImageEqualize, 
    unsigned char* enh_img_data, unsigned char* fg_mask_data);
	
#ifdef __cplusplus
}
#endif