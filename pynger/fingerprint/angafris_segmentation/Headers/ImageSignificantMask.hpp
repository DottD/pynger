/*
 Questo header file contiene la definizione delle procedure che consentono di ottenere
 la parte significativa dell'impronta digitale
 */

#ifndef ImageSignificantMask_hpp
#define ImageSignificantMask_hpp

#include <exception>
#include <opencv2/opencv.hpp>
#include "ImageMaskSimplify.hpp"
#include "myMathFunc.hpp"

namespace fm {
    /*
     ImageSignificantMask(src, mask, ...)
     
     
     Argomenti:
     - src immagine di partenza
     - mask maschera finale: la funzione interseca i risultati con i dati contenuti precedentemente
            nella maschera, quindi essa deve essere preallocata ed avere valori in {0,1}
     
     */
    void ImageSignificantMask(const cv::Mat1d& src,
                              Mask__& mask,
                              const int& medFilterSide,
                              const int& gaussFilterSide,
                              const int& minFilterSide,
                              const double& binLevVarMask,
                              const int& dilate1RadiusVarMask,
                              const int& erodeRadiusVarMask,
                              const int& dilate2RadiusVarMask,
                              const int& maxCompNumVarMask,
                              const int& minCompThickVarMask,
                              const int& maxHolesNumVarMask,
                              const int& minHolesThickVarMask,
                              const int& histeresisThreshold1,
                              const int& histeresisThreshold2,
                              const int& radiusGaussFilterGmask,
                              const double& minMeanIntensityGmask,
                              const int& dilate1RadiusGmask,
                              const int& erodeRadiusGmask,
                              const int& dilate2RadiusGmask,
                              const int& maxCompNumGmask,
                              const int& minCompThickGmask,
                              const int& maxHolesNumGmask,
                              const int& minHolesThickGmask,
                              const int& histeresisThreshold3Gmask,
                              const int& histeresisThreshold4Gmask,
                              const int& dilate3RadiusGmask,
                              const int& erode2RadiusGmask,
                              const int& histeresisThreshold5Gmask,
                              const int& histeresisThreshold6Gmask,
                              const int& dilate4RadiusGmask,
                              const int& radiusGaussFilterComp,
                              const double& meanIntensityCompThreshold,
                              const int& dilateFinalRadius,
                              const int& erodeFinalRadius,
                              const int& smoothFinalRadius,
                              const int& maxCompNumFinal,
                              const int& minCompThickFinal,
                              const int& maxHolesNumFinal,
                              const int& minHolesThickFinal,
                              const int& fixedFrameWidth,
                              const int& smooth2FinalRadius);
    
    /*
     MaskSmoothDilate(mask, r)
     
     Questa funzione esegue un'espansione della maschera in src, non tramite
     massimo, ma facendo uno smussamento gaussiano di raggio r, quindi applicando
     una soglia a 0.05.
     
     Argomenti:
     - mask immagine binaria di partenza
     - r raggio dello smussamento gaussiano
     
     */
    void MaskSmoothDilate(Mask__& mask, const int& r);
	void MaskSmoothDilate(cv::Mat1f& mask, const int& r);
	
    /*
     MaskSmoothErode(mask, r)
     
     Questa funzione esegue una contrazione della maschera in src, non tramite
     minimo, ma facendo uno smussamento gaussiano di raggio r, quindi applicando
     una soglia a 0.95.
     
     Argomenti:
     - mask immagine binaria di partenza
     - r raggio dello smussamento gaussiano
     
     */
    void MaskSmoothErode(Mask__& mask, const int& r);
    void MaskSmoothErode(cv::Mat1f& mask, const int& r);
	
    /*
     MaskSmoothErodeDist(mask, lev)
     
     This function contracts the mask so that every pixel with a distance from the background smaller than lev is set to 0.
     */
    void MaskSmoothErodeDist(cv::InputOutputArray mask, const float& lev);
    
    /*
     MaskSmoothDilateDist(mask, lev)
     
     This function dilates the mask so that every background pixel with a distance from the foreground smaller than lev is set to 1.
     */
    void MaskSmoothDilateDist(cv::InputOutputArray mask, const float& lev);
    
    /*
     MaskFilterByMaxCentroidDistance(inMask, outMask, maxVal)
     
     Set to background the pixels of an inMask connected component that has maximum distance from its centroid below maxVal.
     Arguments:
     - inMask           initial mask
     - outMask          the resulting mask (CV_32F)
     - maxVal           minimum maximum distance from centroid
     */
//    void MaskFilterByMaxCentroidDistance(cv::InputArray inMask, cv::OutputArray outMask, const float& maxVal);
    
    /*
     MaskSmoothing(mask, r)
     
     Questa funzione esegue uno smussamento della maschera in src facendo
     uno smussamento gaussiano di raggio r, quindi applicando una soglia a 0.5.
     
     Argomenti:
     - mask immagine binaria di partenza
     - r raggio dello smussamento gaussiano
     
     */
    void MaskSmoothing(Mask__& mask, const int& r);
    
    /*
     FilterComponentByIntensity(mask, intensity, threshold)
     
     Questa funzione riduce la maschera passata in input, eliminando le componenti connesse la cui
     intensità media non raggiunge il livello di soglia threshold richiesto. Per calcolare 
     l'intensità media di ogni componente si utilizza la matrice intensity, in cui ad ogni pixel
     è associato un valore di intensità.
     
     Argomenti:
     - mask maschera di input/output (CV_8U)
     - intensity immagine utilizzata per calcolare l'intensità media (CV_32F)
     - threshold valore di soglia richiesto per accettare una componente
     
     */
    void FilterComponentByIntensity(Mask__& mask, const cv::Mat1f& intensity, const double& threshold);
}

#endif /* ImageSignificantMask_hpp */
