/*
 File provvisorio per le definizioni di funzioni matematiche utili
 */

#ifndef scontornamento_hpp
#define scontornamento_hpp

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <map>
#include <exception>
// #define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include "myMathFunc.hpp"

namespace fm {
    typedef std::map<double,double> setOfPointsType_;
    typedef setOfPointsType_::value_type pointType_;
    /*
     imageNormalize(src,dst,[Argomenti opzionali])
     
     Funzione che normalizza l'immagine passata in input.
     
     Argomenti
     - src immagine di partenza (eventualmente convertita in CV_8U)
     - dst immagine finale
     - brightness parametro al cui crescere aumenta la luminosità finale dell'immagine; serve a calcolare il valore che verrà normalizzato a 0.5
     - leftCut porzione della coda sinistra da tenere
     - rightCut porzione della coda destra da tenere
     - histSmooth ampiezza smussamento gaussiano dell'istogramma
     - reparSmooth ampiezza smussamento gaussiano dell'istogramma dopo riparametrizzazione
     - minMaxFilter grandezza del filtro per ottenere l'immagine dei minimi e dei massimi
     - eqSmooth ampiezza smussamento gaussiano dopo equalizzazione
     */
    void imageNormalize(const Image__& src,
                        Image__& dst,
                        const double& brightness,
                        const double& leftCut,
                        const double& rightCut,
                        const int& histSmooth,
                        const int& reparSmooth,
                        const int& minMaxFilter,
                        const double& mincp1,
                        const double& mincp2,
                        const double& maxcp1,
                        const double& maxcp2);
    
    /*
     imageBimodalize(src,dst,[Argomenti opzionali])
     
     Funzione che regolarizza il contrasto dell'immagine.
     src e dst possono essere la stessa funzione.
     
     Argomenti
     - src immagine di partenza (eventualmente convertita in CV_8U)
     - dst immagine finale
     - brightness parametro al cui crescere aumenta la luminosità finale dell'immagine; serve a calcolare il valore che verrà normalizzato a 0.5
     - leftCut porzione della coda sinistra da tenere
     - rightCut porzione della coda destra da tenere
     - histSmooth ampiezza smussamento gaussiano dell'istogramma
     - reparSmooth ampiezza smussamento gaussiano dell'istogramma dopo riparametrizzazione
     */
    void imageBimodalize(const Image__& src,
                         Image__& dst,
                        const double& brightness,
                        const double& leftCut,
                        const double& rightCut,
                        const int& histSmooth,
                        const int& reparSmooth);
    
    /*
     ImageEqualize(src, dst, minMaxFilter, mincp1, mincp2, maxcp1, maxcp2)
     
     Funzione che equalizza l'immagine. src e dst possono essere la stessa matrice.
     
     Argomenti
     - src immagine di input
     - dst immagine di output
     - minMaxFilter ampiezza dei filtri circolari di minimo e di massimo
     - mincp1, mincp2, maxcp1, maxcp2 ascissa dei punti di controllo 
            per il riscalamento dei minimi e dei massimi
     */
    void ImageEqualize(const Image__& src,
                       Image__& dst,
                       const int& minMaxFilter,
                       const double& mincp1,
                       const double& mincp2,
                       const double& maxcp1,
                       const double& maxcp2,
                       cv::InputArray& mask = cv::noArray());
}

#endif /* scontornamento_hpp */
