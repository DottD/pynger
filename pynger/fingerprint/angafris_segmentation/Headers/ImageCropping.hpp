/*
 
 In questo header file sono presenti le definizioni delle funzioni che si occupano dello scontornamento
 dell'immagine.
 
 */

#ifndef ImageCropping_hpp
#define ImageCropping_hpp

/* Libreria OpenCV */
#include <opencv2/opencv.hpp>
/* Libreria personale */
#include "myMathFunc.hpp" 

namespace fm {
    
    /*
     ImageCroppingLines
     
     Questa funzione elimina dalla maschera le zone che stanno tra le linee individuate
     ed il bordo dell'immagine.
     La maschera da passare in input deve essere già inizializzta; inoltre si faccia caso
     che viene eseguita l'intersezione della maschera passata con quella generata dalla
     funzione.
     
     */
    void ImageCroppingLines(const Image__&, // matrice di input
                            Mask__&, // maschera con la porzione di immagine scontornata
                            const double& scanAreaAmount, // porzione di ricerca bordo
                            const double& gradientFilterWidth, // ampiezza filtro gradiente
                            const int& gaussianFilterSide, // ampiezza filtro gaussiano
                            const double& binarizationLevel, // livello binarizationLevelbinarizzazione
                            const double& f, // fattore del livello per inizio ricerca
                            const double& slopeAngle, // inclinazione massima bordo in gradi
                            const double& lev, // livello individuazione bordo
                            const int& marg); // margine di taglio in pixel
    
    
    /*
     TopMask
     
     */
    void TopMask(const Image__&, // matrice di input
                 Mask__&, // maschera con la porzione di immagine scontornata
                 const double& scanAreaAmount, // porzione di ricerca bordo
                 const double& gradientFilterWidth, // ampiezza filtro gradiente
                 const double& binarizationLevel, // livello binarizzazione
                 const double& f, // fattore del livello per inizio ricerca
                 const double& slopeAngle, // inclinazione massima bordo in gradi
                 const double& lev, // livello individuazione bordo
                 const int& marg); // margine di taglio in pixel
    
    
    /*
     ImageCroppingSimple
     
     La funzione scorre l'immagine dal margine verso l'interno ed elimina le linee esaminate a meno che
     la loro variazione (max-min) non sia superiore ad un valore prestabilito.
     
     Argomenti:
     - input immagine di partenza
     - output maschera generata: viene eseguita l'intersezione della nuova maschera con essa, quindi
            deve essere già allocata
     
     */
    void ImageCroppingSimple(const Image__& src, // input
                             Mask__& dst, // output
                             const double& minVariation, // variazione minima affinché la linea sia considerata parte dell'impronta
                             const int& marg); // margine aggiuntivo
}

#endif /* ImageCropping_hpp */
