/*
 In questo header file vengono definite alcune operazioni sulle maschere
 */

#ifndef ImageMaskSimplify_hpp
#define ImageMaskSimplify_hpp

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include "myMathFunc.hpp"

namespace fm {
    enum struct holesFilling {
        THICKNESS, // max distance of an internal point to the perimeter of a component
        AREA // area of a component
    }; // never used ... to be implemented
    
    /*
     MaskSimplify(mask, maxCompNum, minCompThickness, maxHolesNum, minHolesThickness)
     
     Questa funzione è un alias per
     MaskComponentsReduce (mask, maxCompNum, minCompThickness);
     MaskHolesFill (mask, maxHolesNum, minHolsThickness);
     */
    void MaskSimplify(Mask__& mask,
                      const int& maxCompNum,
                      const int& minCompThickness,
                      const int& maxHolesNum,
                      const int& minHolesThickness);
    
    /*
     MaskComponentsReduce (mask, maxCompNum, minThickness)
     
     Questa funzione riduce il numero di componenti connesse della maschera data in input;
     le componenti con massima distanza dal perimetro troppo piccola vengono eliminate a priori,
     quindi le rimanenti vengono ordinate in base alla loro area e si scartano quelle
     più piccole.
     
     Argomenti:
     - mask la maschera su cui si vuole operare
     - maxCompNum numero massimo di componenti che si vogliono tenere (inserire un numero negativo per non forzare un numero massimo di componenti)
     - minThickness spessore minimo che deve avere una componente
     
     */
    void MaskComponentsReduce(Mask__& mask, // la maschera su cui si vuole operare
                              const int& maxCompNum, // maxCompNum numero massimo di componenti che si vogliono tenere
                              const int& minThickness); // spessore minimo che deve avere una componente
    
    /*
     MaskHolesFill (mask, maxCompNum, minThickness)
     
     Questa funzione elimina i buchi nella maschera passata in input: vengono preservati
     solamente i maxCompNum buchi con maggiore distanza dal baricentro e
     che hanno una massima distanza dal perimetro sufficiente.
     
     Argomenti:
     - mask la maschera su cui si vuole operare
     - maxCompNum numero massimo di componenti che si vogliono tenere (inserire un numero negativo per non forzare un numero massimo di buchi)
     - minThickness spessore minimo che deve avere una componente
     
     */
    void MaskHolesFill(Mask__& mask, // la maschera su cui si vuole operare
                       const int& maxCompNum, // maxCompNum numero massimo di componenti che si vogliono tenere
                       const int& maxPerimeterDistance); // minThickness spessore minimo che deve avere una componente
    
    void MaskHolesFillArea(cv::InputOutputArray& mask, // la maschera su cui si vuole operare
                           const int& maxCompNum, // maxCompNum numero massimo di componenti che si vogliono tenere
                           const int& maxArea); // maxArea massima area di un buco per essere colmato
}

#endif /* ImageMaskSimplify_hpp */
