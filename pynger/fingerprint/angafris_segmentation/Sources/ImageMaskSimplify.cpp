#include "../Headers/ImageMaskSimplify.hpp"

void fm::MaskSimplify(Mask__& mask,
                      const int& maxCompNum,
                      const int& minCompThickness,
                      const int& maxHolesNum,
                      const int& minHolesThickness)
{
    MaskComponentsReduce(mask, maxCompNum, minCompThickness);
    MaskHolesFill(mask, maxHolesNum, minHolesThickness);
}

void fm::MaskComponentsReduce(Mask__& mask,
                              const int& maxCompNum,
                              const int& minThickness)
{
    /* Dichiarazioni */
    int i = 0, j = 0; // variabili di controllo per i cicli
    
    cv::Mat1i labels; // matrice con i label delle componenti connesse
    cv::Mat1i::iterator labelsIterator; // iteratore a tale matrice
    
    cv::Mat1i stats; // matrice contenente informazioni su ogni componente connessa
    cv::Mat1d centroids; // matrice contenente informazioni sui baricentri delle componenti
    
    cv::Mat1f distFromBorder; // matrice con le distanze di ogni pixel dal bordo della componente di appartenenza
    cv::Mat1f::iterator dfbit; // iteratore alla matrice distanceFromBorder
    
    std::vector<int> keepLabels; // vettore con gli indici delle componenti da tenere
    std::vector<int>::iterator keepLabelsIt; // iteratore a tale vettore
    
    std::map<int, double> maxPerimeterDistance; // struttura in cui l'indice di ogni componente è associato alla massima distanza dal bordo
    std::map<int, double>::iterator mpdit; // iteratore alla struttura precedente
    
    /* Calcolo le componenti connesse della maschera */
    cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8 /* connectivity */, CV_32S);
    
    /* Per ogni componente calcolo la massima distanza dei pixel dal bordo */
    cv::distanceTransform(mask, distFromBorder,
                          cv::DistanceTypes::DIST_L2, // tipo di distanza da usare
                          cv::DistanceTransformMasks::DIST_MASK_5, // grandezza della maschera da utilizzare
                          CV_32F);
    for (dfbit = distFromBorder.begin(), labelsIterator = labels.begin();
         labelsIterator != labels.end();
         labelsIterator++, dfbit++)
    {
        mpdit = maxPerimeterDistance.find(*labelsIterator);
        if (mpdit != maxPerimeterDistance.end()) // label presente
        {
            // aggiorno la massima distanza dal perimetro utilizzando il pixel corrente in distanceFromBorder
            std::get<1>(*mpdit) = std::max<double>(std::get<1>(*mpdit) /*cast to double*/, *dfbit);
        }
        else // label non presente: lo aggiungo
            maxPerimeterDistance.insert(std::pair<int, double>(*labelsIterator, *dfbit /*cast to double*/));
    }
    
    /* Elimino da maxPerimeterDistance le componenti che non soddisfano i requisiti,
     mentre inserisco in keepLabelsDistance quelle che li soddisfano */
    for (mpdit = maxPerimeterDistance.begin(); mpdit != maxPerimeterDistance.end();)
    {
        if (std::get<1>(*mpdit) < minThickness) // non ha lo spessore minimo
            // elimino l'elemento corrente (l'iteratore ora punta al successivo elemento -> non bisogna incrementarlo)
            mpdit = maxPerimeterDistance.erase(mpdit);
        else
        {
            keepLabels.push_back(std::get<0>(*mpdit)); // aggiungo l'elemento corrente a keepLabelsSorted
            mpdit++; // incremento l'iteratore
        }
    }
    
    /* Se maxCompNum < 0 non esiste limite al numero di componenti */
    if ((maxCompNum >= 0)&&(maxCompNum < (int)keepLabels.size()))
    {
        /* Ordino le componenti connesse per area (in ordine discendente) */
        std::sort(keepLabels.begin(), keepLabels.end(), [&stats] (const int& a, const int& b)
                  {
                      return stats(a,cv::CC_STAT_AREA) > stats(b,cv::CC_STAT_AREA);
                  });
        
        /* Prendo solamente le prime maxCompNum componenti */
        keepLabelsIt = keepLabels.begin();
        for (i = 0; i < maxCompNum; i++) keepLabelsIt++;
        keepLabels.erase(keepLabelsIt, keepLabels.end());
    }
    
    /* Modifico la maschera per tenere solamente le componenti in keepLabelsSorted */
    for (i = 0; i < mask.rows; i++)
    {
        for (j = 0; j < mask.cols; j++)
        {
            if (std::find(keepLabels.begin(), keepLabels.end(), labels(i,j)) != keepLabels.end()) mask(i,j) = 255;
            else mask(i,j) = 0;
        }
    }
}

void fm::MaskHolesFill(Mask__& mask,
                       const int& maxHolesNum,
                       const int& maxPerimeterDistance)
{
    /* Dichiarazioni */
    int i = 0, j = 0; // variabili di controllo per i cicli
    
    cv::Mat1i labels; // matrice con i label delle componenti connesse
    cv::Mat1i::iterator labelsIterator; // iteratore a tale matrice
    
    std::vector<int> keepLabelsSorted; // vettore con gli indici delle componenti da tenere
    std::vector<int> borderRegions; // vettore con gli indici delle componenti che toccano il bordo
    
    std::map<int, double> componentsFeature; // mappa: indice componente -> caratteristica della componente
    std::map<int, double>::iterator cfit; // iteratore alla mappa precedente
    
    cv::Mat1f distanceFromBorder; // matrice con le distanze di ogni pixel dal bordo della componente di appartenenza
    cv::Mat1f::iterator dfbit; // iteratore alla matrice distanceFromBorder
    
    cv::Mat componentCoordinates; // coordinate dei punti di una componente
    cv::Point2f enclosingCircleCenter; // inutile centro della circonferenza che racchiude una componente
    float enclosingCircleRadius; // raggio della circonferenza che racchiude una componente
    
    cv::Mat1i flattenedBorder; // matrice con i valori del bordo della maschera
    
    /* Prendo il negativo della maschera */
    fm::maskComplement(mask, mask);
    
    /* Calcolo le componenti connesse della maschera */
    cv::connectedComponents(mask, labels, 8 /* connectivity */, CV_32S);
    
    /* Per ogni componente calcolo la massima distanza dei pixel dal bordo */
    cv::distanceTransform(mask, distanceFromBorder,
                          cv::DistanceTypes::DIST_L2, // tipo di distanza da usare
                          cv::DistanceTransformMasks::DIST_MASK_5, // grandezza della maschera da utilizzare
                          CV_32F);
    for (dfbit = distanceFromBorder.begin(), labelsIterator = labels.begin();
         labelsIterator != labels.end();
         labelsIterator++, dfbit++)
    {
        cfit = componentsFeature.find(*labelsIterator);
        if (cfit != componentsFeature.end()) // label presente
        {
            // aggiorno la massima distanza dal perimetro utilizzando il pixel corrente in distanceFromBorder
            std::get<1>(*cfit) = std::max<double>(std::get<1>(*cfit) /*cast to double*/, *dfbit);
        }
        else // label non presente: lo aggiungo
            componentsFeature.insert(std::pair<int, double>(*labelsIterator, *dfbit /*cast to double*/));
    }
    
    /* Elimino da maxPerimeterDistance le componenti che non soddisfano i requisiti,
     mentre inserisco in keepLabelsDistance quelle che li soddisfano */
    for (cfit = componentsFeature.begin(); cfit != componentsFeature.end();)
    {
        if (std::get<1>(*cfit) < maxPerimeterDistance) // non ha lo spessore minimo
            // elimino l'elemento corrente (l'iteratore ora punta al successivo elemento -> non bisogna incrementarlo)
            cfit = componentsFeature.erase(cfit);
        else
        {
            keepLabelsSorted.push_back(std::get<0>(*cfit)); // aggiungo l'elemento corrente a keepLabelsSorted
            cfit++; // incremento l'iteratore
        }
    }
    
    /* Elimino le componenti connesse che comprendono il bordo della maschera da maxPerimeterDistance,
     ma le inserisco in borderRegions per riutilizzarle alla fine */
    flattenedBorder = labels.row(0);
    cv::hconcat(flattenedBorder, labels.row(labels.rows-1), flattenedBorder);
    cv::transpose(flattenedBorder, flattenedBorder);
    cv::vconcat(flattenedBorder, labels.col(0), flattenedBorder);
    cv::vconcat(flattenedBorder, labels.col(labels.cols-1), flattenedBorder);
    std::for_each(flattenedBorder.begin(), flattenedBorder.end(), [&componentsFeature, &borderRegions] (auto& x)
                  {
                      /* Tolgo la componente corrente da keepLabels; poiché gli elementi in maxPerimeterDistance sono unici,
                       mentre in borderRegions no, devo controllare se effettivamente la funzione erase
                       ha cancellato un elemento oppure se esso era già stato eliminato: solo quando viene eliminato
                       da maxPerimeterDistance allora esso va aggiunto in borderRegions */
                      if (componentsFeature.erase(x) > 0) borderRegions.push_back(x);
                  });
    
    /* Per ogni componente rimasta calcolo la circonferenza di raggio minimo che la racchiude,
     quindi inserisco tale raggio come membro associato all'indice della componente
     nella struttura maxPerimeterDistance */
    for (cfit = componentsFeature.begin(); cfit != componentsFeature.end(); cfit++)
    {
        const int& idx = std::get<0>(*cfit);
        double& val = std::get<1>(*cfit);
        cv::findNonZero( (labels == idx), // maschera con la componente corrente
                        componentCoordinates); // coordinate dei punti della componente corrente
        cv::minEnclosingCircle(componentCoordinates,
                               enclosingCircleCenter, // variabile non utilizzata
                               enclosingCircleRadius); // raggio della circonferenza
        val = double(enclosingCircleRadius);
    }
    
    /* Copio i valori di maxPerimeterDistance in keepLabelsSorted per ordinarli */
    std::for_each(componentsFeature.begin(), componentsFeature.end(), [&keepLabelsSorted] (auto& p)
                  {
                      keepLabelsSorted.push_back(std::get<0>(p));
                  }); // copio i labels in keepLabelsSorted
    
    /* Se maxHolesNum è <0 allora non devo togliere nessun buco */
    if ((maxHolesNum >= 0)&&(maxHolesNum < (int)keepLabelsSorted.size()))
    {
        /* Ordino le componenti connesse per massimo raggio della circonferenza (in ordine discendente) */
        std::sort(keepLabelsSorted.begin(), keepLabelsSorted.end(), [&componentsFeature] (const int& a, const int& b)
                  {
                      return componentsFeature[a] > componentsFeature[b];
                  }); // riordino gli elementi
        
        /* Prendo solamente le prime maxCompNum componenti */
        while ((int)keepLabelsSorted.size() > maxHolesNum) keepLabelsSorted.pop_back();
    }
    
    /* Riaggiungo le componenti di bordo che avevo tolto inizialmente (senza copiare gli elementi) */
    keepLabelsSorted.insert(keepLabelsSorted.end(),
                            std::make_move_iterator(borderRegions.begin()),
                            std::make_move_iterator(borderRegions.end()));
    
    /* Modifico la maschera per tenere solamente le componenti in keepLabelsSorted */
    for (i = 0; i < mask.rows; i++)
    {
        for (j = 0; j < mask.cols; j++)
        {
            if (std::find(keepLabelsSorted.begin(), keepLabelsSorted.end(), labels(i,j)) != keepLabelsSorted.end()) mask(i,j) = 255;
            else mask(i,j) = 0;
        }
    }
    
    /* Prendo nuovamente il negativo della maschera per tornare all'originale */
    fm::maskComplement(mask, mask);
}

/**
 @brief Fills the holes in the mask with areas smaller than maxArea
 
 The function sorts every connected component in ascending order; then it deletes the last component until there are only maxHolesNum component left. The remaining components undergo a deleting procedure based on their area, so that the smallest are removed.
 
 @param mask mask with holes to be deleted
 @param maxHolesNum maximum number of holes to be left
 @param minArea minimum area required to a component to survive
 
 @see fm::MaskHolesFill
 */
void fm::MaskHolesFillArea(cv::InputOutputArray mask,
                           const int& maxHolesNum,
                           const int& minArea){
    /* Dichiarazioni */
    int k, N;
    cv::Mat labels, stats, centroids;
    std::multimap<int, int> keepLabelsSorted;
    
    /* Checks if mask is a cv::Mat */
    if (mask.isMat()){
        /* Regenerates the mask */
        cv::compare(mask, 0, mask, cv::CmpTypes::CMP_NE);
        
        /* Takes the mask negative */
        cv::bitwise_not(mask, mask);
        
        /* Computes the connected components */
        N = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8 /* connectivity */, CV_32S);
        /* Marks as "to-be-kept" only the components with sufficient area */
        for (k = 1/*background has not to be kept*/; k < N; k++){
            const float& area = stats.at<int>(k, cv::ConnectedComponentsTypes::CC_STAT_AREA);
            if (area > minArea) keepLabelsSorted.emplace(area, k);
        }
        
        /* If maxHolesNum is negative, no hole will be removed */
        if ((maxHolesNum >= 0)&&(maxHolesNum < (int)keepLabelsSorted.size())){
            /* Takes only the last maxCompNum components */
            while ((int)keepLabelsSorted.size() > maxHolesNum) keepLabelsSorted.erase(keepLabelsSorted.begin());
        }
        
        /* Rebuilds mask from labels */
        mask.setTo(255);
        /* For each to-be-kept component, adds its mask */
        for (const std::pair<int, int>& toBeKept: keepLabelsSorted){
            const int& label = std::get<1>(toBeKept);
            cv::bitwise_and(mask, ~(labels==label), mask);
        }
    } else {
        throw std::invalid_argument("mask is not a cv::Mat");
    }
}
