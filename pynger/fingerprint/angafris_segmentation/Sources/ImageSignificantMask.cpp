#include "../Headers/ImageSignificantMask.hpp"

void fm::ImageSignificantMask(const Image__& src,
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
                              const int& histeresisThreshold1Gmask,
                              const int& histeresisThreshold2Gmask,
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
                              const int& smooth2FinalRadius)
{
    /* Dichiarazioni */
    cv::Mat1f data, minImage; // copia della matrice di input (CV_32F) - serve per poterla modificare
    cv::Mat1b data8U; // matrice di input reinterpretata come CV_8U
    Mask__ lmask; // maschera generata utilizzando informazioni sulla variabilità locale dell'immagine
    Mask__ gmask; // maschera generata utilizzando il gradiente
    Mask__ mask0; // maschera convessa
    Mask__ gmask0; // maschera gradiente all'interno della maschera convessa
    Mask__ gmask1; // maschera gradiente all'esterno della maschera convessa
    cv::Mat1f gmaskFloat;
    cv::Mat1i labels; // etichette delle componenti connesse
    std::map<int, float> meanIntensity; // mappa (indice_componente) --> (intensità_media)
    std::map<int, float>::iterator mit; // iteratore alla struttura meanIntensity
    std::map<int, int> meanIntensityCount; // struttura con il numero di punti per componente
    cv::Mat contourPoints; // contorno della maschera
    std::vector<cv::Point> convexHullPoints; // indici dell'involucro convesso della maschera
    
    /* Copia di src in srcCopy e conversione */
    src.convertTo(data, CV_32F);
    
    /* Filtro mediano: rende più uniforme la distribuzione delle creste e delle valli */
    //    cv::medianBlur(data, data, 2*medFilterSide+1); // NIENTE KERNEL CIRCOLARE
    cv::Mat1f data2(data.rows, data.cols);
    fm::CircularMedianBlur(data2, data, medFilterSide, cv::BorderTypes::BORDER_REPLICATE);
    data2.copyTo(data);
    
    /* Filtro gaussiano */
    cv::GaussianBlur(data, data, cv::Size(2*gaussFilterSide+1, 2*gaussFilterSide+1),
                     gaussFilterSide/2.0, gaussFilterSide/2.0, cv::BorderTypes::BORDER_REPLICATE);
    
    /* Creo un'immagine che abbia l'immagine di partenza nella zona positiva della maschera
     e 1 nella zona negativa */
    fm::maskComplement(mask, lmask); // così il bordo non influisce sul filtro di minimo
    minImage.create(data.rows, data.cols);
    std::transform(lmask.begin(), lmask.end(), // input 1
                   data.begin(), // input 2
                   minImage.begin(), // output
                   [] (const auto& a, const auto& b) {return a==255? 1 : b;}); // massimo elemento per elemento tra i due input
    
    /* Filtro di minimo nella zona selezionata dalla maschera (aumenta le creste - punti di minimo locale) */
    cv::erode(minImage, minImage, // input/output
              cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(2*minFilterSide+1,2+minFilterSide+1)), // kernel
              cv::Point(-1,-1), // punto di ancoraggio centrale
              1, // numero di iterazioni
              cv::BorderTypes::BORDER_REPLICATE, 1); // tipo di estrapolazione al bordo
    
    /* Elevazione al quadrato per enfatizzare i minimi (ovvero le creste) */
    cv::pow(minImage, 2.0, minImage);
    
    /* Prendo il negativo dell'immagine per avere le creste come punti di massimo locale */
    minImage = 1-minImage;
    
    /* Riscalatura nell'intervallo [0,1] */
    fm::rescale(minImage);
    
    /* Binarizzazione tramite sogliatura */
    lmask = minImage > binLevVarMask;
    
    /* Intersezione con la maschera di partenza */
    fm::maskIntersection(lmask, mask, lmask);
    
    /* Piccola espansione a colmare i vuoti tra le creste, contrazione aggressiva per eliminare componenti spurie,
     semplificazione maschera (si tengono solamente le componenti principali e si eliminano i buchi),
     espansione per compensare la contrazione */
    fm::MaskSmoothDilate(lmask, dilate1RadiusVarMask);
    fm::MaskSmoothErode(lmask, erodeRadiusVarMask);
    fm::MaskSimplify(lmask, maxCompNumVarMask, minCompThickVarMask, maxHolesNumVarMask, minHolesThickVarMask);
    fm::MaskSmoothDilate(lmask, dilate2RadiusVarMask);
    
    /* Approccio tramite gradiente */
    /* Utilizzo l'algoritmo di Canny per l'individuazione dei contorni */
    data8U = data * 255.0;
    cv::Canny(data8U, gmask, // input/output
              histeresisThreshold1Gmask, histeresisThreshold2Gmask, // livelli di histeresis (basso/alto) (rapporto consigliato 1:2 - 1:3)
              3, // grandezza filtro di Sobel
              true); // consente di calcolare il modulo del gradiente in norma L2 anziché approssimato
    
    /* Interseco i contorni con la maschera di partenza */
    fm::maskIntersection(gmask, mask, gmask);
    
    /* Applico un filtro gaussiano all'immagine binarizzata, così nelle zone in cui sono presenti molti contorni,
     ovvero la parte centrale dell'impronta, niente viene cancellato, mentre sfumano le linee ed i punti spuri
     presenti attorno */
    gmaskFloat = gmask/255.0;
    cv::GaussianBlur(gmaskFloat, gmaskFloat, cv::Size(2*radiusGaussFilterGmask+1,2*radiusGaussFilterGmask+1),
                     radiusGaussFilterGmask/2.0, radiusGaussFilterGmask/2.0, cv::BorderTypes::BORDER_REPLICATE);
    
    /* Si esegue una media aritmetica tra minImage e gmaskFloat */
    gmaskFloat = 0.5 * gmaskFloat + 0.5 * minImage;
    
    /* Calcolo delle componenti connesse della maschera gmask, quindi selezione delle
     componenti in base alla loro intensità media */
    fm::FilterComponentByIntensity(gmask, gmaskFloat, minMeanIntensityGmask);
    /* Espansione per riempire la parte interna delle creste (per ora abbiamo solo i contorni nella maschera) */
    fm::MaskSmoothDilate(gmask, dilate1RadiusGmask);
    /* Elimino le componenti di piccolo spessore, colmo i buchi piccoli */
    fm::MaskSimplify(gmask, maxCompNumGmask, minCompThickGmask, maxHolesNumGmask, minHolesThickGmask);
    /* Contrazione per eliminare linee isolate (come le lettere segnate sull'impronta) */
    fm::MaskSmoothErode(gmask, erodeRadiusGmask);
    /* Espansione recuperare parte della maschera perduta con la contrazione */
    fm::MaskSmoothDilate(gmask, dilate2RadiusGmask);
    /* Riduzione del numero di componenti connesse (potrebbero essere state create con l'erosione) */
    fm::MaskComponentsReduce(gmask, maxCompNumGmask, minCompThickGmask);
    
    /* Unisco le maschere derivanti dall'analisi del gradiente e dell'intensità,
     quindi ne calcolo l'involucro convesso */
    fm::maskUnion(lmask, gmask, mask0);
    cv::findNonZero(mask0, contourPoints);
	if (contourPoints.total() == 0) {
		mask = Mask__::zeros(src.size());
		return;
	}
    cv::convexHull(contourPoints, convexHullPoints, false/*clockwise*/, true/*parametro ignorato*/);
    cv::drawContours(mask0, // immagine da colorare
                     std::vector<std::vector<cv::Point>>(1,convexHullPoints), // array di array di punti (una serie di contorni)
                     -1, // indice del contorno da disegnare (-1 permette di disegnarli tutti)
                     255, // valore da inserire all'interno del contorno
                     -1); // un valore negativo istruisce la funzione a riempire il contorno
    
    /* Utilizzo parametri più permissivi all'interno dell'involucro convesso,
     più restrittivi al di fuori, così da selezionare tutto ciò che può servire */
    
    /* Utilizzo l'algoritmo di Canny per estrarre i contorni, utilizzando
     dei parametri più permissivi */
    data8U = data * 255.0;
    cv::Canny(data8U, gmask0, // input/output
              histeresisThreshold3Gmask, histeresisThreshold4Gmask, // livelli di histeresis (basso/alto) (rapporto consigliato 1:2 - 1:3)
              3, // grandezza filtro di Sobel
              false); // consente di calcolare il modulo del gradiente in norma L2 anziché approssimato
    /* Espansione per riempire l'interno delle creste e contrazione leggera */
    fm::MaskSmoothDilate(gmask0, dilate3RadiusGmask);
    fm::MaskSmoothErode(gmask0, erode2RadiusGmask);
    /* Intersezione con la maschera convessa */
    fm::maskIntersection(gmask0, mask0, gmask0);
    
    /* Utilizzo l'algoritmo di Canny per estrarre i contorni, utilizzando
     dei parametri più permissivi */
    data8U = data * 255.0;
    cv::Canny(data8U, gmask1, // input/output
              histeresisThreshold5Gmask, histeresisThreshold6Gmask, // livelli di histeresis (basso/alto) (rapporto consigliato 1:2 - 1:3)
              3, // grandezza filtro di Sobel
              false); // consente di calcolare il modulo del gradiente in norma L2 anziché approssimato
    /* Intersezione con la maschera iniziale (togliamo i contorni esterni alla maschera - con la dilatazione potrebbero capitare al suo interno) */
    fm::maskIntersection(gmask1, mask, gmask1);
    /* Espansione per riempire l'interno delle creste e contrazione leggera */
    fm::MaskSmoothDilate(gmask1, dilate4RadiusGmask);
    /* Intersezione con la maschera iniziale (la dilatazione potrebbe aver espanso gmask1 oltre la maschera) */
    fm::maskIntersection(gmask1, mask, gmask1);
    /* Prendo il complementare di mask0 in gmask1, per prendere la parte di gmask1
     all'esterno dell'involucro convesso */
    fm::maskComplement(mask0, gmask1, gmask1);
    
    /* Calcolo le componenti connesse della maschera e le filtro rispetto all'intensità media */
    gmaskFloat = gmask1/255.0;
    cv::GaussianBlur(gmaskFloat, gmaskFloat, cv::Size(2*radiusGaussFilterComp+1, 2*radiusGaussFilterComp+1), radiusGaussFilterComp/2.0, radiusGaussFilterComp/2.0, cv::BorderTypes::BORDER_REPLICATE);
    fm::FilterComponentByIntensity(gmask1, gmaskFloat, meanIntensityCompThreshold);
    /* Unisco la maschera interna ed esterna all'involucro */
    fm::maskUnion(gmask0, gmask1, mask);
    /* Espansione, eliminazione componenti, eliminazione buchi, contrazione e smussamento finali */
    fm::MaskSmoothDilate(mask, dilateFinalRadius);
    fm::MaskSimplify(mask, maxCompNumFinal, minCompThickFinal, maxHolesNumFinal, minHolesThickFinal);
    fm::MaskSmoothErode(mask, erodeFinalRadius);
    fm::MaskSmoothing(mask, smoothFinalRadius);
    /* Aggiungo una cornice fissa */
    mask.rowRange(0, fixedFrameWidth-1) = 0; // banda superiore
    mask.rowRange(mask.rows-fixedFrameWidth, mask.rows-1) = 0; // banda inferiore
    mask.colRange(0, fixedFrameWidth-1) = 0; // banda sinistra
    mask.colRange(mask.cols-fixedFrameWidth, mask.cols-1) = 0; // banda destra
    /* Lisciamento finale */
    fm::MaskSmoothing(mask, smooth2FinalRadius);
    /* Semplificazione maschera finale */
    fm::MaskSimplify(mask, maxCompNumFinal, minCompThickFinal, maxHolesNumFinal, minHolesThickFinal);
}

void fm::MaskSmoothDilate(Mask__& mask, const int& r)
{
    int R = 2*r+1;
    float sigma = r > 1 ? r/2.0f : 1.0f;
    cv::Mat1f fmask, blurred;
    mask.copyTo(fmask);
    fm::rescale(fmask);
    cv::GaussianBlur(fmask, blurred, cv::Size(R,R), sigma, sigma, cv::BorderTypes::BORDER_REPLICATE);
    mask = blurred > 0.05f;
}

void fm::MaskSmoothDilate(cv::Mat1f& mask, const int& r)
{
    int R = 2*r+1;
    float sigma = r/2.0f;
    cv::GaussianBlur(mask, mask, cv::Size(R,R), sigma, sigma, cv::BorderTypes::BORDER_REPLICATE);
    std::for_each(mask.begin(), mask.end(), [](float& x){x = (x > 0.05f ? 1.0f : 0.0f);});
}

void fm::MaskSmoothErode(Mask__& mask, const int& r)
{
    int R = 2*r+1;
    float sigma = r > 1 ? r/2.0f : 1.0f;
    cv::Mat1f fmask, blurred;
    mask.copyTo(fmask);
    fm::rescale(fmask);
    cv::GaussianBlur(fmask, blurred, cv::Size(R,R), sigma, sigma, cv::BorderTypes::BORDER_REPLICATE);
    mask = blurred > 0.95f;
}

void fm::MaskSmoothErode(cv::Mat1f& mask, const int& r)
{
    int R = 2*r+1;
    float sigma = r/2.0f;
    cv::GaussianBlur(mask, mask, cv::Size(R,R), sigma, sigma, cv::BorderTypes::BORDER_REPLICATE);
    std::for_each(mask.begin(), mask.end(), [](float& x){x = (x > 0.95f ? 1.0f : 0.0f);});
}

void fm::MaskSmoothErodeDist(cv::InputOutputArray mask, const float& lev){
    cv::Mat1f distance;
    cv::distanceTransform(mask.getMat() != 0, distance, cv::DIST_L2, cv::DIST_MASK_5);
    cv::threshold(distance, mask, lev, 1.0f, cv::THRESH_BINARY);
}

void fm::MaskSmoothDilateDist(cv::InputOutputArray mask, const float& lev){
    mask.getMat() = 1.0f-mask.getMat();
    MaskSmoothErodeDist(mask, lev);
    mask.getMat() = 1.0f-mask.getMat();
}

//void fm::MaskFilterByMaxCentroidDistance(cv::InputArray inMask,
//                                         cv::OutputArray outMask,
//                                         const float& maxVal){
//    /* Declarations */
//    cv::Mat1b bMask = inMask.getMat_() != 0;
//    cv::Mat labels, stats, centroids;
//    std::map<int, float> maxCentroidDistance;
//    /* Copy the input mask to the output one */
//    inMask.copyTo(outMask);
//    /* Compute the connected components */
//    if (cv::countNonZero(bMask) > 0){
//        cv::connectedComponentsWithStats(bMask, labels, stats, centroids);
//        centroids.col(1) = (labels.rows-1)-centroids.col(1); // change y with i (flip the y-axis)
//        for (int i = 0; i < inMask.rows(); i++){ // compute the maximum distance from centroid for each component
//            for (int j = 0; j < inMask.cols(); j++){
//                const int& label = labels.at<int>(i,j);
//                if (label > 0) {
//                    const float iDist = i-centroids.at<double>(label,1),
//                    jDist = j-centroids.at<double>(label,0),
//                    dist = iDist*iDist + jDist*jDist;
//                    std::map<int, float>::iterator it = maxCentroidDistance.find(label);
//                    if (it != maxCentroidDistance.end()){
//                        /* Label found, update the maximum centroid distance for this component */
//                        if (it->second < dist){
//                            it->second = dist;
//                        }
//                    } else {
//                        /* Label not found, insert the computed distance */
//                        maxCentroidDistance[label] = dist;
//                    }
//                }
//            }
//        }
//        for (std::map<int, float>::iterator iter = maxCentroidDistance.begin(); iter != maxCentroidDistance.end(); ){
//            if (iter->second >= maxVal*maxVal){ // remove the components to be kept
//                maxCentroidDistance.erase(iter++);
//            } else {
//                ++iter;
//            }
//        }
//        for (int i = 0; i < inMask.rows(); i++){
//            for (int j = 0; j < inMask.cols(); j++){
//                if (maxCentroidDistance.find(labels.at<int>(i,j)) != maxCentroidDistance.end()){
//                    /* Label found among the ones to be removed, so set this pixel to background */
//                    outMask.getMatRef().at<float>(i,j) = 0.0f;
//                }
//            }
//        }
//    }
//}

void fm::MaskSmoothing(Mask__& mask, const int& r)
{
    int R = 2*r+1;
    float sigma = r > 1 ? r/2.0f : 1.0f;
    cv::Mat1f fmask, blurred;
    mask.copyTo(fmask);
    fm::rescale(fmask);
    cv::GaussianBlur(fmask, blurred, cv::Size(R,R), sigma, sigma, cv::BorderTypes::BORDER_REPLICATE);
    mask = blurred > 0.50f;
}

void fm::FilterComponentByIntensity(Mask__& mask, const cv::Mat1f& intensity, const double& threshold)
{
    /* Dichiarazioni */
    cv::Mat1i labels; // immagine con le etichette delle componenti connesse
    int i = 0, j = 0; // variabili di controllo per i cicli
    std::map<int, float> meanIntensity; // mappa (indice_componente) --> (intensità_media)
    std::map<int, float>::iterator mit; // iteratore alla struttura meanIntensity
    std::map<int, int> meanIntensityCount; // struttura con il numero di punti per componente
    
    /* Calcolo le componenti connesse della maschera */
    cv::connectedComponents(mask, labels, 4 /* connectivity */, CV_32S);
    for(i = 0; i < labels.rows; i++)
    {
        for (j = 0; j < labels.cols; j++)
        {
            int& idx = labels(i,j);
            const float& val = intensity(i,j);
            mit = meanIntensity.find(idx);
            if (mit == meanIntensity.end())
            {
                meanIntensity[idx] = val; // creazione ed assegnazione
                meanIntensityCount[idx] = 1;
            }
            else
            {
                std::get<1>(*mit) += val; // aggiunta del valore corrente
                meanIntensityCount[idx] += 1;
            }
        }
    }
    for (mit = meanIntensity.begin(); mit != meanIntensity.end(); mit++)
    {
        /* Divido per il numero di punti trovati nella componente corrente */
        const int& idx = std::get<0>(*mit);
        meanIntensity[idx] /= meanIntensityCount[idx];
    }
    for (mit = meanIntensity.begin(); mit != meanIntensity.end(); /*incremento condizionale*/)
    {
        /* Seleziono le componenti in base alla loro intensità media */
        float& val = std::get<1>(*mit);
        if (val < threshold) mit = meanIntensity.erase(mit);
        else mit++;
    }
    std::transform(labels.begin(), labels.end(), mask.begin(), [&meanIntensity] (const auto& x)
                   {
                       if (x == 0) return 0; // sfondo
                       if (meanIntensity.find(x) != meanIntensity.end()) return 255; // componente trovata
                       else return 0; // componente non trovata
                   });
}
