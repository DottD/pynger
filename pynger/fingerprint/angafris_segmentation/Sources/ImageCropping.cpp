#include "../Headers/ImageCropping.hpp"

void fm::TopMask(const Image__& src,
                 Mask__& mask,
                 const double &scanAreaAmount,
                 const double &gradientFilterWidth,
                 const double &binarizationLevel,
                 const double &f,
                 const double &slopeAngle,
                 const double &lev,
                 const int &marg)
{
    /* Dichiarazioni */
    const int& d1 = src.rows, d2 = src.cols; // Uso degli alias per leggere le dimensioni dell'immagine di partenza
    double l, a = 0.0, b = 0.0, tg = 0.0/*a quanto deve essere inizializzato?*/, lineSum;
    int scanAreaInPixel, k, s, r, diff;
    Image__ horFilter, gradImage, imgOnMask;
    cv::Vec<double, 5> vertFilter;
    cv::Mat1d scanArea;
    Image__ scanAreaPartSum;
	std::vector<int> searchPositions;
    std::vector<double> lineCenters, lineDiff;
    std::vector<double>::iterator lcit, ldit;
    cv::Mat maskNonZeroPoints;
    cv::Rect boundingBox;
    Mask__ innerMask;
    
    /* Mi limito a considerare solamente i punti contenuti nel minimo bounding box
     dei punti in cui la maschera è positiva */
    cv::findNonZero(mask, maskNonZeroPoints);
    boundingBox = cv::boundingRect(maskNonZeroPoints);
    imgOnMask = src(boundingBox);
    
    /* Applico all'immagine un filtro verticale, per evidenziare le variazioni di grigio
     lungo le linee verticali, ed un filtro mediano orizzontale; quindi ne calcolo
     il valore assoluto */
    vertFilter << 1.0, 1.0, 0.0, -1.0, -1.0;
    r = int(std::floor(gradientFilterWidth * d2/2.0)); // Calcolo l'ampiezza del filtro gradiente in funzione della larghezza dell'immagine
    horFilter = Image__::ones(1, 2*r+1);
    cv::sepFilter2D(imgOnMask, gradImage, // input/output
                    -1, // il dato rimane dello stesso tipo
                    horFilter, vertFilter, // filtri lungo le righe e lungo le colonne rispettivamente
                    cv::Point(-2,-r), // punto di ancoraggio del filtro (centro)
                    0.0, // valore da aggiungere all'immagine finale
                    cv::BorderTypes::BORDER_REPLICATE); // comportamento al bordo (replica dei valori al bordo)
    gradImage = abs(gradImage);
    
    /* Riscalo l'immagine tra 0 ed 1 */
    fm::rescale(gradImage);
    
    /* Binarizzo l'immagine utilizzando la soglia passata in input */
    gradImage.forEach([&binarizationLevel] (double& x, const int*) { x = (x >= binarizationLevel) ? 1.0 : 0.0; });
    
    /* A partire dall'inclinazione massima che può avere il bordo che vogliamo riconoscere,
     calcoliamo il massimo discostamento dal caso orizzontale */
    diff = (int)std::ceil(std::tan(slopeAngle*CV_PI/180.0) * d2 / 2.0);
    
    /* Mi riduco a considerare solamente una porzione dell'immagine per la ricerca del bordo */
    scanAreaInPixel = int(std::ceil(scanAreaAmount * d1)) + diff; // porzione ricerca bordo + scostamento massimo dall'orizzontale
    scanArea = gradImage.rowRange(0, scanAreaInPixel);
    
    /* Sommo gli elementi lungo le righe di quest'area */
    cv::reduce(scanArea, scanArea,
               1, // la matrice viene ridotta ad una singola colonna (= riduzione lungo le righe)
               cv::ReduceTypes::REDUCE_SUM);
    
    /* Partiziono scanArea in sottoinsiemi connessi di diff elementi, quindi calcolo
     la somma su ogni sottoinsieme ed inserisco i risultati in un nuovo array */
    scanAreaPartSum = Image__::zeros((int)scanArea.total()-diff+1, 1);
    for (k = 0; k <= (int)scanArea.total()-diff; k++)
    {
        for (s = 0; s < diff; s++) scanAreaPartSum(k) += scanArea(k+s);
    }
    
    /* Calcolo il numero di punti di una linea necessari affinché essa sia individuata */
    l = lev * d2;
    
    /* Si individuano le porzioni d'immagine che hanno un numero sufficiente di punti a gradiente 1,
     quindi si conservano le loro posizioni per la ricerca successiva */
    for(k = 0; k < (int)scanAreaPartSum.total(); k++)
    {
        if (scanAreaPartSum(k) > f * l) searchPositions.push_back(k);
    }
    
    /* Se non ci sono porzioni di immagine adeguate, allora esco dalla funzione senza eseguire
     alcuno scontornamento */
    if (searchPositions.empty())
    {
        mask = Mask__::ones(d1, d2) * 255;
        mask.row(0) = 0;
        mask.row(d1-1) = 0;
        mask.col(0) = 0;
        mask.col(d2-1) = 0;
        return;
    }
    
    /* Creo la lista dei candidati ad essere centri di linea */
    for (k = searchPositions.back(); k >= 0; k--) lineCenters.push_back(k);
    
    /* Creo la lista dei possibili scostamenti degli estremi rispetto alla posizione del centro di linea */
    for (k = -diff; k <= diff; k++) lineDiff.push_back(k);
    std::sort(lineDiff.begin(), lineDiff.end(), [] (const double& x, const double& y) {return (std::abs(x) < std::abs(y)) ? true : false;} );
    
    /* Scorro ogni possibile combinazione di centri di linea e scostamenti
     (ogni elemento del prodotto cartesiano lineCenters x lineDiff) */
    for (lcit = lineCenters.begin(), lineSum = 0.0; // lineSum va inizializzato da qualche parte (ad esempio qui)
         lcit != lineCenters.end() && lineSum < l;
         lcit++)
    {
        for (ldit = lineDiff.begin();
             ldit != lineDiff.end() && lineSum < l;
             ldit++)
        {
            lineSum = 0.0;
            a = (*lcit)+(*ldit);
            b = (*lcit)-(*ldit);
            tg = (a-b) / d2;
            for (k = 0; k < d2; k++) {
                s = (int)std::ceil(b+k*tg); // coordinata di riga del punto corrente
                lineSum += ((s < 0)||(s >= d1)) ? 0.0 : gradImage(s,k); // se il punto è fuori immagine non aggiungo nulla
            }
        }
    }
    
    /* Creo la maschera da restituire in output */
    innerMask = mask(boundingBox);
    for (k = 0; k < innerMask.rows; k++)
    {
        for (s = 0; s < innerMask.cols; s++)
        {
            if (k <= std::ceil(b+s*tg)+marg) innerMask(k,s) = MASK_FALSE;
        }
    }
}

void fm::ImageCroppingLines(const Image__& src,
                            Mask__& mask,
                            const double &scanAreaAmount,
                            const double &gradientFilterWidth,
                            const int& gaussianFilterSide,
                            const double &binarizationLevel,
                            const double &f,
                            const double &slopeAngle,
                            const double &lev,
                            const int &marg)
{
    /* Dichiarazioni */
    Image__ srcCopy, workingImage;
    Mask__ tempMask;
    
    /* Faccio una copia di src in srcCopy, per permettere la modifica dell'immagine;
     se src e dst sono la stessa matrice questa operazione non viene eseguita */
    src.copyTo(srcCopy);
    tempMask.create(src.rows, src.cols);
    
    /* Applico all'immagine un filtro gaussiano */
    cv::GaussianBlur(srcCopy, srcCopy, // immagini di input ed output
                     cv::Size(2*gaussianFilterSide+1, 2*gaussianFilterSide+1), // dimensioni del filtro
                     gaussianFilterSide/2.0); // sigma del filtro gaussiano
    
    /* Ricerco linee da eliminare nella parte superiore */
    TopMask(srcCopy, mask, scanAreaAmount, gradientFilterWidth, binarizationLevel, f, slopeAngle, lev, marg);
    
    /* Ricerco linee da eliminare nella parte inferiore */
    cv::flip(srcCopy, workingImage, 0); // flip attorno all'asse x
    tempMask = 255; // inizializzo ed azzero la maschera
    TopMask(workingImage, tempMask, scanAreaAmount, gradientFilterWidth, binarizationLevel, f, slopeAngle, lev, marg);
    cv::flip(tempMask, tempMask, 0); // flip inverso della maschera per poterla sovrapporre all'immagine originale
    fm::maskIntersection(mask, tempMask, mask); // interseco la maschera precedente con questa
    
    /* Ricerco linee da eliminare nella parte sinistra */
    cv::transpose(srcCopy, workingImage); // trasposizione della matrice
    tempMask = 255; // azzero la maschera
    cv::transpose(tempMask, tempMask);
    TopMask(workingImage, tempMask, scanAreaAmount, gradientFilterWidth, binarizationLevel, f, slopeAngle, lev, marg);
    cv::transpose(tempMask, tempMask); // inverto la trasposizione
    fm::maskIntersection(mask, tempMask, mask); // interseco la maschera precedente con questa
    
    /* Ricerco linee da eliminare nella parte destra */
    cv::transpose(srcCopy, workingImage); // trasposizione della matrice
    cv::flip(workingImage, workingImage, 0); // flip attorno all'asse x per avere la parte destra dell'immagine originale sopra
    tempMask = 255; // azzero la maschera
    cv::transpose(tempMask, tempMask);
    TopMask(workingImage, tempMask, scanAreaAmount, gradientFilterWidth, binarizationLevel, f, slopeAngle, lev, marg);
    cv::flip(tempMask, tempMask, 0); // inverto il flipping
    cv::transpose(tempMask, tempMask); // inverto la trasposizione
    fm::maskIntersection(mask, tempMask, mask); // interseco la maschera precedente con questa
}

void fm::ImageCroppingSimple(const Image__& src,
                             Mask__& mask,
                             const double& minVariation,
                             const int& marg)
{
    /* Dichiarazioni */
    int t,b,l,r; // indici di linea che indicheranno i nuovi margini
    double localMin, localMax;
    
    /* Scorro l'immagine da sopra */
    for (t = 0; t < src.rows;)
    {
        cv::minMaxLoc(src.row(t), &localMin, &localMax);
        if (localMax-localMin < minVariation) t++;
        else break;
    }
    
    /* Scorro l'immagine da sotto */
    for (b = src.rows-1; b >= 0;)
    {
        cv::minMaxLoc(src.row(b), &localMin, &localMax);
        if (localMax-localMin < minVariation) b--;
        else break;
    }
    
    /* Scorro l'immagine da sinistra */
    for (l = 0; l < src.cols;)
    {
        cv::minMaxLoc(src.col(l), &localMin, &localMax);
        if (localMax-localMin < minVariation) l++;
        else break;
    }
    
    /* Scorro l'immagine da destra */
    for (r = src.cols-1; r >= 0;)
    {
        cv::minMaxLoc(src.col(r), &localMin, &localMax);
        if (localMax-localMin < minVariation) r--;
        else break;
    }
    
    /* Aggiungo il margine passato in input */
    l += marg;
    t += marg;
    b -= marg;
    r -= marg;
    
    /* Metto false nella cornice appena trovata */
    mask.colRange(0, l) = MASK_FALSE;
    mask.rowRange(0, t) = MASK_FALSE;
    mask.colRange(r+1, mask.cols) = MASK_FALSE;
    mask.rowRange(b+1, mask.rows) = MASK_FALSE;
}
