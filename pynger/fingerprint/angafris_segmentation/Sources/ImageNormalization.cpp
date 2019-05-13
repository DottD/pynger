#include "../Headers/ImageNormalization.hpp"

void fm::imageNormalize(const Image__& src,
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
                        const double& maxcp2)
{
    /* Regolarizzo l'immagine */
    imageBimodalize(src, dst, brightness, leftCut, rightCut, histSmooth, reparSmooth);
    
    /* Equalizzo l'immagine */
    ImageEqualize(dst, dst, minMaxFilter, mincp1, mincp2, maxcp1, maxcp2);
    
}

void fm::imageBimodalize(const Image__& src,
                         Image__& dst,
                         const double& brightness,
                         const double& leftCut,
                         const double& rightCut,
                         const int& histSmooth,
                         const int& reparSmooth)
{
    auto armaConvReplicate = [](arma::vec& A, const arma::vec& B){
        int L = B.n_elem/2;
        arma::vec padLeft = arma::ones<arma::vec>(L)*A[0];
        arma::vec padRight = arma::ones<arma::vec>(L)*A[A.n_elem-1];
        A.insert_rows(0, padLeft);
        A.insert_rows(A.n_elem-1, padRight);
        arma::vec C = arma::conv(A, B, "same");
        C.shed_rows(0, L-1);
        C.shed_rows(C.n_elem-L, C.n_elem-1);
        return C;
    };
    /* Copio gli elementi di src in dst; l'output sarà costruito
     da questo vettore senza copia di elementi, quindi questa operazione è necessaria */
    src.copyTo(dst);
    /* Calcolo l'istogramma di dataVector */
	arma::vec histCount = arma::conv_to<arma::vec>::from(arma::hist(arma::vec(dst.ptr<double>(), dst.total(), false, false), 256));
    /* Smussamento gaussiano dell'istogramma */
    Image__ kernel = cv::getGaussianKernel(2*histSmooth+1, double(histSmooth)/2.0, CV_64F);
    histCount = armaConvReplicate(histCount, arma::vec(kernel.ptr<double>(), kernel.total(), false, false));
    /* Calcolo del valore di threshold (valore iniziale = 128) */
	double leftMean = 0.0, rightMean = 0.0;
	int threshold = 128, oldThreshold = 0;
	arma::vec positions = arma::regspace(0, 255);
    do {
        oldThreshold = threshold; // aggiorno il threshold precedente
        /* Calcolo la media sinistra */
		leftMean = arma::sum(histCount.subvec(0, threshold-1) % positions.subvec(0, threshold-1)) / arma::sum(histCount.subvec(0, threshold-1));
        /* Calcolo la media destra */
        rightMean = arma::sum(histCount.subvec(threshold, 255) % positions.subvec(threshold, 255)) / arma::sum(histCount.subvec(threshold, 255));
        /* Aggiorno il threshold */
        threshold = int( std::round( (1-brightness)*leftMean + brightness*rightMean ) );
    } while (threshold != oldThreshold);
    /* Eliminazione delle code */
    double leftTailBreak = leftMean * leftCut;
    double rightTailBreak = rightMean * rightCut + (1.0-rightCut) * 255.0;
    /* Creo la trasformazione dei livelli di grigio (uso lo stesso vettore vec256Len) */
    for(int k = 0; k < cv::saturate_cast<int>(histCount.n_elem); ++k){
        double& x = positions[k];
        double& y = histCount[k];
        if (x <= leftTailBreak) y = 0.0;
        else if (x > leftTailBreak && x <= threshold) y = (x-leftTailBreak)/(threshold-leftTailBreak) * 127.5;
        else if (x > threshold && x <= rightTailBreak) y = (1.0+(x-threshold)/(rightTailBreak-threshold)) * 127.5;
        else y = 255.0;
    }
    /* Smussamento gaussiano del nuovo istogramma */
	kernel = cv::getGaussianKernel(2*reparSmooth+1, double(reparSmooth)/2.0, CV_64F);
    histCount = armaConvReplicate(histCount, arma::vec(kernel.ptr<double>(), kernel.total(), false, false));
    /* Ricostruisco l'immagine a partire dall'istogramma modificato */
	rescale(dst, 255);
	for(double& x: dst){
		x = histCount(int(x))/255.0;
	}
}

void fm::ImageEqualize(const Image__& src,
                       Image__& dst,
                       const int& minMaxFilter,
                       const double& mincp1,
                       const double& mincp2,
                       const double& maxcp1,
                       const double& maxcp2,
                       cv::InputArray& mask)
{
    /* Dichiarazioni */
    Image__ minImage(src.size()), maxImage(src.size()), newMinImage, newMaxImage;
    setOfPointsType_ controlPoints;
    Image__::const_iterator srcIt;
    Image__::iterator minIt, maxIt, newMinIt, newMaxIt;
    int kernelSize = 2*minMaxFilter+1;
    
    /* Per ogni elemento di src calcolo il minimo ed il massimo in un intorno
     circolare di ampiezza minMaxFilter; utilizzo rispettivamente le procedure
     di erosione e dilatazione di OpenCV. */
    cv::erode(src, // sorgente
              minImage, // destinazione
              cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize,kernelSize)), // kernel
              cv::Point2i(-1,-1), // punto di ancoraggio del kernel: (-1,-1) significa al centro
              1, // numero di iterazioni
              cv::BORDER_REPLICATE, // comportamento al bordo
              0); // valore da utilizzare in caso di bordo a valore costante
    cv::dilate(src, // sorgente
               maxImage, // destinazione
               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize,kernelSize)), // kernel
               cv::Point2i(-1,-1), // punto di ancoraggio del kernel: (-1,-1) significa al centro
               1, // numero di iterazioni
               cv::BORDER_REPLICATE, // comportamento al bordo
               0); // valore da utilizzare in caso di bordo a valore costante
    
    /* Creo due nuove matrici per rimappare massimi e minimi, così da poter ancora
     utilizzare i precedenti valori massimi e minimi */
    minImage.copyTo(newMinImage);
    maxImage.copyTo(newMaxImage);
    
    /* Rimappo i massimi con la funzione StepFunction lineare a due punti di controllo:
     - i massimi <= maxcp1 vengono portati a 0.0
     - i massimi >= maxcp2 vengono portati a 1.0
     - gli altri massimi vengono riscalati linearmente di conseguenza
     */
    controlPoints.insert(fm::pointType_(maxcp1,0.0));
    controlPoints.insert(fm::pointType_(maxcp2,1.0));
    std::for_each(newMaxImage.begin(), newMaxImage.end(), [&controlPoints] (auto& x) { x = StepFunction(x, 1, controlPoints); });
    
    /* Rimappo i massimi con la funzione StepFunction lineare a due punti di controllo:
     - i massimi <= maxcp1 vengono portati a 0.0
     - i massimi >= maxcp2 vengono portati a 1.0
     - gli altri massimi vengono riscalati linearmente di conseguenza
     */
    controlPoints.clear();
    controlPoints.insert(fm::pointType_(mincp1,0.0));
    controlPoints.insert(fm::pointType_(mincp2,1.0));
    std::for_each(newMinImage.begin(), newMinImage.end(), [&controlPoints] (auto& x) { x = StepFunction(x, 1, controlPoints); });
    
    /* Riscalo i valori dell'immagine in modo che abbiamo come minimi e massimi
     i valori appena calcolati: se massimi e minimi precedenti sono uguali
     lascio il valore iniziale */
    cv::Mat1d range = maxImage-minImage;
    std::for_each(range.begin(), range.end(), [] (auto& x) { if (x == 0.0) x = 1.0; } ); // sostituisce 1.0 agli 0.0
    range = newMinImage + (newMaxImage-newMinImage).mul((src-minImage) / range); // range è una matrice temporanea
    if (!mask.empty())
    {
        range.copyTo(dst, mask.getMat());
        src.copyTo(dst, ~mask.getMat()); // al di fuori della maschera riprendo l'immagine iniziale
    }
    else dst = range;
}
