/*
 Questo file contiene alcune procedure di matematica che non sono include in OpenCV
 */

#ifndef myMathFunc_hpp
#define myMathFunc_hpp

#include <vector>
#include <valarray>
#include <complex>
#include <iterator>
#include <opencv2/opencv.hpp>
#include "TypesTraits.hpp"
#include "ImageRescale.hpp"

//#include <fmUtilities.hpp>// Only for development

#define MASK_FALSE 0
#define MASK_TRUE 255

#define CV_PIf float(CV_PI)
const double CV_PI_2 = CV_PI / 2.0;
const double CV_PI_4 = CV_PI / 4.0;
const double CV_PI_8 = CV_PI / 8.0;
const double CV_PI_16 = CV_PI / 16.0;

#define _DEG2RAD_CONST 0.017453292519943
#define _RAD2DEG_CONST 57.295779513082323
#define deg2rad(x) ((x)*_DEG2RAD_CONST)
#define rad2deg(x) ((x)*_RAD2DEG_CONST)

// Funzione spow2 : elevamento a quadrato preservando il segno
#define fm_spow2(x) (sgn(x)*(x)*(x))
#define fm_cdot(x,y) ( (x*std::conj(y)).real() )
#define fm_cdet(x,y) ( (x*std::conj(y)).imag() )

namespace fm {	
	/* Funzioni per la creazione di maschere false o true */
	void maskComplement(const Mask__& in, Mask__& out, cv::InputArray& universe = cv::noArray()); // out = universe-in, oppure out = not in
	void maskIntersection(const Mask__& in1, const Mask__& in2, Mask__& out);
	void maskUnion(const Mask__& in1, const Mask__& in2, Mask__& out);
	
	/* Funzione segno */
	template <typename T> int sgn(T val) {
		return (T(0) < val) - (val < T(0));
	}
	
	/* Funzioni per il calcolo della fase e la normalizzazione di un numero complesso, in maniera sicura */
//	void SafeNormalize(ComplexNumber& c);
//	ComplexNumber SafeNormalizeRet(const ComplexNumber& c);
//	ComplexNumber::value_type SafeArg(const ComplexNumber& c);
	
	/*
	 convolution1D(src,dst,kernel,borderType,shape)
	 
	 Funzione che esegue la convoluzione tra il vettore src ed il kernel, restituendo il
	 risultato in dst; è possibile specificare il comportamento al bordo, ovvero
	 è possibile scegliere tra la replica dell'ultimo elemento dell'array oppure l'utilizzo
	 di elementi tutti nulli al di fuori dell'array.
	 La dimensione dell'array di output è sempre pari alla dimensione dell'array di input.
	 NON E' POSSIBILE assegnare lo stesso array in input e output.
	 */
	void convolution1D(const Image__ &src,
					   Image__ &dst,
					   const Image__ &kernel,
					   const cv::BorderTypes& borderType = cv::BorderTypes::BORDER_REPLICATE,
					   const float& const_val = 0.0f);
	void convolution1D(const cv::Mat1f &src,
					   cv::Mat1f &dst,
					   const cv::Mat1f &kernel,
					   const cv::BorderTypes& borderType = cv::BorderTypes::BORDER_REPLICATE,
					   const float& const_val = 0.0f);
	
	/*
	 genGaussianKernel(kernel,side)
	 Questa funzione genera un kernel gaussian di lato side, ovvero
	 di dimensioni (2*side+1)x(2*side+1), e deviazione standard side/2.
	 */
	void genGaussianKernel(Image__ &kernel, const int& side = 5);
	void gen2DGaussianKernel(cv::Mat1f &kernel, const int& side = 5);
	
	/*
	 CircularMedianBlur(dst, src, radius)
	 
	 Applica a src un filtro mediano con finestra circolare.
	 
	 Argomenti:
	 - dst          risultato del filtro mediano circolare
	 - src          matrice su cui applicare il filtro
	 - radius       raggio d'influenza di ciascun punto di src
	 */
	void CircularMedianBlur(cv::Mat1f& dst,
							const cv::Mat1f& src,
							const int& radius,
							const cv::BorderTypes& borderType);
	
	/*
	 GenDiskMatrix(radius)
	 
	 Genera una maschera di lato 2*radius+1, in cui hanno valore true solamente i punti che si
	 trovano all'interno di un cerchio di raggio radius centrato nel centro della maschera.
	 
	 Argomenti:
	 -radius lato della maschera e raggio del cerchio
	 
	 */
	void GenDiskMatrix(cv::Mat1d& kernel,
					   const int& radius);
	
	/*
	 GenDirGradMatrix(radius, angle, s1, e1, s2, e2)
	 
	 Genera una maschera di lato 2*radius+1; tale maschera è uno smussamento gaussiano bidimensionale controllato;
	 i parametri s1, e1, s2, e2 servono per modificare la varianza della gaussiana e la sua pendenza.
	 
	 Argomenti:
	 - radius grandezza della maschera
	 - angle direzione lungo cui eseguire il filtro gradiente
	 - s1, e1 varianza e pendenza del filtro lungo la direzione individuata da angle
	 - s2, e2 varianza e pendenza del filtro lungo la direzione perpendicolare a quella individuata da angle
	 
	 */
	void GenDirGaussMatrix(cv::Mat1d& kernel,
						   const int& radius,
						   const double& angle,
						   const double& s1,
						   const double& e1,
						   const double& s2,
						   const double& e2);
	
	/*
	 RealLineList(line, radius, angle, center, samplingStep)
	 
	 Restituisce in line la lista di coordinate reali del segmento di raggio radius,
	 centrato in center, con pendenza angle. Tutte le coordinate sono trattate in un sistema di riferimento
	 cartesiano di tipo xOy (se si inseriscono coppie di indici (riga, colonna), si avrà
	 lo stesso risultato, ma nell'utilizzo delle coordinate ottenute bisognerà considerare
	 che saranno ruotate di 90° in senso orario).
	 
	 Argomenti:
	 - line output della funzione
	 - radius raggio del segmento
	 - angle pendenza del segmento
	 - center centro del segmento (nella forma (i,j) )
	 - samplingStep numero di campioni per ogni pixel
	 */
	void RealLineList(cv::Mat2f& line,
					  const int& radius,
					  const float& angle,
					  const cv::Vec2f& center,
					  const int& samplingStep);
//	void RealLineList(cv::Mat2f& line,
//					  const int& radius,
//					  const ComplexNumber& cmplx,
//					  const cv::Vec2f& center,
//					  const int& samplingStep);
	
	/*
	 CircleList(circle, radius, center)
	 
	 Restituisce in circle la lista di coordinate intere della circonferenza di raggio radius,
	 centrata in center. Le coordinate dei punti sono approssimate all'intero più vicino.
	 Tutte le coordinate sono trattate in un sistema di riferimento
	 cartesiano di tipo xOy (se si inseriscono coppie di indici (riga, colonna), si avrà
	 lo stesso risultato, ma nell'utilizzo delle coordinate ottenute bisognerà considerare
	 che saranno ruotate di 90° in senso orario).
	 
	 Argomenti:
	 - circle output della funzione
	 - radius raggio della circonferenza
	 - center centro della circonferenza (nella forma (x,y))
	 */
	void CircleList(cv::Mat2f& circle,
					const int& radius,
					const cv::Vec2f& center);
	
	/*
	 GenDiskList(radius)
	 
	 Restituisce i punti del disco di raggio radius centrato in center.
	 Tutte le coordinate sono trattate in un sistema di riferimento
	 cartesiano di tipo xOy (se si inseriscono coppie di indici (riga, colonna), si avrà
	 lo stesso risultato, ma nell'utilizzo delle coordinate ottenute bisognerà considerare
	 che saranno ruotate di 90° in senso orario).
	 
	 Argomenti:
	 - radius       raggio del disco
	 - center       centro del disco
	 */
	void GenDiskList(cv::Mat2f& disk,
					 const int& radius,
					 const cv::Vec2f& center,
					 const bool& reorder);
	
	/*
	 GenFuzzyDiskMatrix(disk, radius, smoothRadius)
	 GenFuzzyDiskMatrix(disk, radius, smoothRadius, center, dims) **
	 
	 Genera una matrice con una maschera circolare di raggio radius, che nella corona circolare
	 tra radius-smoothRadius/2 e radius+smoothRadius/2 sfuma verso lo zero.
	 **: La seconda versione restituisce il risultato della prima, ma centrato nel punto
	 center di una matrice nulla di dimensioni dims.
	 
	 Argomenti
	 - disk             disco creato
	 - radius           raggio del disco
	 - smoothRadius     raggio dello smussamento
	 - center?          punto della matrice finale (espresso in forma (j,i))
	 - dims?            dimensioni della matrice finale (width, height)
	 */
	void GenFuzzyDiskMatrix(cv::Mat1f& disk,
							const int& radius,
							const float& smoothRadius);
	
	void GenFuzzyDiskMatrix(cv::Mat1f& disk,
							const int& radius,
							const float& smoothRadius,
							const cv::Point2i& center,
							const cv::Size2i& dims);
	
	void GenFuzzyCircleMatrix(cv::Mat1f& disk,
							  const int& radius,
							  const float& smoothRadius);
	
	void GenFuzzyCircleMatrix(cv::Mat1f& disk,
							  const int& radius,
							  const float& smoothRadius,
							  const cv::Point2i& center,
							  const cv::Size2i& dims);
	
	
	/*
	 RotField(field, ang)
	 
	 Ruota il campo (o il singolo numero complesso) passato in input di un angolo ang.
	 
	 Argomenti:
	 - field campo di input/output
	 - ang angolo di rotazione
	 */
//	void RotField(fieldType& field, const double& ang);
//	void RotField(ComplexNumber& field, const double& ang);
	
	/*
	 val = Interpolate(matrix, pos, mode)
	 
	 Restituisce il valore corrispondente alla posizione pos della matrice matrix.
	 
	 Argomenti:
	 - matrix matrice di qualunque tipo ed eventualmente a più canali
	 - pos posizione nella matrice (coppia di numeri reali)
	 - mode tipo di interpolazione
	 */
//	template<typename T = cv::Mat1f>
//	typename T::value_type Interpolate(const T& inputImage,
//									   const cv::Vec2f& pt,
//									   const interpMode& mode = interpMode::INTERP_LINEAR,
//									   const cv::BorderTypes& borderMode = cv::BorderTypes::BORDER_REPLICATE,
//									   const worldCoordinate& coordType = worldCoordinate::IJ)
//	{
//		/* Assunzioni */
//		CV_DbgAssert(!inputImage.empty());
//		CV_DbgAssert(mode == interpMode::INTERP_LINEAR);
//		
//		/* Assegno le coordinate dei vertici tra cui interpolare */
//		int i, i0, i1, j, j0, j1;
//		float di, dj, tmp;
//		if (coordType == worldCoordinate::IJ)
//		{
//			i = int(std::floor(pt(0)));
//			j = int(std::floor(pt(1)));
//			
//			i0 = cv::borderInterpolate(i, inputImage.rows, borderMode);
//			i1 = cv::borderInterpolate(i+1, inputImage.rows, borderMode);
//			j0 = cv::borderInterpolate(j, inputImage.cols, borderMode);
//			j1 = cv::borderInterpolate(j+1, inputImage.cols, borderMode);
//			
//			di = pt(0) - (float)i;
//			dj = pt(1) - (float)j;
//		} else {
//			tmp = inputImage.rows - 1 - pt(1);
//			i = int(std::floor(tmp));
//			j = int(std::floor(pt(0)));
//			
//			i0 = cv::borderInterpolate(i, inputImage.rows, borderMode);
//			i1 = cv::borderInterpolate(i+1, inputImage.rows, borderMode);
//			j0 = cv::borderInterpolate(j, inputImage.cols, borderMode);
//			j1 = cv::borderInterpolate(j+1, inputImage.cols, borderMode);
//			
//			di = tmp - (float)i;
//			dj = pt(0) - (float)j;
//		}
//		
//		const typename T::value_type& z00 = inputImage(i0,j0);
//		const typename T::value_type& z01 = inputImage(i0,j1);
//		const typename T::value_type& z10 = inputImage(i1,j0);
//		const typename T::value_type& z11 = inputImage(i1,j1);
//		
//		return (z00 * (1.0f - di) + z10 * di) * (1.0f - dj) + (z01 * (1.0f - di) + z11 * di) * dj;
//	}
	
	/*
	 StepFunction(x, degree, setOfPoints)
	 
	 Interpola i punti setOfPoints, con un polinomio di grado degree,
	 quindi restituisce l'ordinata corrispondente all'ascissa x.
	 Se x è minore della minima ascissa di setOfPoints, allora viene restituita
	 l'ordinata corrispondente alla minima ascissa,
	 se x è maggiore della massima ascissa, viene restituita l'ordinata
	 corrispondente alla massima ascissa.
	 L'insieme di punti passata in input è sempre ordinata, per costruzione.
	 */
	float LinearStepFunction(const float &x,
							 const cv::Vec3f& u,
							 const cv::Vec3f& v);
	
	double StepFunction(const double &x,
						const int &degree,
						const setOfPointsType_ &set);
}

/** Complex multiplication for fm::Field__ elements.
 This overload simply enables to use the conventional complex multiplication
 operator with fm::Field__ elements.
 @param[in] a One of the two factors.
 @param[in] b One of the two factors.
 @return The result of the operation.
 */
//fm::Field__::value_type operator*=(fm::Field__::value_type a, fm::Field__::value_type b) {
//	fm::Field__::value_type c;
//	c.re = a.re * b.re - a.im * b.im;
//	c.im = a.re * b.im + a.im * b.re;
//	return c;
//}

#endif /* myMathFunc_hpp */
