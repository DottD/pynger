#include "../Headers/myMathFunc.hpp"

//void fm::SafeNormalize(ComplexNumber& c)
//{
//	if (c.real() == 0.0f) {
//		if (c.imag() != 0.0f) {
//			// Se c ha solo parte immaginaria, la imposto ad 1
//			c.imag(1.0f);
//		}
//		// Se c è nullo rimane tale
//	} else {
//		if (c.imag() == 0.0f) {
//			// Se c ha solo parte reale, la imposto ad 1
//			c.real(1.0f);
//		} else {
//			// Se c ha entrambe le componenti non nulle, allora lo divido per il suo modulo
//			c /= std::abs(c);
//		}
//	}
//}
//fm::ComplexNumber fm::SafeNormalizeRet(const ComplexNumber& c)
//{
//	ComplexNumber r(0.0f,0.0f);
//	if (c.real() == 0.0f) {
//		if (c.imag() != 0.0f) {
//			// Se c ha solo parte immaginaria, la imposto ad 1
//			r.imag(1.0f);
//		}
//		// Se c è nullo r rimane tale
//	} else {
//		if (c.imag() == 0.0f) {
//			// Se c ha solo parte reale, la imposto ad 1
//			r.real(1.0f);
//		} else {
//			// Se c ha entrambe le componenti non nulle, allora lo divido per il suo modulo
//			r = c/std::abs(c);
//		}
//	}
//	return r;
//}
//fm::ComplexNumber::value_type fm::SafeArg(const ComplexNumber& c)
//{
//	if (c.real() == 0.0f && c.imag() == 0.0f) {
//		return 0.0f;
//	} else {
//		return std::arg(c);
//	}
//}

void fm::convolution1D(const Image__ &src,
					   Image__ &dst,
					   const Image__ &kernel,
					   const cv::BorderTypes& borderType,
					   const float& const_val)
{
	/* Assunzioni */
	assert(borderType != cv::BorderTypes::BORDER_TRANSPARENT);
	assert(!src.empty());
	assert(!kernel.empty());
	assert(kernel.total() % 2 != 0); // Controllo che kernel abbia un numero dispari di elementi
	assert(&src != &dst); // Controllo che src e dst siano due array distinti
	
	/* Dichiarazioni */
	int kernelSide, dstRef, kerRef, srcRef, srcLocalRef;
	float dstElem;
	
	/* Assegno a kernelSide la metà dei valori di kernel */
	kernelSide = (int)kernel.total() / 2;
	
	/* Libero memoria per dst e alloco lo spazio necessario a contenere il risultato della convoluzione */
	dst.create(1, (int)src.total());
	
	/* Scorro tutti gli elementi dell'array dst per assegnargli i nuovi valori in base al tipo di estrapolazione scelto
	 fuori dal bordo */
	if (borderType == cv::BorderTypes::BORDER_ISOLATED)
	{
		for (dstRef = 0; dstRef < (int)dst.total(); dstRef++)
		{
			dstElem = 0.0;
			for (kerRef = 0, srcRef = dstRef-kernelSide; kerRef < (int)kernel.total(); kerRef++, srcRef++)
			{
				if (srcRef >= 0 && srcRef < (int)src.total()) dstElem += src(srcRef) * kernel(kerRef);
			}
			dst(dstRef) = dstElem;
		}
	}
	else if (borderType == cv::BorderTypes::BORDER_CONSTANT)
	{
		for (dstRef = 0; dstRef < (int)dst.total(); dstRef++)
		{
			dstElem = 0.0;
			for (kerRef = 0, srcRef = dstRef-kernelSide; kerRef < (int)kernel.total(); kerRef++, srcRef++)
			{
				if (srcRef >= 0 && srcRef < (int)src.total()) dstElem += src(srcRef) * kernel(kerRef);
				else dstElem += const_val * kernel(kerRef);
			}
			dst(dstRef) = dstElem;
		}
	}
	else
	{
		for (dstRef = 0; dstRef < (int)dst.total(); dstRef++) {
			dstElem = 0.0;
			for (kerRef = 0, srcRef = dstRef-kernelSide; kerRef < (int)kernel.total(); kerRef++, srcRef++) {
				srcLocalRef = cv::borderInterpolate(srcRef, (int)src.total(), borderType);
				dstElem += src(srcLocalRef) * kernel(kerRef);
			}
			dst(dstRef) = dstElem;
		}
	}
}
void fm::convolution1D(const cv::Mat1f &src,
					   cv::Mat1f &dst,
					   const cv::Mat1f &kernel,
					   const cv::BorderTypes& borderType,
					   const float& const_val)
{
	/* Assunzioni */
	assert(borderType != cv::BorderTypes::BORDER_TRANSPARENT);
	assert(!src.empty());
	assert(!kernel.empty());
	assert(kernel.total() % 2 != 0); // Controllo che kernel abbia un numero dispari di elementi
	assert(std::abs(cv::sum(kernel)[0]-1.0f) < 1E-5); // controllo che kernel abbia somma unitaria
	assert(&src != &dst); // Controllo che src e dst siano due array distinti
	
	/* Dichiarazioni */
	int kernelSide, dstRef, kerRef, srcRef, srcLocalRef;
	float dstElem, localKernelSum;
	
	/* Assegno a kernelSide la metà dei valori di kernel */
	kernelSide = (int)kernel.total() / 2;
	
	/* Libero memoria per dst e alloco lo spazio necessario a contenere il risultato della convoluzione */
	dst.create(1, (int)src.total());
	
	/* Scorro tutti gli elementi dell'array dst per assegnargli i nuovi valori in base al tipo di estrapolazione scelto
	 fuori dal bordo */
	if (borderType == cv::BorderTypes::BORDER_ISOLATED)
	{
		for (dstRef = 0; dstRef < (int)dst.total(); dstRef++)
		{
			dstElem = 0.0f;
			localKernelSum = 0.0f;
			for (kerRef = 0, srcRef = dstRef-kernelSide; kerRef < (int)kernel.total(); kerRef++, srcRef++)
			{
				if (srcRef >= 0.0f && srcRef < (int)src.total())
				{
					const float& kerVal = kernel(kerRef); // forse si evita l'accesso consecutivo alla stessa memoria
					dstElem += src(srcRef) * kerVal;
					localKernelSum += kerVal;
				}
			}
			dst(dstRef) = dstElem / localKernelSum;
		}
	}
	else if (borderType == cv::BorderTypes::BORDER_CONSTANT)
	{
		for (dstRef = 0; dstRef < (int)dst.total(); dstRef++)
		{
			dstElem = 0.0;
			for (kerRef = 0, srcRef = dstRef-kernelSide; kerRef < (int)kernel.total(); kerRef++, srcRef++)
			{
				if (srcRef >= 0 && srcRef < (int)src.total()) dstElem += src(srcRef) * kernel(kerRef);
				else dstElem += const_val * kernel(kerRef);
			}
			dst(dstRef) = dstElem;
		}
	}
	else
	{
		for (dstRef = 0; dstRef < (int)dst.total(); dstRef++) {
			dstElem = 0.0;
			for (kerRef = 0, srcRef = dstRef-kernelSide; kerRef < (int)kernel.total(); kerRef++, srcRef++) {
				srcLocalRef = cv::borderInterpolate(srcRef, (int)src.total(), borderType);
				dstElem += src(srcLocalRef) * kernel(kerRef);
			}
			dst(dstRef) = dstElem;
		}
	}
}

void fm::GenDirGaussMatrix(cv::Mat1d& kernel,
						   const int& radius,
						   const double& ang,
						   const double& s1,
						   const double& e1,
						   const double& s2,
						   const double& e2)
{
	/* Definisco la funzione che crea un filtro gradiente direzionale lungo l'asse x */
	auto f = [&s1, &e1, &s2, &e2] (const double& t1, const double& t2) {
		/* Derivata della gaussiana lungo l'asse x e gaussiana lungo l'asse y */
		return exp( -pow(fabs(t1/s1), 2.0*e1) - pow(fabs(t2/s2), 2.0*e2) );
	};
	
	/* Alloco la matrice di output */
	kernel.create(2*radius+1, 2*radius+1);
	
	/* Ruoto la matrice di -angle, quindi applico la funzione f */
	for (int i = -radius; i <= radius; i++) {
		for (int j = -radius; j <= radius; j++) {
			kernel(i+radius,j+radius) = f(cos(ang)*i + sin(ang)*j, -sin(ang)*i + cos(ang)*j);
		}
	}
	
	/* Al kernel sottraggo la matrice che si ottiene ruotandolo di 180°, al fine di ottenere summa nulla */
	cv::Mat1d tempKernel;
	cv::flip(kernel, tempKernel, -1); // flip su entrambi gli assi
	kernel += tempKernel;
	
	/* Moltiplico la maschera ottenuta per una maschera circolare della stessa dimensione */
	GenDiskMatrix(tempKernel, radius);
	kernel = kernel.mul(tempKernel);
	
	/* Faccio in modo che il kernel abbia somma dei valori assoluti unitaria */
	kernel /= cv::sum(abs(kernel))[0];
}

void fm::GenDiskMatrix(cv::Mat1d& kernel,
					   const int& radius) {
	kernel.create(2*radius+1, 2*radius+1);
	double r2 = (double)(radius*radius);
	for (int i = -radius; i <= radius; i++) {
		for(int j = -radius; j <= radius; j++) {
			kernel(i+radius,j+radius) = (i*i+j*j) <= r2 ? 1.0 : 0.0;
		}
	}
}

void fm::RealLineList(cv::Mat2f& line,
					  const int& radius,
					  const float& angle,
					  const cv::Vec2f& center,
					  const int& samplingStep)
{
	/* Dichiarazioni */
	float step, r;
	
	/* Calcolo il passo tra un campione ed il successivo sul segmento */
	step = 1.0f/float(samplingStep);
	/* Genero la lista di campioni */
	line.release();
	for (r = -float(radius); r <= float(radius); r += step) line.push_back(center + r * cv::Vec2f(cosf(angle), sinf(angle)));
}
//void fm::RealLineList(cv::Mat2f& line,
//					  const int& radius,
//					  const ComplexNumber& cmplx,
//					  const cv::Vec2f& center,
//					  const int& samplingStep)
//{
//	/* Dichiarazioni */
//	float step, r;
//	
//	/* Calcolo il passo tra un campione ed il successivo sul segmento */
//	step = 1.0/float(samplingStep);
//	/* Genero la lista di campioni */
//	line.release();
//	for (r = -radius; r <= radius; r += step) line.push_back(center + r * cv::Vec2f(cmplx.real(), cmplx.imag()));
//}

void fm::CircleList(cv::Mat2f& circle,
					const int& radius,
					const cv::Vec2f& center)
{
	/* Dichiarazioni */
	float t, min, max, step;
	float r2; // radius al quadrato
	float c; // coordinata
	
	/* Calcolo i limiti delle variabili X e Y */
	min = 0.0f; // l'eventuale caso 0.0f lo tratto a parte
    max = floorf(float(radius)/float(sqrt(2.0))); // non prendo tutta la semicirconferenza
	step = 1.0f;
	/* Calcolo alcune quantità per velocizzare i calcoli successivi */
	r2 = radius*radius;
	/* Genero la lista di campioni (non ci sono ripetizioni per costruzione) */
	circle.release();
	circle.reserve(size_t((max-min)/step));
	/* Eventuale caso t = 0.0f */
	if (min == 0.0f) { // In questo caso si creano ripetizioni ai punti cardinali
		c = roundf(std::sqrt(r2-min*min));
		circle.push_back( cv::Vec2f(c, min) );
		circle.push_back( cv::Vec2f(-c, min) );
		circle.push_back( cv::Vec2f(min, c) );
		circle.push_back( cv::Vec2f(min, -c) );
		min += step; // così non è incluso nel successivo ciclo
	}
	for (t = min; t <= max; t += step) {
		c = roundf(std::sqrt(r2-t*t));
		circle.push_back( cv::Vec2f(c, t) );
		circle.push_back( cv::Vec2f(-c, t) );
		circle.push_back( cv::Vec2f(c, -t) );
		circle.push_back( cv::Vec2f(-c, -t) );
		circle.push_back( cv::Vec2f(t, c) );
		circle.push_back( cv::Vec2f(t, -c) );
		circle.push_back( cv::Vec2f(-t, -c) );
		circle.push_back( cv::Vec2f(-t, c) );
	}
	
	/* Riordino i punti della circonferenza */
	std::sort(circle.begin(), circle.end(), [](const cv::Vec2f& a, const cv::Vec2f& b){
		/* Riordino i punti della circonferenza in senso antiorario partendo dal punto (radius, 0). */
		if (a(1) >= 0 && b(1) < 0) return true; // a è nella semicirc. superiore, b in quella inferiore
		else if (a(1) < 0 && b(1) >= 0) return false; // b è nella semicirc. superiore e a in quella inferiore
		else if (a(1) >= 0 && b(1) >= 0) /* entrambi nella semicirc. superiore -> ordino per x decrescente */ {
			if (a(0) > b(0)) return true;
			else if (a(0) < b(0)) return false;
			else /* stessa x, ordino in base alla y */ {
				if (a(0) >= 0) /* I quadrante - y crescente */ return a(1) < b(1);
				else /* II quadrante - y decrescente */ return a(1) > b(1);
			}
		}
		else /* entrambi nella semicirc. inferiore -> ordino per x crescente */ {
			if (a(0) < b(0)) return true;
			else if (a(0) > b(0)) return false;
			else /* stessa x, ordino in base alla y */ {
				if (a(0) >= 0) /* IV quadrante - y crescente */ return a(1) < b(1);
				else /* III quadrante - y decrescente */ return a(1) > b(1);
			}
		}
	});
	
	/* Traslo il loop in center */
	// circle += center;
	std::transform(circle.begin(), circle.end(), circle.begin(), [&center](const cv::Vec2f& a){ return a + center; });
}

void fm::GenDiskList(cv::Mat2f& disk,
					 const int& radius,
					 const cv::Vec2f& center,
					 const bool& reorder)
{
	/* Dichiarazioni */
	float r2 = radius*radius;
	int x, y;
	
	/* Inserisco in disk tutti i punti del disco centrato nell'origine */
	for (x = -radius; x <= radius; x++)
	{
		for(y = -radius; y <= radius; y++)
		{
			if (x*x+y*y <= r2) {
				disk.push_back(cv::Vec2f(x,y));
			}
		}
	}
	
	/* Se rischiesto riordino i punti del disco per distanza crescente dal centro */
	if (reorder) {
		std::sort(disk.begin(), disk.end(), [](const cv::Vec2f& a, const cv::Vec2f& b){
			return (a(0)*a(0)+a(1)*a(1)) < (b(0)*b(0)+b(1)*b(1));
		});
	}
	
	/* Centro il disco in center */
	std::transform(disk.begin(), disk.end(), disk.begin(), [&center](const cv::Vec2f& a){ return a + center; });
	// disk += center;
}

void fm::GenFuzzyDiskMatrix(cv::Mat1f& disk,
							const int& radius,
							const float& smoothRadius)
{
	/* Dichiarazioni */
	int size, i, j;
	setOfPointsType_ points;
	
	// Calcolo la misura finale della maschera
	size = int(std::ceil(float(radius) + smoothRadius/2.0f));
	// Definisco i punti d'interpolazione per la funzione di smussamento
	points.insert(pointType_(0, 0));
	points.insert(pointType_(smoothRadius, 1));
	
	// Genero la maschera circolare
	disk.create(2*size+1, 2*size+1);
	for (i = -size; i <= size; i++) {
		for(j = -size; j <= size; j++) {
			disk(i+size,j+size) = 1.0f - float(StepFunction(std::sqrt(i*i+j*j) - radius + smoothRadius/2, 5, points));
		}
	}
}

void fm::GenFuzzyDiskMatrix(cv::Mat1f& finalDisk,
							const int& radius,
							const float& smoothRadius,
							const cv::Point2i& center,
							const cv::Size2i& dims)
{
	/* Dichiarazioni */
	cv::Mat1f smallDisk;
	cv::Rect2i smallROI, finalROI;
	cv::Point2i smallCenter, shift;
	
	/* Genero la maschera circolare */
	GenFuzzyDiskMatrix(smallDisk, radius, smoothRadius);
	
	/* Definisco le ROI per sovrapporre le due matrici */
	smallCenter.x = (smallDisk.rows-1)/2;
	smallCenter.y = smallCenter.x;
	shift = center - smallCenter;
	smallROI = cv::Rect2f(0, 0, smallDisk.cols, smallDisk.rows);
	finalROI = cv::Rect2f(0, 0, dims.width, dims.height);
	smallROI &= finalROI - shift;
	finalROI &= smallROI + shift;
	
	/* Creo la maschera finale */
	finalDisk = cv::Mat1f::zeros(dims);
	smallDisk(smallROI).copyTo(finalDisk(finalROI));
}

void fm::GenFuzzyCircleMatrix(cv::Mat1f& disk,
							  const int& radius,
							  const float& smoothRadius)
{
	/* Dichiarazioni */
	int size, i, j;
	setOfPointsType_ points;
	
	// Calcolo la misura finale della maschera
	size = int(std::ceil(float(radius) + smoothRadius/2.0f));
	// Definisco i punti d'interpolazione per la funzione di smussamento
	points.insert(pointType_(0, 0));
	points.insert(pointType_(smoothRadius, 1));
	
	// Genero la maschera circolare
	disk.create(2*size+1, 2*size+1);
	for (i = -size; i <= size; i++) {
		for(j = -size; j <= size; j++) {
			disk(i+size,j+size) = 1.0f - float(StepFunction(std::abs(std::sqrt(i*i+j*j) - radius + smoothRadius/2), 5, points));
		}
	}
}

void fm::GenFuzzyCircleMatrix(cv::Mat1f& finalDisk,
							  const int& radius,
							  const float& smoothRadius,
							  const cv::Point2i& center,
							  const cv::Size2i& dims)
{
	
	/* Dichiarazioni */
	cv::Mat1f smallDisk;
	cv::Rect2i smallROI, finalROI;
	cv::Point2i smallCenter, shift;
	
	/* Genero la maschera circolare */
	GenFuzzyCircleMatrix(smallDisk, radius, smoothRadius);
	
	/* Definisco le ROI per sovrapporre le due matrici */
	smallCenter.x = (smallDisk.rows-1)/2;
	smallCenter.y = smallCenter.x;
	shift = center - smallCenter;
	smallROI = cv::Rect2f(0, 0, smallDisk.cols, smallDisk.rows);
	finalROI = cv::Rect2f(0, 0, dims.width, dims.height);
	smallROI &= finalROI - shift;
	finalROI &= smallROI + shift;
	
	/* Creo la maschera finale */
	finalDisk = cv::Mat1f::zeros(dims);
	smallDisk(smallROI).copyTo(finalDisk(finalROI));
}

//void fm::RotField(fieldType& field,
//				  const double& ang)
//{
//	ComplexNumber complxAng = exp(ComplexNumber(0,ang));
//	std::for_each(field.begin(), field.end(), [&complxAng] (auto& x) {x *= complxAng;});
//}
//void fm::RotField(ComplexNumber& field, const double& ang)
//{
//	ComplexNumber complxAng = exp(ComplexNumber(0,ang));
//	field *= complxAng;
//}

void fm::genGaussianKernel(Image__ &kernel, const int& side)
{
	kernel = cv::getGaussianKernel(2*side+1, side/2.0, CV_64F);
}
void fm::gen2DGaussianKernel(cv::Mat1f &kernel, const int& side)
{
	cv::Mat1f gaussLine = cv::getGaussianKernel(2*side+1, side/2.0, CV_32F);
	kernel = gaussLine * gaussLine.t();
}

void fm::CircularMedianBlur(cv::Mat1f& dst,
							const cv::Mat1f& src,
							const int& radius,
							const cv::BorderTypes& borderType)
{
	/* Dichiarazioni */
	cv::Vec2f P0; // variabile di controllo
	cv::Mat2f Disk0, Disk; // coordinate dei punti del disco
	cv::Mat1f Values; // valori dell'immagini sui punti del disco
	
	/* Le immagini sorgente e destinazione devono essere distinte */
	CV_DbgAssert(&dst != &src);
	
	/* Eventualmente alloco la matrice dst */
	dst.create(src.rows, src.cols);
	
	/* Genero la lista dei punti del disco di raggio radius */
	GenDiskList(Disk0, radius, cv::Vec2f(0,0), false);
	
	/* Scorro l'immagine dst e ne modifico il contenuto */
	float& i = P0(0);
	float& j = P0(1);
	for (i = 0; i < dst.rows; i++) {
		for (j = 0; j < dst.cols; j++) {
			/* Traslazione del disco sul punto corrente */
			// Disk = Disk0 + P0;
			std::transform(Disk0.begin(), Disk0.end(), std::back_inserter(Disk), [&P0](const cv::Vec2f& a){ return a + P0; });
			/* Prendo la lista dei valori dell'immagine src sopra a tali punti */
			Values.release();
			for (cv::Vec2f& P: Disk) {
				P(0) = cv::borderInterpolate((int)P(0), src.rows, borderType);
				P(1) = cv::borderInterpolate((int)P(1), src.cols, borderType);
				Values.push_back(src((int)P(0), (int)P(1)));
			}
			/* Calcolo la mediana dei valori e la sostituisco al valore corrente in dst */
			std::sort(Values.begin(), Values.end());
			dst((int)i, (int)j) = Values( int(std::floor( float( int(Values.total())/2 ) ) ) );
			// Clear Disk
			Disk.release();
		}
	}
}

void fm::maskComplement(const Mask__& in, Mask__& out, cv::InputArray& universe)
{
	if (!universe.empty()) fm::maskIntersection(~in, universe.getMat(), out);
	else out = ~in;
}
void fm::maskIntersection(const Mask__& in1, const Mask__& in2, Mask__& out)
{
	out = in1 & in2;
}
void fm::maskUnion(const Mask__& in1, const Mask__& in2, Mask__& out)
{
	out = in1 | in2;
}

float fm::LinearStepFunction(const float &x,
							 const cv::Vec3f& u,
							 const cv::Vec3f& v)
{
	if (x <= u(0)) return v(0);
	else if (x >= u(2)) return v(2);
	else if (x <= u(1)) return v(0) + (v(1)-v(0)) * (x-u(0)) / (u(1)-u(0));
	else return v(1) + (v(2)-v(1)) * (x-u(1)) / (u(2)-u(1));
}

double fm::StepFunction(const double &x, const int &degree, const setOfPointsType_ &set)
{
	if (!(
		  (degree == 1)||
		  ( (degree == 3)&&(set.size() == 2) )||
		  ( (degree == 5)&&(set.size() == 2) )
		  ))
		throw std::invalid_argument("StepFunction: parametri non previsti");
	/* Recupero gli iteratori al primo ed ultimo elemento */
	setOfPointsType_::const_iterator begin = set.begin();
	setOfPointsType_::const_iterator end = set.end();  end--;
	const double& a = std::get<0>(*begin);
	const double& b = std::get<0>(*end);
	
	if (x <= a) // se x è minore del primo elemento di set
		return std::get<1>(*begin); // restituisco l'ordinata di quest'ultimo
	else if (x >= b) // se x è maggiore dell'ultimo elemento di set
		return std::get<1>(*end); // restituisco l'ordinata di quest'ultimo
	else {
		switch (degree) {
			case 1: {
				setOfPointsType_::const_iterator it = set.upper_bound(x); // iteratore all'elemento dopo x
				const double& x2 = std::get<0>(*it);
				const double& y2 = std::get<1>(*it);
				it--;
				const double& x1 = std::get<0>(*it);
				const double& y1 = std::get<1>(*it);
				return y1 + (y2-y1) * (x-x1) / (x2-x1);
			}
			case 3:
				return (3*b-a-2*x) * std::pow(a-x,2.0) / std::pow(b-a,3.0);
			case 5:
				return -std::pow(a-x, 3.0) * (10*b*b + a*a + 3*a*x + 6*x*x - 5*b*(a+3*x)) / std::pow(b-a, 5.0);
			default:
				return 0.0;
		}
	}
}
