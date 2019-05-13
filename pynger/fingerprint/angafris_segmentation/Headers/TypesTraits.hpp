#ifndef TypesTraits_h
#define TypesTraits_h

#include <complex>
#include <functional>
#include <list>
#include <map>
#include <exception>
#include <opencv2/opencv.hpp>

namespace fm {
	typedef std::runtime_error custom_exception;
    /* Tipi di dato condivisi da più funzioni */
    typedef cv::Mat1d Image__;
	typedef cv::Mat1b Mask__;
	const unsigned char MaskTRUE = 0xf;
	const unsigned char MaskFALSE = 0x0;
//	typedef cv::Mat_<cv::Complexd> Field__;
//    typedef std::complex<float> ComplexNumber;
//    typedef cv::Mat_<ComplexNumber> fieldType;
	
    // Assign value to nullComplex
//    const ComplexNumber nullComplex = ComplexNumber(0,0);
	
//    typedef enum{
//        INTERP_NN,
//        INTERP_LINEAR,
//        INTERP_CUBIC,
//        INTERP_NONE
//    } interpMode;
//    
//    typedef enum{
//        XY,
//        IJ
//    } worldCoordinate;
	
    typedef std::map<double,double> setOfPointsType_;
    typedef setOfPointsType_::value_type pointType_;
    
//    typedef enum {
//        NONE,
//        NORMAL,
//        TANGENT
//    } weightType_;
	
//    struct _minutia{
//        enum struct type{
//            isolated,
//            termination,
//            ridge,
//            bifurcation,
//            complex
//        } flag;
//        int i, j;
//        ComplexNumber dir;
//    };
//    typedef std::list<_minutia> _minutiae;
	
	template<class T_in, class T_out, class T_supp>
	class matElwiseOp {
	private:
		/** Vector of values to be used during computation if needed. */
		std::vector<T_supp> support;
		/** Functor to be applied to every matrix element. */
		std::function<T_out(T_in&)> functor;
	public:
		/** Constructor with functor and support vector initialization. */
		matElwiseOp(std::function<T_out(T_in&)> functor, int n_support, T_supp init_value):
		functor(functor),
		support(n_support, init_value){
			
		}
		
		/** Apply the functor to each element and return the matrix of the results.
		 Uses T_out as element type of the output matrix.
		 */
		cv::Mat_<T_out> operator[](cv::Mat_<T_in> input){
			// If the input matrix is empty, return an empty matrix
			if (input.empty()) return cv::Mat_<T_out>();
			// Initialize the output matrix
			cv::Mat_<T_out> output(input.size());
			// Apply the functor to each element of the input matrix
			std::transform(input.begin(), input.end(), output.begin(), functor);
			// Returns the resulting matrix
			return output;
		}
		
		/** Apply the functor to each element and return a scalar value.
		 Uses T_out as the returning type and the first element of the
		 support vector as the returned element.
		 */
		T_out operator()(cv::Mat_<T_in> input){
			// If the input matrix is empty, throw exception
			if (input.empty()) std::runtime_error("Input matrix empty");
			// Apply the functor to each input matrix element
			std::for_each(input.begin(), input.end(), functor);
			// Return the first element of the support vector
			return support[0];
		}
	};
}

//class Timer
//{
//private:
//    float TickCount;
//public:
//    Timer(){Init();};
//    
//    void Init(){TickCount = (float)cv::getTickCount();}; // azzera il TickCount
//    float Time(){return ( (float)cv::getTickCount() - this->TickCount ) / (float)cv::getTickFrequency();};
//};

#endif /* TypesTraits_h */
