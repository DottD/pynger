/*
 In questo header file viene definita la procedura per riscalare un immagine
 */

#ifndef ImageRescale_hpp
#define ImageRescale_hpp

#include <opencv2/opencv.hpp>

namespace fm {
	
	/*
	 rescale(image, [maxVal])
	 
	 Riscala l'immagine in input affinché abbia valori in [0,maxVal].
	 L'opzione rounding permette di arrotondare il risultato all'intero più vicino.
	 Per default maxVal = 1 e l'arrotondamento non viene eseguito.
	 */
	//	template <class T = cv::Mat1d>
	//	void rescale(T& image, const typename T::value_type& maxVal = 1.0, const bool& rounding = false){
	//		/* Dichiarazioni */
	//		typename T::value_type diff;
	//		std::pair<typename T::iterator, typename T::iterator> minMaxRef;
	//
	//		/* Calcolo minimo e massimo tra i valori di input */
	//		minMaxRef = std::minmax_element(image.begin(), image.end());
	//		typename T::value_type& minElement = *std::get<0>(minMaxRef);
	//		typename T::value_type& maxElement = *std::get<1>(minMaxRef);
	//
	//		/* Normalizzo i valori di input tra 0 e 255 */
	//		diff = (maxElement-minElement);
	//		if (maxVal == 1.0)
	//			std::for_each(image.begin(), image.end(), [minElement, &diff] (typename T::value_type& x)
	//						  {
	//							  x = (x-minElement) / diff;
	//						  });
	//		else if ((maxVal != 1.0) && rounding)
	//			std::for_each(image.begin(), image.end(), [minElement, &diff, &maxVal] (typename T::value_type& x)
	//						  {
	//							  x = (x-minElement) / diff * maxVal;
	//						  });
	//		else if ((maxVal != 1.0) && !rounding)
	//			std::for_each(image.begin(), image.end(), [minElement, &diff, &maxVal] (typename T::value_type& x)
	//						  {
	//							  x = std::round((x-minElement) / diff * maxVal);
	//						  });
	//	};
	
	template <class T = double>
	void rescale_(cv::Mat& image,
				 const T& maxVal = 1,
				 const bool& rounding = false){
		/* Dichiarazioni */
		T diff;
		std::pair<cv::MatIterator_<T>, cv::MatIterator_<T>> minMaxRef;
		
		/* Calcolo minimo e massimo tra i valori di input */
		minMaxRef = std::minmax_element(image.begin<T>(), image.end<T>());
		T& minElement = *std::get<0>(minMaxRef);
		T& maxElement = *std::get<1>(minMaxRef);
		
		/* Normalizzo i valori di input tra 0 e 255 */
		diff = (maxElement-minElement);
		if (maxVal == 1)
			std::for_each(image.begin<T>(), image.end<T>(), [minElement, &diff] (T& x)
						  {
							  x = (x-minElement) / diff;
						  });
		else if ((maxVal != 1) && rounding)
			std::for_each(image.begin<T>(), image.end<T>(), [minElement, &diff, &maxVal] (T& x)
						  {
							  x = (x-minElement) / diff * maxVal;
						  });
		else if ((maxVal != 1) && !rounding)
			std::for_each(image.begin<T>(), image.end<T>(), [minElement, &diff, &maxVal] (T& x)
						  {
							  x = (x-minElement) / diff * maxVal;
						  });
	};
	
	void rescale(cv::InputOutputArray image_,
				 const double& maxVal = 1,
				 const bool& rounding = false);
}
#endif /* ImageRescale_hpp */
