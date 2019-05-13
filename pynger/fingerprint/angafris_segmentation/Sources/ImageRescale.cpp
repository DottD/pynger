#include "../Headers/ImageRescale.hpp"

void fm::rescale(cv::InputOutputArray image_,
				 const double& maxVal,
				 const bool& rounding){
	if(image_.isMat()){
		std::vector<cv::Mat> channelsVector;
		cv::split(image_, channelsVector);
		for(cv::Mat& channel: channelsVector){
			switch (channel.depth()) {
				case CV_8U: rescale_<unsigned char>(channel, (unsigned char)maxVal, rounding); break;
				case CV_8S: rescale_<char>(channel, (char)maxVal, rounding); break;
				case CV_16U: rescale_<unsigned short>(channel, (unsigned short)maxVal, rounding); break;
				case CV_16S: rescale_<short>(channel, (short)maxVal, rounding); break;
				case CV_32S: rescale_<int>(channel, (int)maxVal, rounding); break;
				case CV_32F: rescale_<float>(channel, (float)maxVal, rounding); break;
				case CV_64F: rescale_<double>(channel, (double)maxVal, rounding); break;
					default: throw "image has a not recognised value type";
			}
		}
		cv::merge(channelsVector, image_);
	} else {
		throw "image_ is not a cv::Mat";
	}
};
