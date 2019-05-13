#include "../Headers/AdaptiveThreshold.hpp"

float fm::adaptiveThreshold(const cv::Mat1f& image,
                            const float& hint){
    /* Declarations */
    float oldt, t, overMean, underMean;
    int overCount, underCount;
    std::function<void(const float&)> tSum = [&t](const float& a){t += a;};
    /* Checks hint */
    if (hint!= FLT_MIN){
        t = hint;
    } else {
        /* Assigns the image mean value to t */
        t = 0.0f;
        std::for_each(image.begin(), image.end(), tSum);
        t /= image.total();
    }
    /* Does the followign until convergence */
    do{
        /* Reset local variables */
        overMean = 0.0f;
        underMean = 0.0f;
        overCount = 0;
        underCount = 0;
        /* Uses the current threshold */
        for(const float& f: image){
            if (f > t){
                overMean += f;
                overCount++;
            } else {
                underMean += f;
                underCount++;
            }
        }
        overMean /= overCount;
        underMean /= underCount;
        /* If no value exceed the threshold the image if flat, so break the cycle */
        if (overCount == 0){
            break;
        }
        /* Compute new threshold and check convergence */
        oldt = t;
        t = 0.5f * (overMean+underMean);
    }while( std::abs(1.0f - oldt/t) > 2e-3 );
    return t;
}

void fm::medianAdaptiveThreshold(cv::Mat1b& bin,
                                 const cv::Mat1f& image,
                                 const cv::Mat1b& mask,
                                 const int& binarizationStep,
                                 const float& percentile){
    /* Declarations */
    int i, j, medianPosition = int(percentile/100.0*float(binarizationStep*binarizationStep));
    cv::Rect2i ROI(0, 0, binarizationStep, binarizationStep);
    cv::Mat1f block;
    cv::Mat1f::iterator medianIt;
    /* Initialize bin */
    bin = cv::Mat1f::zeros(image.rows, image.cols);
    /* Cycle through the image pixels */
    for(i = binarizationStep; i < image.rows-binarizationStep; i++){
        for(j = binarizationStep; j < image.cols-binarizationStep; j++){
            /* Check if the current pixel is in the mask, otherwise set the current pixel to 1 in the binary image */
            if(mask(i,j) != 0){
                /* Update the ROI to isolate the current block */
                ROI.x = j-binarizationStep;
                ROI.y = i-binarizationStep;
                /* Compute the median value of the current block and assign to bin */
                image(ROI).copyTo(block);
                medianIt = block.begin()+medianPosition;
                std::nth_element(block.begin(), medianIt, block.end());
                if (image(i,j) > *medianIt) bin(i,j) = 1;
                else bin(i,j) = 0;
            } else bin(i,j) = 0;
        }
    }
}
