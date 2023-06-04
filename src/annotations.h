#include <string>
#include <opencv2/core/types.hpp>

struct ImageAnnotation
{
    std::string FileName;
    cv::Rect Bbox;
};

int readAnnotations(std::string file, std::vector<ImageAnnotation> &result);
