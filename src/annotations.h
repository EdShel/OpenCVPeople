#include <string>
#include <opencv2/core/types.hpp>

struct ImageAnnotation
{
    std::string FileName;
    cv::Rect Bbox;
};

int readAnnotations(const std::string file, std::vector<ImageAnnotation> &result);
int writeAnnotations(const std::string file, const std::vector<ImageAnnotation> &data);
