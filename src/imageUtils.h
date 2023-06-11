#include <vector>
#include <opencv2/core/core.hpp>

void imresizeContain(const cv::Mat &source, cv::Mat &dest, const cv::Size destinationSize);

bool overlapsAny(const cv::Rect &rect, const std::vector<cv::Rect> &rects);

std::vector<cv::Rect> findBoxesOnBlackBackground(cv::Mat grayscaleImage);
