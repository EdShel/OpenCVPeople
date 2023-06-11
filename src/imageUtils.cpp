#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void imresizeContain(const cv::Mat &source, cv::Mat &dest, const cv::Size destinationSize)
{
    cv::Size sourceSize = source.size();

    double scaleX = static_cast<double>(destinationSize.width) / static_cast<double>(sourceSize.width);
    double scaleY = static_cast<double>(destinationSize.height) / static_cast<double>(sourceSize.height);

    double smallestScale = std::min(scaleX, scaleY);

    cv::Mat scaledSource;
    cv::resize(source, scaledSource, cv::Size(0, 0), smallestScale, smallestScale);
    int paddingLeft = (destinationSize.width - scaledSource.size().width) / 2;
    int paddingTop = (destinationSize.height - scaledSource.size().height) / 2;
    int paddingRight = destinationSize.width - scaledSource.size().width - paddingLeft;
    int paddingBottom = destinationSize.height - scaledSource.size().height - paddingTop;

    cv::copyMakeBorder(scaledSource, dest, paddingTop, paddingBottom, paddingLeft, paddingRight, cv::BORDER_CONSTANT, cv::Scalar(0));
}

std::vector<cv::Rect> findNonOverlappingBoxes(const std::vector<cv::Rect> &rectangles)
{
    std::vector<cv::Rect> overlaps;
    for (int i = 0; i < rectangles.size(); i++)
    {
        cv::Rect currentRect = rectangles[i];
        bool merged = false;
        for (int j = 0; j < overlaps.size(); j++)
        {
            cv::Rect existingOverlap = overlaps[j];
            cv::Rect overlap = currentRect & existingOverlap;
            if (overlap.area() > 0)
            {
                overlaps[j] = currentRect | existingOverlap;
                merged = true;
                break;
            }
        }

        if (!merged)
        {
            overlaps.push_back(currentRect);
        }
    }

    if (rectangles.size() != overlaps.size())
    {
        return findNonOverlappingBoxes(overlaps);
    }
    return overlaps;
}

bool overlapsAny(const cv::Rect &rect, const std::vector<cv::Rect> &rects)
{
    for (int i = 0; i < rects.size(); i++)
    {
        if ((rect & rects[i]).area() > 0)
        {
            return true;
        }
    }
    return false;
}

std::vector<cv::Rect> findBoxesOnBlackBackground(cv::Mat grayscaleImage)
{
    cv::Mat grayscale = grayscaleImage.clone();
    cv::blur(grayscale, grayscale, cv::Size(13, 13));

    cv::Mat binaryImage;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::threshold(grayscale, binaryImage, 5, 255, cv::THRESH_BINARY);
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    std::vector<cv::Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
    }

    return findNonOverlappingBoxes(boundRect);
}
