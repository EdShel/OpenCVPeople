#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <set>
#include "annotations.h"
#include "utils.h"

enum Label
{
    LABEL_PERSON = 1,
    LABEL_BACKGROUND = 2
};

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

void createHog(const cv::FileStorage &params, cv::HOGDescriptor &hog)
{
    hog.winSize = cv::Size(params["windowSizeX"], params["windowSizeY"]);
    hog.histogramNormType = cv::HOGDescriptor::HistogramNormType::L2Hys;

    hog.blockSize.width = params["blockSizeX"];
    hog.blockSize.height = params["blockSizeY"];

    hog.blockStride.width = params["blockStrideX"];
    hog.blockStride.height = params["blockStrideY"];

    hog.cellSize.width = params["cellSizeX"];
    hog.cellSize.height = params["cellSizeY"];

    hog.nbins = 9;
    hog.derivAperture = params["derivAperture"];
    hog.winSigma = params["winSigma"];
    hog.L2HysThreshold = params["L2HysThreshold"];
    hog.gammaCorrection = static_cast<int>(params["gammaCorrection"]) != 0;
    hog.nlevels = params["nlevels"];
    hog.signedGradient = static_cast<int>(params["signedGradient"]) != 0;
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

int main(int argc, char *argv[])
{
    // cv::Mat img = cv::imread("../simple/images/6.jpg");
    // std::vector<cv::Rect> boxes = findBoxesOnBlackBackground(img);

    // for(int i = 0; i < boxes.size(); i++)
    // {
    //     cv::rectangle(img, boxes[i], cv::Scalar(0, 0, 255));
    // }

    // cv::Mat rsz;
    // imresizeContain(img, rsz, cv::Size(1280, 666));

    // cv::imshow("TEST1", img);
    // cv::imshow("TEST2", rsz);
    // cv::waitKey(0);

    // return 0;

    std::string imagesDir = "../simple/images/";
    std::string annotationsFile = "../simple/bboxes.txt";
    std::string paramsFile = "../params.yml";

    cv::FileStorage params(paramsFile, cv::FileStorage::READ);

    cv::HOGDescriptor hog;
    createHog(params, hog);

    std::vector<float> descriptors;
    std::vector<cv::Mat> trainDataList;
    std::vector<int> labelsList;
    int trainRngSeed = params["trainRngSeed"];
    cv::RNG rng(trainRngSeed);
    cv::Size windowSize(params["windowSizeX"], params["windowSizeY"]);

    std::vector<ImageAnnotation> annotations;
    if (readAnnotations(annotationsFile, annotations) != 0)
    {
        return 1;
    }

    std::vector<std::string> trainImages = getImagesSorted(imagesDir);

    for (auto b = trainImages.begin(), e = trainImages.end(); b != e; b++)
    {
        std::string imageFile = *b;

        std::cout << imageFile << std::endl;
        std::string imagePath = combinePath(imagesDir, imageFile);
        cv::Mat trainImage = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);
        if (trainImage.empty())
        {
            std::cout << "Cannot open image " << imagePath << std::endl;
            return 1;
        }
        cv::Mat colImage = cv::imread(imagePath);

        std::vector<cv::Rect> peopleBoxes;
        for (int i = 0; i < annotations.size(); i++)
        {
            if (annotations[i].FileName == imageFile)
            {
                peopleBoxes.push_back(annotations[i].Bbox);
                cv::rectangle(colImage, annotations[i].Bbox, cv::Scalar(0, 0, 255));
            }
        }

        std::vector<cv::Rect> contourBoxes = findBoxesOnBlackBackground(trainImage);
        std::vector<cv::Rect> backgroundBoxes;
        for (int i = 0; i < contourBoxes.size(); i++)
        {
            if (!overlapsAny(contourBoxes[i], peopleBoxes))
            {
                backgroundBoxes.push_back(contourBoxes[i]);
                cv::rectangle(colImage, contourBoxes[i], cv::Scalar(0, 255, 255));
            }
        }

        std::vector<cv::Rect> imageBoxes;
        std::vector<Label> imageLabels;
        for (int i = 0; i < peopleBoxes.size(); i++)
        {
            imageBoxes.push_back(peopleBoxes[i]);
            imageLabels.push_back(Label::LABEL_PERSON);
        }
        for (int i = 0; i < backgroundBoxes.size(); i++)
        {
            imageBoxes.push_back(backgroundBoxes[i]);
            imageLabels.push_back(Label::LABEL_BACKGROUND);
        }

        cv::imshow("Original", colImage);

        for (int i = 0; i < imageBoxes.size(); i++)
        {
            const cv::Rect box = imageBoxes[i];
            const Label label = imageLabels[i];

            cv::Mat sliceImage = trainImage(box);
            cv::Mat windowImage;
            imresizeContain(sliceImage, windowImage, windowSize);

            cv::namedWindow("Original slice", cv::WINDOW_AUTOSIZE);
            cv::imshow("Original slice", sliceImage);

            cv::namedWindow("Slice", cv::WINDOW_AUTOSIZE);
            cv::imshow("Slice", windowImage);
            if (cv::waitKey(10000) == -1)
            {
                return 0;
            }

            hog.compute(windowImage, descriptors);
            trainDataList.push_back(cv::Mat(descriptors).clone());
            labelsList.push_back(label);
        }

        // hog.compute(trainImage());
        // cv::imshow("Test image", colorfulImage);
    }

    return 0;
}

// std::vector<int> argsort(const std::vector<cv::Rect> &array)
// {
//     std::vector<int> indices(array.size());
//     std::iota(indices.begin(), indices.end(), 0);
//     std::sort(indices.begin(), indices.end(),
//               [&array](int left, int right) -> bool
//               {
//                   // sort indices according to corresponding array element
//                   return array[left].x == array[right].x
//                     ? array[left].y < array[right].y
//                     : array[left].x < array[right].x;
//               });

//     return indices;
// }

// /*!
// \brief Applies the Non Maximum Suppression algorithm on the detections to find the detections that do not overlap

// The svm response is used to sort the detections. Translated from http://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

// \param boxes list of detections that are the input for the NMS algorithm
// \param overlap_threshold the area threshold for the overlap between detections boxes. boxes that have overlapping area above threshold are discarded

// \returns list of final detections that are no longer overlapping
// */
// std::vector<cv::Rect> nonMaximumSuppression(std::vector<cv::Rect> boxes, float overlap_threshold)
// {
//     std::vector<cv::Rect> res;
//     std::vector<float> areas;

//     // if there are no boxes, return empty
//     if (boxes.size() == 0)
//         return res;

//     for (int i = 0; i < boxes.size(); i++)
//         areas.push_back(boxes[i].area());

//     std::vector<int> idxs = argsort(boxes);

//     std::vector<int> pick; // indices of final detection boxes

//     while (idxs.size() > 0) // while indices still left to analyze
//     {
//         int last = idxs.size() - 1; // last element in the list. that is, detection with highest SVM response
//         int i = idxs[last];
//         pick.push_back(i); // add highest SVM response to the list of final detections

//         std::vector<int> suppress;
//         suppress.push_back(last);

//         for (int pos = 0; pos < last; pos++) // for every other element in the list
//         {
//             int j = idxs[pos];

//             // find overlapping area between boxes
//             int xx1 = std::max(boxes[i].x, boxes[j].x);           // get max top-left corners
//             int yy1 = std::max(boxes[i].y, boxes[j].y);           // get max top-left corners
//             int xx2 = std::min(boxes[i].br().x, boxes[j].br().x); // get min bottom-right corners
//             int yy2 = std::min(boxes[i].br().y, boxes[j].br().y); // get min bottom-right corners

//             int w = std::max(0, xx2 - xx1 + 1); // width
//             int h = std::max(0, yy2 - yy1 + 1); // height

//             float overlap = float(w * h) / areas[j];

//             if (overlap > overlap_threshold) // if the boxes overlap too much, add it to the discard pile
//                 suppress.push_back(pos);
//         }

//         for (int p = 0; p < suppress.size(); p++) // for graceful deletion
//         {
//             idxs[suppress[p]] = -1;
//         }

//         for (int p = 0; p < idxs.size();)
//         {
//             if (idxs[p] == -1)
//                 idxs.erase(idxs.begin() + p);
//             else
//                 p++;
//         }
//     }

//     for (int i = 0; i < pick.size(); i++) // extract final detections frm input array
//         res.push_back(boxes[pick[i]]);

//     return res;
// }