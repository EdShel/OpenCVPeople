#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include "ioUtils.h"
#include "imageUtils.h"
#include "annotations.h"

enum Label
{
    LABEL_PERSON = 1,
    LABEL_BACKGROUND = 2
};

int testMain();

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

    hog.nbins = params["nbins"];
    hog.derivAperture = params["derivAperture"];
    hog.winSigma = params["winSigma"];
    hog.L2HysThreshold = params["L2HysThreshold"];
    hog.gammaCorrection = static_cast<int>(params["gammaCorrection"]) != 0;
    hog.nlevels = params["nlevels"];
    hog.signedGradient = static_cast<int>(params["signedGradient"]) != 0;
}

int evaluateMain()
{
    std::string actualAnnotationsFile = "../simple/bboxes.txt";
    std::string imagesDir = "../simple/images/";
    std::string paramsFile = "../params.yml";
    std::string outputFile = "../model.yml";

    cv::FileStorage params(paramsFile, cv::FileStorage::READ);
    int sampleRngSeed = params["sampleRngSeed"];
    float sampleSplitRatio = params["sampleSplitRatio"];

    auto allImages = getImagesSorted(imagesDir);
    auto validationSample = getTrainOrValidationSample(allImages, cv::RNG(sampleRngSeed), sampleSplitRatio, false);

    std::vector<ImageAnnotation> actualAnnotations;
    if (readAnnotations(actualAnnotationsFile, actualAnnotations) != 0)
    {
        std::cout << "Can't read actual annotations" << std::endl;
        return 1;
    }

    std::vector<ImageAnnotation> validationAnnotations;
    for (int i = 0; i < actualAnnotations.size(); i++)
    {
        std::string fileName = actualAnnotations[i].FileName;
        bool isAnnotationFromValidationSet = std::find(validationSample.begin(), validationSample.end(), fileName) != validationSample.end();
        if (isAnnotationFromValidationSet)
        {
            validationAnnotations.push_back(actualAnnotations[i]);
        }
    }

    std::string detectedAnnotationsFile = "../results.txt";
    std::vector<ImageAnnotation> detectedAnnotations;
    if (readAnnotations(detectedAnnotationsFile, detectedAnnotations) != 0)
    {
        std::cout << "Can't read detected annotations" << std::endl;
        return 1;
    }

    evaluateDetectionAnnotations(validationAnnotations, detectedAnnotations);
}

int main(int argc, char *argv[])
{
    evaluateMain();
    return 0;
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
    std::string outputFile = "../model.yml";

    cv::FileStorage params(paramsFile, cv::FileStorage::READ);

    cv::HOGDescriptor hog;
    createHog(params, hog);

    std::vector<float> descriptors;
    std::vector<cv::Mat> trainDataList;
    std::vector<int> labelsList;
    cv::Size windowSize(params["windowSizeX"], params["windowSizeY"]);
    int sampleRngSeed = params["sampleRngSeed"];
    float sampleSplitRatio = params["sampleSplitRatio"];

    std::vector<ImageAnnotation> annotations;
    if (readAnnotations(annotationsFile, annotations) != 0)
    {
        return 1;
    }

    std::vector<std::string> allImages = getImagesSorted(imagesDir);
    std::vector<std::string> trainImages = getTrainOrValidationSample(allImages, cv::RNG(sampleRngSeed), sampleSplitRatio, true);

    for (auto b = trainImages.begin(), e = trainImages.end(); b != e; b++)
    {
        std::string imageFile = *b;

        std::string imagePath = combinePath(imagesDir, imageFile);
        cv::Mat trainImage = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);
        if (trainImage.empty())
        {
            std::cout << "Cannot open image " << imagePath << std::endl;
            return 1;
        }
        std::vector<cv::Rect> peopleBoxes;
        for (int i = 0; i < annotations.size(); i++)
        {
            if (annotations[i].FileName == imageFile)
            {
                peopleBoxes.push_back(annotations[i].Bbox);
            }
        }

        std::vector<cv::Rect> contourBoxes = findBoxesOnBlackBackground(trainImage);
        std::vector<cv::Rect> backgroundBoxes;
        for (int i = 0; i < contourBoxes.size(); i++)
        {
            if (!overlapsAny(contourBoxes[i], peopleBoxes))
            {
                backgroundBoxes.push_back(contourBoxes[i]);
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

        // cv::imshow("Original", colImage);

        for (int i = 0; i < imageBoxes.size(); i++)
        {
            const cv::Rect box = imageBoxes[i];
            const Label label = imageLabels[i];

            cv::Mat sliceImage = trainImage(box);
            cv::Mat windowImage;
            imresizeContain(sliceImage, windowImage, windowSize);

            // cv::namedWindow("Original slice", cv::WINDOW_AUTOSIZE);
            // cv::imshow("Original slice", sliceImage);

            // cv::namedWindow("Slice", cv::WINDOW_AUTOSIZE);
            // cv::imshow("Slice", windowImage);
            // if (cv::waitKey(10000) == -1)
            // {
            //     return 0;
            // }

            hog.compute(windowImage, descriptors);
            trainDataList.push_back(cv::Mat(descriptors).clone());
            labelsList.push_back(label);
        }

        // hog.compute(trainImage());
        // cv::imshow("Test image", colorfulImage);
    }

    int trainDataRows = trainDataList.size();
    int trainDataCols = trainDataList[0].rows;
    cv::Mat trainDataMatrix(trainDataRows, trainDataCols, CV_32FC1);
    cv::Mat transposeTmpMatrix(1, trainDataCols, CV_32FC1);

    for (size_t i = 0; i < trainDataList.size(); i++)
    {
        cv::transpose(trainDataList[i], transposeTmpMatrix);
        transposeTmpMatrix.copyTo(trainDataMatrix.row((int)i));
    }

    auto svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->trainAuto(trainDataMatrix, cv::ml::ROW_SAMPLE, labelsList);

    svm->save(outputFile);

    return testMain();

    // return 0;
}

cv::Mat stdVectorToSamplesCvMat(std::vector<cv::Mat> &vec)
{
    int testDataRows = vec.size();
    int testDataCols = vec[0].rows;
    cv::Mat testDataMatrix(testDataRows, testDataCols, CV_32FC1);
    cv::Mat transposeTmpMatrix(1, testDataCols, CV_32FC1);

    for (size_t i = 0; i < vec.size(); i++)
    {
        cv::transpose(vec[i], transposeTmpMatrix);
        transposeTmpMatrix.copyTo(testDataMatrix.row((int)i));
    }

    return testDataMatrix;
}

int detectPeople(
    const cv::Ptr<cv::ml::SVM> &svm,
    const cv::HOGDescriptor &hog,
    const cv::Mat image,
    std::vector<cv::Rect> &locations)
{
    std::vector<float> descriptors;
    std::vector<cv::Mat> testDataList;

    std::vector<cv::Rect> boxes = findBoxesOnBlackBackground(image);
    for (int i = 0; i < boxes.size(); i++)
    {
        cv::Mat imageObject = image(boxes[i]);
        cv::Mat resizedObject;
        imresizeContain(imageObject, resizedObject, hog.winSize);

        hog.compute(resizedObject, descriptors);
        testDataList.push_back(cv::Mat(descriptors).clone());
    }

    cv::Mat testDataMatrix = stdVectorToSamplesCvMat(testDataList);

    cv::Mat results;
    svm->predict(testDataMatrix, results, cv::ml::ROW_SAMPLE);

    for (int i = 0; i < results.rows && i < boxes.size(); i++)
    {
        if (results.at<float>(i, 0) != Label::LABEL_PERSON)
        {
            continue;
        }

        locations.push_back(boxes[i]);
    }

    return 0;
}

int testMain()
{
    std::string classifierCoefficientsFile = "../model.yml";
    std::string paramsFile = "../params.yml";
    std::string outputAnnotationsFile = "../results.txt";
    std::string imagesDir = "../simple/images/";
    auto svm = cv::ml::SVM::load(classifierCoefficientsFile);

    cv::FileStorage params(paramsFile, cv::FileStorage::READ);
    cv::HOGDescriptor hog;
    createHog(params, hog);

    int sampleRngSeed = params["sampleRngSeed"];
    float sampleSplitRatio = params["sampleSplitRatio"];

    std::vector<cv::Rect> results;

    bool shouldShow = true;

    std::vector<std::string> allImages = getImagesSorted(imagesDir);
    std::vector<std::string> testImages = getTrainOrValidationSample(allImages, cv::RNG(sampleRngSeed), sampleSplitRatio, false);
    std::vector<ImageAnnotation> resultAnnotations;
    for (auto b = testImages.begin(), e = testImages.end(); b != e; b++)
    {
        std::string imageFile = *b;
        std::string imagePath = combinePath(imagesDir, imageFile);
        cv::Mat testImage = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);
        if (testImage.empty())
        {
            std::cout << "Cannot open image " << imagePath << std::endl;
            return 1;
        }

        std::vector<cv::Rect> detectionBoxes;
        if (detectPeople(svm, hog, testImage, detectionBoxes) != 0)
        {
            std::cout << "Error during detection" << std::endl;
            return 1;
        }

        for (int i = 0; i < detectionBoxes.size(); i++)
        {
            ImageAnnotation a;
            a.FileName = imageFile;
            a.Bbox = detectionBoxes[i];
            resultAnnotations.push_back(a);
        }

        if (shouldShow)
        {
            cv::Mat colorfulImage = cv::imread(imagePath);
            for (int i = 0; i < detectionBoxes.size(); i++)
            {
                cv::rectangle(colorfulImage, detectionBoxes[i], cv::Scalar(0, 0, 255));
            }
            cv::imshow("Detection", colorfulImage);
            if (cv::waitKey(5000) == -1)
            {
                shouldShow = false;
            }
        }
    }

    if (writeAnnotations(outputAnnotationsFile, resultAnnotations) != 0)
    {
        std::cout << "Can't save annotations" << std::endl;
        return 1;
    }

    return 0;
}
