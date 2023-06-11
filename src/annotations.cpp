#include "annotations.h"
#include "ioUtils.h"
#include <set>
#include <fstream>
#include <iostream>

int readAnnotations(const std::string file, std::vector<ImageAnnotation> &result)
{
    std::ifstream f;
    f.open(file);
    if (!f.is_open())
    {
        std::cout << "Can't open annotations file " << file << std::endl;
        return 1;
    }

    while (true)
    {
        std::string imageFileName;
        int x1, y1, x2, y2;
        f >> imageFileName;
        f >> y1;
        f >> x1;
        f >> y2;
        f >> x2;

        if (f.eof())
        {
            f.close();
            return 0;
        }

        ImageAnnotation annotation;
        annotation.FileName = imageFileName + ".jpg";
        annotation.Bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        result.push_back(annotation);
    }
}

int writeAnnotations(const std::string file, const std::vector<ImageAnnotation> &data)
{
    std::ofstream f;
    f.open(file);
    if (!f.is_open())
    {
        std::cout << "Can't open file to save annotations " << file << std::endl;
        return 1;
    }

    for (int i = 0; i < data.size(); i++)
    {
        ImageAnnotation annotation = data[i];
        std::string imageFileName = fileNameWithoutExtension(annotation.FileName);
        cv::Rect bbox = annotation.Bbox;
        f << imageFileName
          << '\t' << bbox.y
          << '\t' << bbox.x
          << '\t' << bbox.y + bbox.height
          << '\t' << bbox.x + bbox.width
          << '\n';
    }

    f.close();

    return 0;
}

void evaluateDetectionAnnotations(
    const std::vector<ImageAnnotation> &actual,
    const std::vector<ImageAnnotation> &detected)
{
    std::set<std::string> allImages;
    for (auto b = actual.begin(), e = actual.end(); b != e; b++)
    {
        allImages.insert(b->FileName);
    }
    for (auto b = detected.begin(), e = detected.end(); b != e; b++)
    {
        allImages.insert(b->FileName);
    }

    int truePositives = 0;  // Detected pedestrians who overlap at least 50% with the correct pedestrians
    int falsePositives = 0; // Detected pedestrians who overlap less than 50% with the correct pedestrians
    int actualPeopleCount = actual.size();

    for (auto b = allImages.begin(), e = allImages.end(); b != e; b++)
    {
        std::string currentImage = *b;
        std::vector<cv::Rect> actualPeople;
        std::vector<cv::Rect> detectedPeople;

        for (int i = 0; i < actual.size(); i++)
        {
            if (actual[i].FileName == currentImage)
            {
                actualPeople.push_back(actual[i].Bbox);
            }
        }
        for (int i = 0; i < detected.size(); i++)
        {
            if (detected[i].FileName == currentImage)
            {
                detectedPeople.push_back(detected[i].Bbox);
            }
        }

        if (actualPeople.size() == 0)
        {
            falsePositives += detectedPeople.size();
            continue;
        }

        if (detectedPeople.size() == 0)
        {
            continue;
        }

        for (int di = 0; di < detectedPeople.size(); di++)
        {
            bool isCorrect = false;
            cv::Rect detectedBox = detectedPeople[di];
            for (int ai = 0; ai < actualPeople.size(); ai++)
            {
                cv::Rect actualBox = actualPeople[ai];
                int actualAreaSize = actualBox.area();
                int overlapAreaSize = (actualBox & detectedBox).area();
                if (overlapAreaSize >= actualAreaSize / 2)
                {
                    isCorrect = true;
                    break;
                }
            }

            if (isCorrect)
            {
                truePositives++;
            }
            else
            {
                falsePositives++;
            }
        }
    }

    double recall = static_cast<double>(truePositives) / actualPeopleCount;
    double precision = static_cast<double>(truePositives) / (truePositives + falsePositives);

    std::cout << "Recall   : " << recall << std::endl;
    std::cout << "Precision: " << precision << std::endl;
}