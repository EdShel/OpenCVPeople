#include "annotations.h"
#include "utils.h"
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