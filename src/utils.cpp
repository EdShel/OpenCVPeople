#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

std::string combinePath(std::string a, std::string b)
{
    char lastChar = a.at(a.length() - 1);
    if (lastChar == '/' || lastChar == '\\')
    {
        return a + b;
    }
    return a + '/' + b;
}

std::string fileNameWithoutExtension(std::string path)
{
    int dot = path.find_last_of(".");
    int slash = path.find_last_of("/\\");
    return path.substr(slash + 1, dot - slash - 1);
}

std::string fileNameWithExtension(std::string path)
{
    int slash = path.find_last_of("/\\");
    return path.substr(slash + 1);
}

std::vector<std::string> getImagesSorted(const std::string imagesDirectory)
{
    std::vector<std::string> files;
    cv::glob(imagesDirectory + "*.jpg", files, false);
    std::vector<std::string> result;
    for (auto b = files.begin(), e = files.end(); b != e; b++)
    {
        std::string filePath = *b;
        std::string fileName = fileNameWithExtension(filePath);
        result.push_back(fileName);
    }
    std::sort(result.begin(), result.end());

    return result;
}
