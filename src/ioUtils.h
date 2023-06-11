#include <string>

std::string combinePath(std::string a, std::string b);

std::vector<std::string> getImagesSorted(const std::string imagesDirectory);

std::string fileNameWithoutExtension(std::string path);

std::vector<std::string> getTrainOrValidationSample(
    const std::vector<std::string> &fullSet,
    cv::RNG rng,
    float splitRate,
    bool isTrainSample);
