#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <cnpy.h>

const double COSINE_THRESHOLD = 0.363;


std::map<std::string, cv::Mat> nameToFeatures;


bool match(cv::Ptr<cv::FaceRecognizerSF> recognizer, const cv::Mat& feature1, const std::map<std::string, cv::Mat>& dictionary, std::pair<std::string, double>& result) {
    for (const auto& entry : dictionary) {
        double score = recognizer->match(feature1, entry.second, cv::FaceRecognizerSF::DisType::FR_COSINE);
        if (score > COSINE_THRESHOLD) {
            result = std::make_pair(entry.first, score);
            return true;
        }
    }
    result = std::make_pair("unknown", 0.0);
    return false;
}

int main() {
    
    cv::VideoCapture capture("http://root:xs83761188@10.10.10.186/video1s1.mjpg");
    if (!capture.isOpened()) {
        std::cerr << "Failed to open camera." << std::endl;
        return 1;
    }

    
    std::string filename = "Vincent.npy";
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    if (arr.shape.size() == 0) {
        std::cerr << "Failed to load " << filename << std::endl;
        return 1;
    }

    
    cv::Mat features(arr.shape[0], arr.shape[1], CV_32F, arr.data<float>());

    
    nameToFeatures["Vincent"] = features;

    
    std::string weightsFaceDetector = "yunet.onnx";
    std::string weightsFaceRecognizer = "sface.onnx";
    cv::Ptr<cv::FaceDetectorYN> faceDetector = cv::FaceDetectorYN::create(weightsFaceDetector, "", cv::Size(0, 0));
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer = cv::FaceRecognizerSF::create(weightsFaceRecognizer, "");

    while (true) {
        
        cv::Mat image;
        capture.read(image);
        if (image.empty()) {
            std::cerr << "Failed to capture frame." << std::endl;
            break;
        }

        
        int channels = image.channels();
        if (channels == 1) {
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        }
        else if (channels == 4) {
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        }

        int width = image.cols;
        int height = image.rows;
        faceDetector->setInputSize(cv::Size(width, height));

        
        cv::Mat faces;
        faceDetector->detect(image, faces);

        for (int i = 0; i < faces.rows; i++) {
           
            cv::Rect faceRect(faces.at<float>(i, 0), faces.at<float>(i, 1), faces.at<float>(i, 2), faces.at<float>(i, 3));

            
            cv::Rect imgRect(0, 0, image.cols, image.rows);
            cv::Rect finalRect = faceRect & imgRect;

            
            cv::Mat aligned_face = image(finalRect);
            cv::Mat feature;
            faceRecognizer->feature(aligned_face, feature);

            
            std::pair<std::string, double> result;
            bool matched = match(faceRecognizer, feature, nameToFeatures, result);

            
            cv::Scalar color = matched ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            int thickness = 2;
            cv::rectangle(image, faceRect, color, thickness, cv::LINE_AA);
            std::string text = matched ? result.first + " (" + std::to_string(result.second) + ")" : "unknown";
            cv::putText(image, text, cv::Point(faceRect.x, faceRect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, thickness, cv::LINE_AA);
        }

        
        cv::imshow("face recognition", image);

        
        int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
