/*
sudo apt-get update
sudo apt-get install libopencv-dev wiringpi g++ libcudnn8 tensorrt
g++ main.cpp -o tracking `pkg-config --cflags --libs opencv4` -lwiringPi -lnvinfer -lnvrtc -lcuda
./tracking
*/

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <wiringPi.h>
#include <softPwm.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

class ServoKit {
public:
    ServoKit(int channels) : channels_(channels), angles_(channels, 90) {
        wiringPiSetupGpio();
        for (int i = 0; i < channels; ++i) {
            softPwmCreate(i, 0, 200);
        }
    }

    void set_angle(int channel, int angle) {
        if (channel >= 0 && channel < channels_ && angle >= 0 && angle <= 180) {
            angles_[channel] = angle;
            int pwmValue = 5 + (angle * 20 / 180);
            softPwmWrite(channel, pwmValue);
        }
    }

    int get_angle(int channel) const {
        if (channel >= 0 && channel < channels_) {
            return angles_[channel];
        }
        return -1;
    }

private:
    int channels_;
    std::vector<int> angles_;
};

class TrackingConfig {
public:
    TrackingConfig(
        const std::string& source = "0",
        const std::vector<int>& classes = {0},
        float conf = 0.25,
        const std::string& device = "0",
        bool save = true,
        bool show = true,
        const cv::Size& img_size = cv::Size(640, 480),
        const std::string& engine_path = "yolov8n.engine"
    ) : 
        source_(source),
        classes_(classes),
        conf_threshold_(conf),
        device_(device),
        save_video_(save),
        show_video_(show),
        img_size_(img_size),
        engine_path_(engine_path) {
        
        std::filesystem::create_directories("runs/detect");
        std::filesystem::create_directories("runs/track");
    }

    // Getters for configuration parameters
    std::string source() const { return source_; }
    std::vector<int> classes() const { return classes_; }
    float conf_threshold() const { return conf_threshold_; }
    std::string device() const { return device_; }
    bool save_video() const { return save_video_; }
    bool show_video() const { return show_video_; }
    cv::Size img_size() const { return img_size_; }
    std::string engine_path() const { return engine_path_; }

private:
    std::string source_;
    std::vector<int> classes_;
    float conf_threshold_;
    std::string device_;
    bool save_video_;
    bool show_video_;
    cv::Size img_size_;
    std::string engine_path_;
};

class YOLOv8TRT {
public:
    YOLOv8TRT(const std::string& enginePath) {
        runtime_ = createInferRuntime(gLogger);
        std::ifstream engineFile(enginePath, std::ios::binary);
        engineFile.seekg(0, std::ios::end);
        const size_t engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineData(engineSize);
        engineFile.read(engineData.data(), engineSize);
        engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
        context_ = engine_->createExecutionContext();

        inputIndex_ = engine_->getBindingIndex("images");
        outputIndex_ = engine_->getBindingIndex("output0");

        DimsCHW inputDims = static_cast<DimsCHW&>(engine_->getBindingDimensions(inputIndex_));
        inputW_ = inputDims.w;
        inputH_ = inputDims.h;
        inputC_ = inputDims.c;

        cudaMalloc(&inputBuffer_, inputC_ * inputH_ * inputW_ * sizeof(float));
        cudaMalloc(&outputBuffer_, engine_->getBindingDimensions(outputIndex_).d[1] * sizeof(float));
    }

    ~YOLOv8TRT() {
        cudaFree(inputBuffer_);
        cudaFree(outputBuffer_);
        context_->destroy();
        engine_->destroy();
        runtime_->destroy();
    }

    std::vector<std::vector<float>> detect(const cv::Mat& image, float conf_threshold) {
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(inputW_, inputH_));
        cv::Mat blob = cv::dnn::blobFromImage(resizedImage, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

        cudaMemcpy(inputBuffer_, blob.ptr<float>(), inputC_ * inputH_ * inputW_ * sizeof(float), cudaMemcpyHostToDevice);

        void* buffers[] = {inputBuffer_, outputBuffer_};
        context_->execute(1, buffers);

        std::vector<float> output(engine_->getBindingDimensions(outputIndex_).d[1]);
        cudaMemcpy(output.data(), outputBuffer_, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<std::vector<float>> detections;

        int numDetections = output.size() / 6;
        for (int i = 0; i < numDetections; ++i) {
            std::vector<float> detection(output.begin() + i * 6, output.begin() + (i + 1) * 6);
            if(detection[4] > conf_threshold) detections.push_back(detection);
        }

        return detections;
    }

private:
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    int inputIndex_;
    int outputIndex_;
    int inputW_;
    int inputH_;
    int inputC_;
    void* inputBuffer_;
    void* outputBuffer_;
    static ILogger gLogger;
};

class ILogger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

ILogger YOLOv8TRT::gLogger = ILogger(); // Initialize static logger

class ServoTracker {
public:
    ServoTracker(ServoKit& servo_kit) : 
        servo_kit_(servo_kit), 
        pan_(90), 
        tilt_(45) {
        // Initialize servos to default position
        servo_kit_.set_angle(0, pan_);   // Pan servo
        servo_kit_.set_angle(1, tilt_);  // Tilt servo
    }

    void trackObject(const cv::Rect& object_bbox, const cv::Size& frame_size) {
        // Calculate object center
        int c_x = object_bbox.x + object_bbox.width / 2;
        int c_y = object_bbox.y + object_bbox.height / 2;

        // Calculate error from frame center
        int error_pan = frame_size.width / 2 - c_x;
        int error_tilt = frame_size.height / 2 - c_y;

        // Adjust pan and tilt with a simple proportional control
        if (std::abs(error_pan) > 15) {
            pan_ += error_pan / 75;
        }
        if (std::abs(error_tilt) > 15) {
            tilt_ -= error_tilt / 75;
        }

        // Constrain servo angles
        pan_ = std::max(0, std::min(180, pan_));
        tilt_ = std::max(0, std::min(180, tilt_));

        // Update servo positions
        servo_kit_.set_angle(0, pan_);
        servo_kit_.set_angle(1, tilt_);
    }

private:
    ServoKit& servo_kit_;
    int pan_;
    int tilt_;
};

class ObjectSelector {
public:
    ObjectSelector(const std::string& window_name) : 
        window_name_(window_name), 
        selection_made_(false) {}

    void setupMouseCallback() {
        cv::setMouseCallback(window_name_, mouseCallback, this);
    }

    bool hasSelection() const { return selection_made_; }
    cv::Rect getSelectedROI() const { return selected_roi_; }
    void reset() { selection_made_ = false; }

private:
    static void mouseCallback(int event, int x, int y, int flags, void* userdata) {
        ObjectSelector* selector = reinterpret_cast<ObjectSelector*>(userdata);
        
        if (event == cv::EVENT_LBUTTONDOWN) {
            selector->start_point_ = cv::Point(x, y);
            selector->selection_made_ = false;
        }
        
        if (event == cv::EVENT_LBUTTONUP) {
            selector->end_point_ = cv::Point(x, y);
            selector->selected_roi_ = cv::Rect(
                std::min(selector->start_point_.x, selector->end_point_.x),
                std::min(selector->start_point_.y, selector->end_point_.y),
                std::abs(selector->end_point_.x - selector->start_point_.x),
                std::abs(selector->end_point_.y - selector->start_point_.y)
            );
            selector->selection_made_ = true;
        }
    }

    std::string window_name_;
    cv::Point start_point_;
    cv::Point end_point_;
    cv::Rect selected_roi_;
    bool selection_made_;
};

class TrackingSystem {
public:
    TrackingSystem(const TrackingConfig& config) : 
        config_(config),
        servo_kit_(16),
        servo_tracker_(servo_kit_),
        object_selector_("Tracking"),
        yolo_(config.engine_path()) {
        
        // Open video capture
        if (config_.source() == "0") {
            cap_.open(0);
        } else {
            cap_.open(config_.source());
        }
        
        // Set capture resolution
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, config_.img_size().width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.img_size().height);

        // Create window and setup mouse callback
        cv::namedWindow("Tracking");
        object_selector_.setupMouseCallback();
    }

    void run() {
        cv::Mat frame;
        cv::Rect tracking_roi;
        bool is_tracking = false;
        int selected_class = -1;

        while (cap_.read(frame)) {
            // Detect objects using YOLOv8
            std::vector<std::vector<float>> detections = yolo_.detect(frame, config_.conf_threshold());
            
            // Check for user ROI selection
            if (object_selector_.hasSelection()) {
                tracking_roi = object_selector_.getSelectedROI();
                is_tracking = true;
                object_selector_.reset();
            }

            // Draw and process detections
            for (const auto& detection : detections) {
                cv::Rect bbox(
                    static_cast<int>(detection[0]), 
                    static_cast<int>(detection[1]), 
                    static_cast<int>(detection[2] - detection[0]), 
                    static_cast<int>(detection[3] - detection[1])
                );
                int cls = static_cast<int>(detection[5]);
                float conf = detection[4];

                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, 
                    "Class: " + std::to_string(cls) + 
                    " Conf: " + std::to_string(conf), 
                    cv::Point(bbox.x, bbox.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(0, 255, 0), 2
                );

                // Optional: Check if this detection matches user-selected ROI
                if (tracking_roi.contains(bbox.tl()) && tracking_roi.contains(bbox.br())) {
                    is_tracking = true;
                    selected_class = cls;
                }
            }

            // If tracking is active
            if (is_tracking && !detections.empty()) {
                // Find the best matching detection for tracking
                auto best_detection = std::min_element(
                    detections.begin(), 
                    detections.end(),
                    [&tracking_roi](const std::vector<float>& a, const std::vector<float>& b) {
                        cv::Rect bbox_a(
                            static_cast<int>(a[0]), 
                            static_cast<int>(a[1]), 
                            static_cast<int>(a[2] - a[0]), 
                            static_cast<int>(a[3] - a[1])
                        );
                        cv::Rect bbox_b(
                            static_cast<int>(b[0]), 
                            static_cast<int>(b[1]), 
                            static_cast<int>(b[2] - b[0]), 
                            static_cast<int>(b[3] - b[1])
                        );
                        return cv::norm(bbox_a.tl() - tracking_roi.tl()) < cv::norm(bbox_b.tl() - tracking_roi.tl());
                    }
                );

                if (best_detection != detections.end()) {
                    cv::Rect current_roi(
                        static_cast<int>((*best_detection)[0]), 
                        static_cast<int>((*best_detection)[1]), 
                        static_cast<int>((*best_detection)[2] - (*best_detection)[0]), 
                        static_cast<int>((*best_detection)[3] - (*best_detection)[1])
                    );

                    // Servo tracking
                    servo_tracker_.trackObject(current_roi, frame.size());
                }
            }

            // Display frame
            cv::imshow("Tracking", frame);

            // Exit on 'q' key
            if (cv::waitKey(1) == 'q') break;
        }
    }

private:
    TrackingConfig config_;
    cv::VideoCapture cap_;
    ServoKit servo_kit_;
    ServoTracker servo_tracker_;
    ObjectSelector object_selector_;
    YOLOv8TRT yolo_;
};

int main() {
    try {
        // Configure tracking system
        TrackingConfig config(
            "0",           // source
            {0},           // classes
            0.25,          // confidence threshold
            "0",           // device
            true,          // save video
            true,          // show video
            cv::Size(640, 480), // image size
            "yolov8n.engine" // TensorRT engine path
        );

        // Run tracking system
        TrackingSystem tracking_system(config);
        tracking_system.run();
    } 
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}