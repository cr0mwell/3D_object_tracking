#include <map>

#include "matching2D.hpp"

using namespace std;


// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      vector<cv::DMatch> &matches, string descriptorType, string matcherType, string selectorType, bool crossCheck, int k)
{
    // configure matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType == "DES_HOG" ? cv::NORM_L2 : cv::NORM_HAMMING;
        string normTypeStr = normType == cv::NORM_L2 ? "L2 norm. " : "Hamming norm. ";
        normTypeStr += crossCheck ? "Cross-check ENABLED." : "Cross-check DISABLED.";
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "Used brute force matching with " << normTypeStr << endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
        cout << "Used FLANN-based matching." << endl;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    {
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches);
        cout << "Deployed NN filtering. Number of matched keypoints: " << matches.size() << endl;
    }

    else if (selectorType.compare("SEL_KNN") == 0)
    {
        // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        // Filter matches using descriptor distance ratio test
        float minDescDistRatio = 0.8;
        for (auto &m: knn_matches)
            if (m[0].distance < minDescDistRatio * m[1].distance)
                matches.push_back(m[0]);
        
        cout << "Deployed KNN filtering (K=" << k << "). Number of matched keypoints: " << matches.size() << endl;
    }
}


// Use one of several types of state-of-art descriptors to uniquely identify keypoints
template<typename T>
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor = T::create();
    
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}


template void descKeypoints<cv::AKAZE>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType);
template void descKeypoints<cv::xfeatures2d::BriefDescriptorExtractor>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType);
template void descKeypoints<cv::BRISK>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType);
template void descKeypoints<cv::xfeatures2d::FREAK>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType);
template void descKeypoints<cv::ORB>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType);
template void descKeypoints<cv::SIFT>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType);
template void descKeypoints<cv::xfeatures2d::SURF>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType);


void visualize(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorName, int windowNumber = 1)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = detectorName + " Corner Detector Results";
    cv::namedWindow(windowName, windowNumber);
    imshow(windowName, visImage);
    cv::waitKey(0);
}


// Shi-Thomasi detector implementation
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 4;  //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
        visualize(keypoints, img, "ShiTomasi");
}


// Harris corner detector implementation
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    int intensity_threshold {100};

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    
    // Locating local maximums in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates will be stored in vector<cv::KeyPoint> structure.
    for (int i=0; i<dst_norm_scaled.rows; i++) 
        for (int j=0; j<dst_norm_scaled.cols; j++) {
            int intensity = dst_norm_scaled.at<unsigned char>(i, j);
            
            if (intensity > intensity_threshold)  {
                cv::KeyPoint new_keypoint(j, i, 2*apertureSize, intensity);
                
                bool overlaps = false;
                for (auto &keypoint: keypoints) {
                    double overlapping_area = cv::KeyPoint::overlap(new_keypoint, keypoint);
                    
                    if (overlapping_area) {
                        overlaps = true;
                        if (new_keypoint.response > keypoint.response) {
                            keypoint = new_keypoint;
                            break;
                        }
                    }
                }
                
                if (!overlaps)
                    keypoints.push_back(new_keypoint); 
            }
        }
    
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
        visualize(keypoints, img, "Harris");
}


// FAST detector implementation
void detKeypointsFast(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(20, true, type);
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
        visualize(keypoints, img, "FAST");
}


// Function to create detectors that have cv::Feature2D as a base class and thus a common interface
template<typename T>
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    map<string, string> detectorNames {{"N2cv5AKAZEE", "AKAZE"},
                                       {"N2cv5BRISKE", "BRISK"},
                                       {"N2cv3ORBE", "ORB"},
                                       {"N2cv4SIFTE", "SIFT"},
                                       {"N2cv11xfeatures2d4SURFE", "SURF"}};
    string className = typeid(T).name();
    string detectorName = detectorNames.count(className) == 1 ? detectorNames[className] : className;
    
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::FeatureDetector> detector = T::create();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorName << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
        visualize(keypoints, img, detectorName);
}


template void detKeypointsModern<cv::AKAZE>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis);
template void detKeypointsModern<cv::BRISK>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis);
template void detKeypointsModern<cv::ORB>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis);
template void detKeypointsModern<cv::SIFT>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis);
template void detKeypointsModern<cv::xfeatures2d::SURF>(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis);