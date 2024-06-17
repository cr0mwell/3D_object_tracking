#include <cmath>
#include <limits>
#include <list>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;


enum OptionType {Detector, Descriptor};
enum class Detectors {AKAZE=1, BRISK, FAST, Harris, ORB, ShiTomasi, SIFT, SURF};
enum class Descriptors {AKAZE, BRIEF, BRISK, FREAK, ORB, SIFT, SURF};

template<typename T>
void processInput(T &option) {
    cin >> option;
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

void processInput(string &option) {
    cin >> option;
    transform(option.begin(), option.end(), option.begin(), [](unsigned char c){ return tolower(c);});
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}


void printDetectorOptions() {
    cout << "* the detector type (1-7):" << endl;
    cout << "1. AKAZE" << endl;
    cout << "2. BRISK" << endl;
    cout << "3. FAST" << endl;
    cout << "4. Harris" << endl;
    cout << "5. ORB" << endl;
    cout << "6. ShiTomasi" << endl;
    cout << "7. SIFT" << endl;
    cout << "8. SURF" << endl;
}


void printDescriptorOptions() {
    cout << "* the descriptor type (1-5):" << endl;
    cout << "1. BRIEF" << endl;
    cout << "2. BRISK" << endl;
    cout << "3. FREAK" << endl;
    cout << "4. ORB" << endl;
    cout << "5. SIFT" << endl;
    cout << "6. SURF" << endl;
}


bool getBoolOption(string message) {
    char drawOption;
    cout << message;
    processInput(drawOption);
    
    while (tolower(drawOption) != 'y' && tolower(drawOption) != 'n') {
        cout << "\nInvalid value entered (enter 'y' or 'n') ";
        processInput(drawOption);
    }
    
    return (drawOption == 'y');
}


int getIntOption(string message, int limit=10000) {
    int option;
    cout << message << limit << "]: ";
    processInput(option);
    
    while (!isdigit(option) && (option == 0 || option > limit)) {
        cout << "\nInvalid value entered. " << message << limit << "]: ";
        processInput(option);
    }
    
    return option;
}


int getEnumOption(int optionType) {
    int option;
    vector<int> validOptions;
    void (*printFunc)() = NULL;
    
    if (optionType == Detector) {
        validOptions = {1, 2, 3, 4, 5, 6, 7, 8};
        printFunc = &printDetectorOptions;
    } else {
        validOptions = {1, 2, 3, 4, 5, 6};
        printFunc = &printDescriptorOptions;
    }
    
    printFunc();
    processInput(option);
    
    while (find(validOptions.begin(), validOptions.end(), option) == validOptions.end()) {
        cout << "\nInvalid value entered" << endl;
        printFunc();
        processInput(option);
    }
    
    return option;
}


int main(int argc, char** argv)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "media/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // Calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // Misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    list<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;   // visualize results
    cout << fixed << setprecision(2);
    
    /* SET RUNTIME VARIABLES */
    
    Detectors detectorType = static_cast<Detectors>(getEnumOption(Detector));
    Descriptors descriptorType;
    if (detectorType != Detectors::AKAZE)
        descriptorType = static_cast<Descriptors>(getEnumOption(Descriptor));
    else
        // AKAZE descriptor works with AKAZE Keypoint format only
        descriptorType = Descriptors::AKAZE;
    
    // Limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = getBoolOption("* set the limit for the key points detection? (y/n): ");
    int maxKeypoints;
    if (bLimitKpts)
        maxKeypoints = getIntOption("* max key points number to process? [0-");
    
    // Use BF or FLANN-based matching?
    string matcherType = getBoolOption("* use brute force (y) or FLANN-based matching (n)?: ") ? "MAT_BF" : "MAT_FLANN";
    
    // Use crosscheck for BF matching? (For HOG descriptors only)
    bool crossCheck {false};
    if (matcherType.compare("MAT_BF") == 0)
        crossCheck = getBoolOption("* use crosscheck method for the brute force matching? (y/n): ");
    
    string selectorType = "SEL_NN";
    int k {2};
    // No way of using KNN if crosscheck was chosen as an option
    if (!crossCheck)
    {
        selectorType = getBoolOption("* use NN matching filtering algorithm (y) or KNN (n)?: ") ? "SEL_NN" : "SEL_KNN";
        if (selectorType.compare("SEL_KNN") == 0)
            k = getIntOption("* max number of nearest neighbours to process for each keypoint matching (default 2)? [1-", 5);
    }

    /* MAIN LOOP OVER ALL IMAGES */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */

        // Assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // Load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);
        
        // To optimize memory consumption, keep only the last <dataBufferSize> frames in the buffer
        if (dataBuffer.size() > dataBufferSize)
            dataBuffer.pop_front();

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.4;
        float nmsThreshold = 0.4;
        detectObjects(next(dataBuffer.end(), -1)->cameraImg, next(dataBuffer.end(), -1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

        /* CROP LIDAR POINTS */

        // Load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // Remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        next(dataBuffer.end(), -1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // Associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI(next(dataBuffer.end(), -1)->boundingBoxes, next(dataBuffer.end(), -1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        show3DObjects(next(dataBuffer.end(), -1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000));
        
        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        
        /* DETECT IMAGE KEYPOINTS */

        // Convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor(next(dataBuffer.end(), -1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // Extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        
        switch (detectorType)
        {
            case Detectors::AKAZE:
                detKeypointsModern<cv::AKAZE>(keypoints, imgGray, bVis);
                break;
            case Detectors::BRISK:
                detKeypointsModern<cv::BRISK>(keypoints, imgGray, bVis);
                break;
            case Detectors::FAST:
                detKeypointsFast(keypoints, imgGray, bVis);
                break;
            case Detectors::Harris:
                detKeypointsHarris(keypoints, imgGray, bVis);
                break;
            case Detectors::ORB:
                detKeypointsModern<cv::ORB>(keypoints, imgGray, bVis);
                break;
            case Detectors::ShiTomasi:
                detKeypointsShiTomasi(keypoints, imgGray, bVis);
                break;
            case Detectors::SIFT:
                detKeypointsModern<cv::SIFT>(keypoints, imgGray, bVis);
                break;
            case Detectors::SURF:
                detKeypointsModern<cv::xfeatures2d::SURF>(keypoints, imgGray, bVis);
                break;
        }

        // Limit number of keypoints (helpful for debugging and learning)
        if (bLimitKpts)
        {
            if (detectorType == Detectors::ShiTomasi)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << "Note: keypoints size have been limited to " << maxKeypoints << endl;
        }

        // Push keypoints and descriptor for current frame to end of data buffer
        next(dataBuffer.end(), -1)->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        switch (descriptorType)
        {
            case Descriptors::AKAZE:
                descKeypoints<cv::AKAZE>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "AKAZE");
                break;
            case Descriptors::BRIEF:
                descKeypoints<cv::xfeatures2d::BriefDescriptorExtractor>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "BRIEF");
                break;
            case Descriptors::BRISK:
                descKeypoints<cv::BRISK>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "BRISK");
                break;
            case Descriptors::FREAK:
                descKeypoints<cv::xfeatures2d::FREAK>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "FREAK");
                break;
            case Descriptors::ORB:
                descKeypoints<cv::ORB>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "ORB");
                break;
            case Descriptors::SIFT:
                descKeypoints<cv::SIFT>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "SIFT");
                break;
            case Descriptors::SURF:
                descKeypoints<cv::xfeatures2d::SURF>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "SURF");
                break;
        }
        
        // Push descriptors for current frame to end of data buffer
        next(dataBuffer.end(), -1)->descriptors = descriptors;

        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string descriptorStructType = (descriptorType == Descriptors::SIFT || descriptorType == Descriptors::SURF) ? "DES_HOG" : "DES_BINARY";
            
            matchDescriptors((next(dataBuffer.end(), -2))->keypoints, (next(dataBuffer.end(), -1))->keypoints,
                             (next(dataBuffer.end(), -2))->descriptors, (next(dataBuffer.end(), -1))->descriptors,
                             matches, descriptorStructType, matcherType, selectorType, crossCheck, k);
            
            // store matches in current data frame
            (next(dataBuffer.end(), -1))->kptMatches = matches;
            
            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            // Associate bounding boxes between current and previous frame using keypoint matches
            map<int, int> bbBestMatches;
            matchBoundingBoxes(bbBestMatches, *next(dataBuffer.end(), -2), *next(dataBuffer.end(), -1));

            // store matches in current data frame
            next(dataBuffer.end(), -1)->bbMatches = bbBestMatches;
            
            /* COMPUTE TTC ON OBJECT IN FRONT */

            // Loop over all BB match pairs
            for (auto it1 = next(dataBuffer.end(), -1)->bbMatches.begin(); it1 != next(dataBuffer.end(), -1)->bbMatches.end(); ++it1)
            {
                // Find bounding boxes associates with current match
                BoundingBox *prevBB = nullptr, *currBB = nullptr;
                for (auto it2 = next(dataBuffer.end(), -1)->boundingBoxes.begin(); it2 != next(dataBuffer.end(), -1)->boundingBoxes.end(); ++it2)
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                        currBB = &(*it2);
                
                for (auto it2 = next(dataBuffer.end(), -2)->boundingBoxes.begin(); it2 != next(dataBuffer.end(), -2)->boundingBoxes.end(); ++it2)
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                        prevBB = &(*it2);
                
                // Compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    // Compute TTC based on Lidar points
                    double ttcLidar;
                    computeTTCLidar(prevBB, currBB, sensorFrameRate, ttcLidar);
                    
                    // Compute time-to-collision based on camera
                    double ttcCamera;
                    // Assign enclosed keypoint matches to bounding box
                    clusterKptMatchesWithROI(*currBB, next(dataBuffer.end(), -2)->keypoints, next(dataBuffer.end(), -1)->keypoints, next(dataBuffer.end(), -1)->kptMatches);
                    cv::Mat visImg = next(dataBuffer.end(), -1)->cameraImg.clone();
                    computeTTCCamera(next(dataBuffer.end(), -2)->keypoints, next(dataBuffer.end(), -1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera, visImg);
                    showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT);
                    cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                    
                    char str[200];
                    visImg(cv::Rect(290, 25, 750, 30)).setTo(cv::Scalar(255, 255, 255));
                    sprintf(str, "TTC Lidar : %2.2f s, TTC Camera : %2.2f s", ttcLidar, ttcCamera);
                    putText(visImg, str, cv::Point2f(300, 50), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0,0,255));

                    string windowName = "Final Results : TTC";
                    cv::namedWindow(windowName, 4);
                    cv::imshow(windowName, visImg);
                    cout << "Press key to continue to next frame" << endl;
                    cv::waitKey(0);
                } // eof TTC computation
            } // eof loop over all BB matches            
        }
    } // eof loop over all images
    
    return 0;
}
