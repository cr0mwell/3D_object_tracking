#include "camFusion.hpp"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(vector<BoundingBox> &boundingBoxes, vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        double shrinkFactor = 0.10;
        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (auto it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }
        } // eof loop over all bounding boxes

        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }
    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));
    
    float minx=1e6, miny=1e6, maxy=0.0;
    int bottom=0, left=1e6, top=1e6, right=0;
    
    // plot Lidar points into image
    for (auto it = boundingBoxes.begin(); it != boundingBoxes.end(); ++it)
    {
        // Creating random number generator
        cv::RNG rng(it->boxID);
        cv::Scalar color = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));
        
        for (auto it2 = it->lidarPoints.begin(); it2 != it->lidarPoints.end(); ++it2) {
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.height / worldSize.height) + imageSize.width / 2;
            
            minx = xw>minx ? minx : xw;
            miny = yw>miny ? miny : yw;
            maxy = yw<maxy ? maxy : yw;
            
            bottom = y<bottom ? bottom : y;
            top = y>top ? top : y;
            left = x>left ? left : x;
            right = x<right ? right : x;
            
            cv::circle(topviewImg, cv::Point(x, y), 4, color, -1);
        }
        
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        char str1[200], str2[200];
        sprintf(str1, "id=%id, #pts=%d", it->boxID, static_cast<int>(it->lidarPoints.size()));
        putText(topviewImg, str1, cv::Point(left-100, bottom+50), cv::FONT_ITALIC, 1, color);
        sprintf(str2, "closest_point=%2.2f m, width=%2.2f m", minx, maxy-miny);
        putText(topviewImg, str2, cv::Point(left-100, bottom+100), cv::FONT_ITALIC, 1, color);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0); // wait for key to be pressed
}


StatsLidarPoints getMeanStd(vector<LidarPoint> &points)
{
    double meanX {}, meanY {}, meanZ {}, stdX {}, stdY {}, stdZ {};
    StatsLidarPoints s;
    
    // Calculating means
    for (const auto &kpt: points)
    {
        meanX += kpt.x;
        meanY += kpt.y;
        meanZ += kpt.z;
    }
    s.meanX = meanX / points.size();
    s.meanY = meanY / points.size();
    s.meanZ = meanZ / points.size();
    
    // Calculating stds
    for (const auto &kpt: points)
    {
        stdX += pow((kpt.x - s.meanX), 2);
        stdY += pow((kpt.y - s.meanY), 2);
        stdZ += pow((kpt.z - s.meanZ), 2);
    }
    s.stdX = sqrt(stdX / points.size());
    s.stdY = sqrt(stdY / points.size());
    s.stdZ = sqrt(stdZ / points.size());
    
    return s;
}


void getKeypointDistanceStats(vector<cv::DMatch> &kptMatches, vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, float &meanDistance, float &stdDistance)
{
    // Calculating distance mean
    for (const auto &m: kptMatches)
        meanDistance += cv::norm(kptsCurr[m.trainIdx].pt - kptsPrev[m.queryIdx].pt);
    meanDistance /= kptMatches.size();
    meanDistance = meanDistance / kptMatches.size();
    
    // Calculating distance std
    for (const auto &m: kptMatches)
        stdDistance += pow(cv::norm(kptsCurr[m.trainIdx].pt - kptsPrev[m.queryIdx].pt) - meanDistance, 2);
    stdDistance = sqrt(stdDistance / kptMatches.size());
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, vector<cv::DMatch> &kptMatches)
{
    // Using z-score outliers filtering
    vector<cv::DMatch*> candidateMatches;
    float meanDistance {}, stdDistance {};
    int prevSize {};
    cv::KeyPoint currKeyPoint, prevKeyPoint;
    
    // Collecting matches within the BB
    for (auto it=kptMatches.begin(); it != kptMatches.end(); it++)
    {
        currKeyPoint = kptsCurr[it->trainIdx];
        prevKeyPoint = kptsPrev[it->queryIdx];
        if (boundingBox.roi.contains(currKeyPoint.pt))
            boundingBox.kptMatches.push_back(*it);
    }
    
    // Filtering the keypoints
    do
    {
        prevSize = boundingBox.kptMatches.size();
        meanDistance = 0;
        stdDistance = 0;
        getKeypointDistanceStats(boundingBox.kptMatches, kptsPrev, kptsCurr, meanDistance, stdDistance);
        size_t i = 0;
        while (i < boundingBox.kptMatches.size())
        {   
            currKeyPoint = kptsCurr[boundingBox.kptMatches[i].trainIdx];
            prevKeyPoint = kptsPrev[boundingBox.kptMatches[i].queryIdx];
            if (cv::norm(currKeyPoint.pt - prevKeyPoint.pt) > meanDistance + 3*stdDistance || cv::norm(currKeyPoint.pt - prevKeyPoint.pt) < meanDistance - 3*stdDistance)
                boundingBox.kptMatches.erase(boundingBox.kptMatches.begin()+i);
            else
                i++;
        }
    } while (prevSize - boundingBox.kptMatches.size() != 0);
    
    cout << "Mean distance for the image KeyPoints: " << meanDistance << ", std: " << stdDistance << endl;
    cout << "Filtered KeyPoint matches: " << kptMatches.size() - boundingBox.kptMatches.size() << endl;
}


// computeTTCCamera function's helper
void processKeyPoints(vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, float &meanX, float &meanY, float &stdX, float &stdY)
{
    for (const auto &kp: kptsPrev)
    {
        meanX += kp.pt.x;
        meanY += kp.pt.y;
    }
    meanX = meanX / kptsPrev.size();
    meanY = meanY / kptsPrev.size();
    
    // Calculating std for previous keypoints
    for (const auto &kp: kptsPrev)
    {
        stdX += pow((kp.pt.x - meanX), 2);
        stdY += pow((kp.pt.y - meanY), 2);
    }
    stdX = sqrt(stdX / kptsPrev.size());
    stdY = sqrt(stdY / kptsPrev.size());
    
    size_t i {};
    while (i < kptsPrev.size())
        if (kptsPrev[i].pt.x > meanX + 3*stdX || kptsPrev[i].pt.x < meanX - 3*stdX ||
            kptsPrev[i].pt.y > meanY + 3*stdY || kptsPrev[i].pt.y < meanY - 3*stdY)
        {
            kptsPrev.erase(kptsPrev.begin()+i);
            kptsCurr.erase(kptsCurr.begin()+i);
        } else
            i++;

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, 
                      vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat &img)
{
    // 1. Collecting list of matching keypoints in the current and in the previous frames
    vector<cv::KeyPoint> prevKeypoints, currKeypoints;
    for (auto it=kptMatches.begin(); it != kptMatches.end(); it++)
    {
        prevKeypoints.push_back(kptsPrev[it->queryIdx]);
        currKeypoints.push_back(kptsCurr[it->trainIdx]);
    }
    
    // 2. Filtering the outliers using Z-score approach
    float meanPrevX {}, meanPrevY {}, stdPrevX {}, stdPrevY {},
          meanCurrX {}, meanCurrY {}, stdCurrX {}, stdCurrY {};
    size_t prevSize = prevKeypoints.size();
    
    // Calculating the mean for previous/current keypoint lists;
    // Filter keypoints:
    //  -first iterate over the previous keypoints and remove the outliers from previous AND current keypoint lists;
    //  -next do the same iterating over the current keypoints list;
    processKeyPoints(prevKeypoints, currKeypoints, meanPrevX, meanPrevY, stdPrevX, stdPrevY);
    processKeyPoints(currKeypoints, prevKeypoints, meanCurrX, meanCurrY, stdCurrX, stdCurrY);
    
    cout << "Previous frame keypoint meanX: " << meanPrevX << ", meanY: " << meanPrevY << ", stdX: " << stdPrevX << ", stdY: " << stdPrevY << endl;
    cout << "Current frame keypoint meanX: " << meanCurrX << ", meanY: " << meanCurrY << ", stdX: " << stdCurrX << ", stdY: " << stdCurrY << endl;
    cout << "Filtered image keypoint outliers from the previous/current frames: " << prevSize-prevKeypoints.size() << endl;
    
    // 3. Calculating mean of the relative distances ratio distancesCurr/distancesPrev
    double distancesRatioMean {}, distancePrev {}, distanceCurr {};
    float distanceThreshold {20.0};
    size_t total {};
    for (size_t i=0; i < prevKeypoints.size()-1; i++)
        for (size_t j=i+1; j < prevKeypoints.size(); j++)
        {
            distancePrev = cv::norm(prevKeypoints[i].pt - prevKeypoints[j].pt);
            distanceCurr = cv::norm(currKeypoints[i].pt - currKeypoints[j].pt);
            
            if (distancePrev > 0 && distanceCurr > distanceThreshold)
            {
                distancesRatioMean += distanceCurr/distancePrev;
                total++;
            }
        }
    distancesRatioMean /= total;
    
    // 4. Calculating the TTC
    double dT = 1 / frameRate;
    TTC = -dT/(1 - distancesRatioMean);
    cout << "Camera TTC: " << TTC << " (passed distance for the last " << dT << "s is ~" << distancesRatioMean << "m" << endl;
    
    // 5. (Optional) Updating the image with the matched keypoints
    //cv::drawKeypoints(img, currKeypoints, img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}


// Compute time-to-collision (TTC) based on LiDAR point distances in successive images
void computeTTCLidar(BoundingBox* BBPrev, BoundingBox* BBCurr, double frameRate, double &TTC)
{
    vector<LidarPoint> lidarPointsPrev = BBPrev->lidarPoints;
    vector<LidarPoint> lidarPointsCurr = BBCurr->lidarPoints;
    
    // Filtering the outliers if necessary using Z-score method
    unsigned int sizeBefore;
    
    // Filtering lidarPointsPrev
    if (!BBPrev->lidarPointsStats.meanX)
    {
        // Calculating mean and std for LidarPoints
        StatsLidarPoints statsPrev = getMeanStd(lidarPointsPrev);
        auto filterFunction = [statsPrev](LidarPoint p) {
            return (p.x > statsPrev.meanX + 3*statsPrev.stdX || p.x < statsPrev.meanX - 3*statsPrev.stdX ||
                    p.y > statsPrev.meanY + 3*statsPrev.stdY || p.y < statsPrev.meanY - 3*statsPrev.stdY ||
                    p.z > statsPrev.meanZ + 3*statsPrev.stdZ || p.z < statsPrev.meanZ - 3*statsPrev.stdZ);
            };
        
        do {
            sizeBefore = lidarPointsPrev.size();
            lidarPointsPrev.erase(remove_if(lidarPointsPrev.begin(), lidarPointsPrev.end(), filterFunction), lidarPointsPrev.end());
            
            // Re-calculating mean and std for the filtered LidarPoints
            statsPrev = getMeanStd(lidarPointsPrev);
            cout << "Removed LiDAR point outliers: " << sizeBefore - lidarPointsPrev.size() << endl;
        
        } while (sizeBefore - lidarPointsPrev.size() > 0);
        
        BBPrev->lidarPointsStats = statsPrev;
    }
    
    // Filtering lidarPointsCurr
    StatsLidarPoints statsCurr = getMeanStd(lidarPointsCurr);
    auto filterFunction = [statsCurr](LidarPoint p) {
            return (p.x > statsCurr.meanX + 3*statsCurr.stdX || p.x < statsCurr.meanX - 3*statsCurr.stdX ||
                    p.y > statsCurr.meanY + 3*statsCurr.stdY || p.y < statsCurr.meanY - 3*statsCurr.stdY ||
                    p.z > statsCurr.meanZ + 3*statsCurr.stdZ || p.z < statsCurr.meanZ - 3*statsCurr.stdZ);
            };
    
    do {
        sizeBefore = lidarPointsCurr.size();
        lidarPointsCurr.erase(remove_if(lidarPointsCurr.begin(), lidarPointsCurr.end(), filterFunction), lidarPointsCurr.end());
        
        // Re-calculating mean and std for the filtered LidarPoints
        statsCurr = getMeanStd(lidarPointsCurr);
        cout << "Removed LiDAR point outliers: " << sizeBefore - lidarPointsCurr.size() << endl;
    
    } while (sizeBefore - lidarPointsCurr.size() > 0);
    BBCurr->lidarPointsStats = statsCurr;
    
    // Searching for the closest x and y coordinates in lidarPointsPrev and lidarPointsCurr
    double minXPrev=1e6, minYPrev=1e6, minXCurr=1e6, minYCurr=1e6;
    for (const auto &p: lidarPointsPrev)
    {
        minXPrev = p.x < minXPrev ? p.x : minXPrev;
        minYPrev = (p.y>0 && p.y<minYPrev) ? p.y : minYPrev;
    }
    
    for (const auto &p: lidarPointsCurr)
    {
        minXCurr = p.x < minXCurr ? p.x : minXCurr;
        minYCurr = (p.y>0 && p.y<minYCurr) ? p.y : minYCurr;
    }
    
    // Calculating the TTC
    double dPrev = sqrt(pow(minXPrev, 2) + pow(minYPrev, 2));
    double dCurr = sqrt(pow(minXCurr, 2) + pow(minYCurr, 2));
    cout << "Previous distance to the LiDAR point: " << dPrev << ", current distance to the LiDAR point: " << dCurr << endl;
    
    TTC = dCurr*frameRate/(abs(dPrev-dCurr)*100);
    cout << "Lidar TTC: " << TTC << " (passed distance for the last " << frameRate/100 << "s is "
         << dPrev-dCurr << "m, with const speed " << (dPrev-dCurr)*36 << "km/h" << endl;
}


void matchBoundingBoxes(map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    map<int, vector<int>> bbMatches;
    
    for (auto it=currFrame.kptMatches.begin(); it != currFrame.kptMatches.end(); it++)
    {
        cv::KeyPoint prevKeyPoint = prevFrame.keypoints[it->queryIdx];
        cv::KeyPoint currKeyPoint = currFrame.keypoints[it->trainIdx];
        
        vector<int> prevBBs;
        // Collecting list of BBs from the previous timestamp the KeyPoint belongs to
        for (auto it1=prevFrame.boundingBoxes.begin(); it1 != prevFrame.boundingBoxes.end(); it1++)
            if (it1->roi.contains(prevKeyPoint.pt))
                prevBBs.push_back(it1->boxID);
        
        // Collecting possible matches for the BBs from the previous timestamp
        for (auto it1=currFrame.boundingBoxes.begin(); it1 != currFrame.boundingBoxes.end(); it1++)
            if (it1->roi.contains(currKeyPoint.pt))
                for (const auto &bb: prevBBs)
                    bbMatches[bb].push_back(it1->boxID);
    }
    
    // Defining the best match for each pair <previousBB>-<currentBB>
    for (auto it=bbMatches.begin(); it != bbMatches.end(); it++)
    {
        unsigned int maxNumber {}, bestMatch {}, occurences {};
        
        set<int> uniqueElements(it->second.begin(), it->second.end());
        for (const auto &i: uniqueElements)
        {
            occurences = count(it->second.begin(), it->second.end(), i);
            if (occurences > maxNumber)
            {
                maxNumber = occurences;
                bestMatch = i;
            }
        }
        bbBestMatches[it->first] = bestMatch;
    }
}
