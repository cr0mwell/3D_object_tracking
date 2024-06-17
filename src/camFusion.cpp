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
    vector<double> xs, ys, zs;
    StatsLidarPoints s;
    
    // Calculating means & medians
    for (const auto &kpt: points)
    {
        meanX += kpt.x;
        meanY += kpt.y;
        meanZ += kpt.z;
        xs.push_back(kpt.x);
        ys.push_back(kpt.y);
        zs.push_back(kpt.z);
    }
    s.meanX = meanX / points.size();
    s.meanY = meanY / points.size();
    s.meanZ = meanZ / points.size();
    
    sort(xs.begin(), xs.end());
    sort(ys.begin(), ys.end());
    sort(zs.begin(), zs.end());
    
    if (xs.size() % 2 == 0)
    {
        s.medianX = (xs[xs.size()/2] + xs[xs.size()/2-1])/2;
        s.medianY = (ys[ys.size()/2] + ys[ys.size()/2-1])/2;
        s.medianZ = (zs[zs.size()/2] + zs[zs.size()/2-1])/2;
    } else {
        s.medianX = xs[xs.size()/2];
        s.medianY = ys[ys.size()/2];
        s.medianZ = zs[zs.size()/2];
    }
    
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


void getKeypointDistanceStats(vector<cv::DMatch> &kptMatches, vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, float &meanDistance, float &medianDistance, float &stdDistance)
{
    vector<float> distances;
    float dist {};
    
    // Calculating distance mean and median
    for (const auto &m: kptMatches)
    {
        dist = cv::norm(kptsCurr[m.trainIdx].pt - kptsPrev[m.queryIdx].pt);
        meanDistance += dist;
        distances.push_back(dist);
    }
    meanDistance /= kptMatches.size();
    
    sort(distances.begin(), distances.end());
    if (distances.size() % 2 == 0)
        medianDistance = (distances[distances.size()/2] + distances[distances.size()/2-1])/2;
    else
        medianDistance = distances[distances.size()/2];
    
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
    float meanDistance {}, medianDistance {}, stdDistance {}, dist {};
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
        medianDistance = 0;
        stdDistance = 0;
        getKeypointDistanceStats(boundingBox.kptMatches, kptsPrev, kptsCurr, meanDistance, medianDistance, stdDistance);
        size_t i = 0;
        while (i < boundingBox.kptMatches.size())
        {   
            currKeyPoint = kptsCurr[boundingBox.kptMatches[i].trainIdx];
            prevKeyPoint = kptsPrev[boundingBox.kptMatches[i].queryIdx];
            dist = cv::norm(currKeyPoint.pt - prevKeyPoint.pt);
            if (dist > meanDistance + 3*stdDistance || dist < meanDistance - 3*stdDistance ||
                dist > medianDistance + 3*stdDistance || dist < medianDistance - 3*stdDistance)
                boundingBox.kptMatches.erase(boundingBox.kptMatches.begin()+i);
            else
                i++;
        }
    } while (prevSize - boundingBox.kptMatches.size() != 0);
    
    cout << "Mean distance for the image KeyPoints: " << meanDistance << ", median: " << medianDistance << ", std: " << stdDistance << endl;
    cout << "Filtered KeyPoint matches: " << kptMatches.size() - boundingBox.kptMatches.size() << endl;
}


// computeTTCCamera function's helper
void processKeyPoints(vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, StatsKeyPoints &keypointStats)
{
    vector<double> xs, ys;
    
    for (const auto &kp: kptsPrev)
    {
        keypointStats.meanX += kp.pt.x;
        keypointStats.meanY += kp.pt.y;
        xs.push_back(kp.pt.x);
        ys.push_back(kp.pt.y);
    }
    keypointStats.meanX = keypointStats.meanX / kptsPrev.size();
    keypointStats.meanY = keypointStats.meanY / kptsPrev.size();
    
    sort(xs.begin(), xs.end());
    sort(ys.begin(), ys.end());
    
    if (xs.size() % 2 == 0)
    {
        keypointStats.medianX = (xs[xs.size()/2] + xs[xs.size()/2-1])/2;
        keypointStats.medianY = (ys[ys.size()/2] + ys[ys.size()/2-1])/2;
    } else {
        keypointStats.medianX = xs[xs.size()/2];
        keypointStats.medianY = ys[ys.size()/2];
    }
    
    // Calculating std for previous keypoints
    for (const auto &kp: kptsPrev)
    {
        keypointStats.stdX += pow((kp.pt.x - keypointStats.meanX), 2);
        keypointStats.stdY += pow((kp.pt.y - keypointStats.meanY), 2);
    }
    keypointStats.stdX = sqrt(keypointStats.stdX / kptsPrev.size());
    keypointStats.stdY = sqrt(keypointStats.stdY / kptsPrev.size());
    
    // Filtering by the median
    size_t i {};
    while (i < kptsPrev.size())
        if (kptsPrev[i].pt.x > keypointStats.medianX + 3*keypointStats.stdX || kptsPrev[i].pt.x < keypointStats.medianX - 3*keypointStats.stdX ||
            kptsPrev[i].pt.y > keypointStats.medianY + 3*keypointStats.stdY || kptsPrev[i].pt.y < keypointStats.medianY - 3*keypointStats.stdY)
        {
            kptsPrev.erase(kptsPrev.begin()+i);
            kptsCurr.erase(kptsCurr.begin()+i);
        } else
            i++;
    
    // Filtering by the mean
    i = 0;
    while (i < kptsPrev.size())
        if (kptsPrev[i].pt.x > keypointStats.meanX + 3*keypointStats.stdX || kptsPrev[i].pt.x < keypointStats.meanX - 3*keypointStats.stdX ||
            kptsPrev[i].pt.y > keypointStats.meanY + 3*keypointStats.stdY || kptsPrev[i].pt.y < keypointStats.meanY - 3*keypointStats.stdY)
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
    StatsKeyPoints prevKeypointStats, currKeypointStats;
    size_t prevSize = prevKeypoints.size();
    
    // Calculating the median, mean and std for previous/current keypoint lists;
    // Filter keypoints (z-score by median and mean using 3*std distance threshold):
    //  -first iterate over the previous keypoints and remove the outliers from previous AND current keypoint lists;
    //  -next do the same iterating over the current keypoints list;
    processKeyPoints(prevKeypoints, currKeypoints, prevKeypointStats);
    processKeyPoints(currKeypoints, prevKeypoints, currKeypointStats);
    
    cout << "Previous frame keypoint meanX: " << prevKeypointStats.meanX << ", meanY: " << prevKeypointStats.meanY <<
            ", medianX: " << prevKeypointStats.medianX << ", medianY: " << prevKeypointStats.medianY <<
            ", stdX: " << prevKeypointStats.stdX << ", stdY: " << prevKeypointStats.stdY << endl;
    cout << "Current frame keypoint meanX: " << currKeypointStats.meanX << ", meanY: " << currKeypointStats.meanY <<
            ", medianX: " << currKeypointStats.medianX << ", medianY: " << currKeypointStats.medianY <<
            ", stdX: " << currKeypointStats.stdX << ", stdY: " << currKeypointStats.stdY << endl;
    cout << "Filtered image keypoint outliers from the previous/current frames: " << prevSize-prevKeypoints.size() << endl;
    
    // 3. Calculating mean of the relative distances ratio distancesCurr/distancesPrev
    double distancesRatioMean {1e-9}, distancePrev {}, distanceCurr {};
    float distanceThreshold {20.0};
    size_t total {};
    //cout << "Distance ratios (in previous and current frames):" << endl;
    for (size_t i=0; i < prevKeypoints.size()-1; i++)
        for (size_t j=i+1; j < prevKeypoints.size(); j++)
        {
            distancePrev = cv::norm(prevKeypoints[i].pt - prevKeypoints[j].pt);
            distanceCurr = cv::norm(currKeypoints[i].pt - currKeypoints[j].pt);
            //cout << distanceCurr << ":" << distancePrev << ", ";
            
            if (distancePrev > 0 && distanceCurr > distanceThreshold)
            {
                distancesRatioMean += distanceCurr/distancePrev;
                total++;
            }
        }
    //cout << endl;
    distancesRatioMean /= total;
    
    // 4. Calculating the TTC
    double dT = 1 / frameRate;
    TTC = -dT/(1 - distancesRatioMean);
    cout << "Camera TTC: " << TTC << " (passed distance for the last " << dT << "s is ~" << distancesRatioMean << "m" << endl;
    
    // 5. (Optional) Updating the image with the matched keypoints
    //cv::drawKeypoints(img, currKeypoints, img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}


void filterLidarPoints(vector<LidarPoint> &lidarPoints, StatsLidarPoints &stats)
{
    stats = getMeanStd(lidarPoints);
    
    // Filter by median
    size_t initialSize, sizeBefore;
    initialSize = lidarPoints.size();
    do {
        sizeBefore = lidarPoints.size();
        lidarPoints.erase(
            remove_if(lidarPoints.begin(),
                      lidarPoints.end(),
                      [stats](LidarPoint p) {
                        return (p.x > stats.medianX + 3*stats.stdX || p.x < stats.meanX - 3*stats.stdX ||
                                p.y > stats.medianY + 3*stats.stdY || p.y < stats.meanY - 3*stats.stdY ||
                                p.z > stats.medianZ + 3*stats.stdZ || p.z < stats.meanZ - 3*stats.stdZ);
                        }),
            lidarPoints.end());
        
        // Re-calculating mean and std for the filtered LidarPoints
        stats = getMeanStd(lidarPoints);
    
    } while (sizeBefore - lidarPoints.size() > 0);
    cout << "Removed LiDAR point outliers by median: " << initialSize - lidarPoints.size() << endl;
    
    // Filter by mean
    initialSize = lidarPoints.size();
    do {
        sizeBefore = lidarPoints.size();
        lidarPoints.erase(
            remove_if(lidarPoints.begin(),
                      lidarPoints.end(),
                      [stats](LidarPoint p) {
                        return (p.x > stats.meanX + 3*stats.stdX || p.x < stats.meanX - 3*stats.stdX ||
                                p.y > stats.meanY + 3*stats.stdY || p.y < stats.meanY - 3*stats.stdY ||
                                p.z > stats.meanZ + 3*stats.stdZ || p.z < stats.meanZ - 3*stats.stdZ);
                        }),
            lidarPoints.end());
        
        // Re-calculating mean and std for the filtered LidarPoints
        stats = getMeanStd(lidarPoints);
    
    } while (sizeBefore - lidarPoints.size() > 0);
    cout << "Removed LiDAR point outliers by mean: " << initialSize - lidarPoints.size() << endl;
}


// Compute time-to-collision (TTC) based on LiDAR point distances in successive images
void computeTTCLidar(BoundingBox* BBPrev, BoundingBox* BBCurr, double frameRate, double &TTC)
{
    vector<LidarPoint> lidarPointsPrev = BBPrev->lidarPoints;
    vector<LidarPoint> lidarPointsCurr = BBCurr->lidarPoints;
    
    // Filtering the outliers if necessary using Z-score method
    unsigned int sizeBefore;
    
    // Filtering lidarPointsPrev
    StatsLidarPoints statsPrev;
    if (!BBPrev->lidarPointsStats.meanX)
    {
        filterLidarPoints(lidarPointsPrev, statsPrev);
        BBPrev->lidarPointsStats = statsPrev;
    }
    
    // Filtering lidarPointsCurr
    StatsLidarPoints statsCurr;
    filterLidarPoints(lidarPointsCurr, statsCurr);
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
    map<int, map<int, int>> bbMatches;
    
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
                    if (bbMatches[bb].count(it1->boxID) > 0)
                        bbMatches[bb][it1->boxID]++;
                    else
                        bbMatches[bb][it1->boxID] = 1;
    }
    
    // Defining the best match for each pair <previousBB>-<currentBB>
    for (auto it=bbMatches.begin(); it != bbMatches.end(); it++)
    {
        unsigned int maxNumber {}, bestMatch {};
        
        for (auto it2=it->second.begin(); it2 != it->second.end(); it2++)
            if (it2->second > maxNumber)
            {
               maxNumber = it2->second;
               bestMatch = it2->first;
            }
        bbBestMatches[it->first] = bestMatch;
    }
}
