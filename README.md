# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.


## [Rubric](https://review.udacity.com/#!/rubrics/2549/view) Points

### 1. Data Buffer

#### MP.1 Data Buffer Optimization
* Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). 
* This can be achieved by pushing in new elements on one end and removing elements on the other end.

* Solution in `MidTermProject_Camera_Student.cpp` lines (132 - 139):

```c++
// push image into data frame buffer
DataFrame frame;
frame.cameraImg = imgGray;
//dataBuffer.push_back(frame);
if(dataBuffer.size()>= dataBufferSize)
{
  dataBuffer.erase(dataBuffer.begin()); 
}
  dataBuffer.push_back(frame);
```

### 2. Keypoints

#### MP.2 Keypoint Detection
* Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.
* solution in `MidTermProject_Camera_Student.cpp` lines(157-168)
```c++
if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST")  == 0 || detectorType.compare("BRISK") == 0 || detectorType.compare("ORB")   == 0 || detectorType.compare("AKAZE") == 0 ||detectorType.compare("SIFT")  == 0)
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
```
* solution in `matching2D_Student.cpp` lines(191-354)

```c++ 
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints

    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
  cv::Ptr<cv::FeatureDetector> detector;
  double t = (double)cv::getTickCount();
  
  if(detectorType == "FAST")
  {
    int threshold = 30;
    bool nonmaxSuppression = true;
     cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
      
    detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
    detector->detect(img, keypoints);
    
  }
    else if (detectorType == "BRISK")
  {
    int threshold = 30;        //   AGAST detection threshold score
    int octaves = 3;           // detection octaves
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint
    detector = cv::BRISK::create(threshold, octaves, patternScale);
    detector->detect(img, keypoints);
  }
  else if (detectorType == "ORB")
  {
    int   nfeatures = 500;     // The maximum number of features to retain.
    float scaleFactor = 1.2f;  // Pyramid decimation ratio, greater than 1.
    int   nlevels = 8;         // The number of pyramid levels.
    int   edgeThreshold = 31;  // This is size of the border where the features are not detected.
    int   firstLevel = 0;      // The level of pyramid to put source image to.
    int   WTA_K = 2;           // The number of points that produce each element of the oriented BRIEF descriptor.
    auto  scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE / FAST_SCORE algorithm is used to rank features.
    int   patchSize = 31;      // Size of the patch used by the oriented BRIEF descriptor.
    int   fastThreshold = 20;  // The FAST threshold.
    detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                               firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    detector->detect(img, keypoints);
  }
  else if (detectorType == "AKAZE")
  {
    // Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT,
    //                                   DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
    auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int   descriptor_size = 0;        // Size of the descriptor in bits. 0 -> Full size
    int   descriptor_channels = 3;    // Number of channels in the descriptor (1, 2, 3).
    float threshold = 0.001f;         //   Detector response threshold to accept point.
    int   nOctaves = 4;               // Maximum octave evolution of the image.
    int   nOctaveLayers = 4;          // Default number of sublevels per scale level.
    auto  diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2,
    //                   DIFF_WEICKERT or DIFF_CHARBONNIER.
    detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                 threshold, nOctaves, nOctaveLayers, diffusivity);
    detector->detect(img, keypoints);
  }
  else if (detectorType == "SIFT")
  {
    std::cout << "STARTING SIFT detector" << std::endl;

    int nfeatures = 0; // The number of best features to retain.
    int nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper.
    // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
    double contrastThreshold = 0.04;
    double edgeThreshold = 10; // The threshold used to filter out edge-like features.
    double sigma = 1.6; // The sigma of the Gaussian applied to the input image at the octave \#0.

    detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    detector->detect(img, keypoints);
    
  }
  
  
  
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
    // plot
  if (bVis)
  {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string windowName = detectorType + " Keypoint Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
  }

}
```

#### MP.3 Keypoint Removal
* Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.
* solution in `MidTermProject_Camera_Student.cpp` line(173-201)

```c++
        vector<cv::KeyPoint>::iterator keypoint;
        vector<cv::KeyPoint> keypoints_roi;
      
        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
           for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
                 {
                     if (vehicleRect.contains(keypoint->pt))
                     {  
                          cv::KeyPoint newKeyPoint;
                          newKeyPoint.pt = cv::Point2f(keypoint->pt);
                          newKeyPoint.size = 1;
                          keypoints_roi.push_back(newKeyPoint);
                     }
                  }

                    keypoints =  keypoints_roi;
                    cout << "IN ROI n= " << keypoints.size()<<" keypoints"<<endl;
                    
        }
      
      
        if(!write_detector)
        {
           detector_file  << ", " << keypoints.size();
        }  
```

### 3. Descriptors

#### MP.4 Keypoint Descriptors
* Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.
* solution in `MidTermProject_Camera_Student.cpp` line(249-257)

```c++
string descriptorType;
if (descriptorType.compare("SIFT") == 0) 
{
   descriptorType == "DES_HOG";
}
else
{
   descriptorType == "DES_BINARY";
}          
```
* solution in `matching2D_Student.cpp` lines(82-139)

```c++
    else if(descriptorType.compare("AKAZE") == 0)
    {

        auto descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptor_size = 0;
        int descriptor_channels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        auto diffusivity = cv::KAZE::DIFF_PM_G2; 
      
      
        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    }
  else if(descriptorType.compare("ORB") == 0)
    {
       int nfeatures = 500;
       float scaleFactor = 1.2f;
       int nlevels = 8;
       int edgeThreshold = 31;
       int firstLevel = 0;
       int WTA_K = 2;
       auto scoreType = cv::ORB::HARRIS_SCORE;
       int patchSize = 31;
       int fastThreshold = 20;  
    
       extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    
    }
  else if(descriptorType.compare("SIFT") == 0)
    {
       int nfeatures = 0;
       int nOctaveLayers = 3;
       double contrastThreshold = 0.04;
       double edgeThreshold = 10;
       double sigma = 1.6;
         
       extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);  
         
    }
  else if(descriptorType.compare("FREAK") == 0)
    {
      bool	orientationNormalized = true;  // Enable orientation normalization.
      bool	scaleNormalized = true;        // Enable scale normalization.
      float patternScale = 22.0f;         // Scaling of the description pattern.
      int	nOctaves = 4;                     // Number of octaves covered by the detected keypoints.
      const std::vector< int > &selectedPairs = std::vector< int >(); 
        
       extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs);
    }
    else if(descriptorType.compare("BRIEF") == 0) {

        
        int bytes = 32;
        bool use_orientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create	(bytes, use_orientation);
    }
```

#### MP.5 Descriptor Matching
* Implement FLANN matching as well as k-nearest neighbor selection. 
* Both methods must be selectable using the respective strings in the main function.
* solution in `matching2D_Student.cpp` line(11-31)

```c++
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {

        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        
    }
```

#### MP.6 Descriptor Distance Ratio
* Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.
* solution in `MidTermProject_Camera_Student.cpp`
```c++
string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
```

* solution in `matching2D_Student.cpp` line(42-63)

```c++
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        //double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
        //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
      
      
        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        } 
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }   
```

### 4. Performance
---
#### MP.7 Performance Evaluation 
* Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. 
* Do this for all the detectors you have implemented.
* solution for data in file `TASK MP.7_Counts_Keypoints.csv` and  in rubric [**Results**](###Keypoint-detected)  below.

#### MP.8 Performance Evaluation
* Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. 
* In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
* solution for data in file  `TASK MP.8_Counts_matched_Keypoints.csv` and  in rubric [**Results**](###matched-keypoints) below.

#### MP.9 Performance Evaluation 
* Log the time it takes for keypoint detection and descriptor extraction. 
* The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.
* solution for data in file  `TASK MP.9_Time_Keypoints.csv` and  in rubric [**Results**](#Time-taken) below.



## Results

After implementation of keypoint-detection, descriptor extraction and keypoint-matching in consecutive images, the results obtained are presented in the following sections. Here are some notes on the results:

* AKAZE detector works only with AKAZE descriptor
* For matching with SIFT descriptor we need to use FLANN. SIFT does not work with BF
* SIFT detector and ORB descriptor do not work together

#### Keypoint-detected

![number_keypoints](https://user-images.githubusercontent.com/34095574/87911422-e45f8c80-ca6b-11ea-8dd0-255523dadbd8.jpg)

#### matched-keypoints

A mean of the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors is shown here.

![Microsoft Excel utilisation non commerciale - TASK MP 8_Counts_matched_Keypoints (4) csv](https://user-images.githubusercontent.com/34095574/87918566-e8dd7280-ca76-11ea-8959-3248b0a07e10.jpg)



![Microsoft Excel utilisation non commerciale - TASK MP 8_Counts_matched_Keypoints (4) csv_2](https://user-images.githubusercontent.com/34095574/87918650-04487d80-ca77-11ea-930b-5afc0d55643e.jpg)



#### Time-taken


![SFND_P3_2D_Feature_TrackingWriteup md at master · studianSFND_P3_2D_Feature_Tracking - Google Chrome](https://user-images.githubusercontent.com/34095574/87935332-2fd86180-ca91-11ea-8b01-7ca9eaea20f4.jpg)

The mean of time taken for keypoint detection and descriptor extraction for all combinations is listed below.

![Microsoft Excel utilisation non commerciale - TASK MP 9_Time_Keypoints (4) 
 csv_2](https://user-images.githubusercontent.com/34095574/87936816-d3c30c80-ca93-11ea-88dd-31a1317468e1.jpg)


## Recommendation

Based on these scores my recommendations are:

| rank | detector | descriptor |Average Time
|:------:|:------:|:------:| --------
| 1 | FAST | BRIEF |11.38 ms
| 2 | FAST | ORB |12.37 ms
| 3 | ORB | BRIEF |16.92 ms
