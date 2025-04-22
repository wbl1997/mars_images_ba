#ifndef _AETRI_H_
#define _AETRI_H_

#include "EO.h"

#include <fstream>
#include <time.h>
#include <iostream>
#include <Eigen\Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "mygdal.h"

using namespace cv;
using namespace std;
using namespace Eigen;


//输入立体像对进行特征点提取并进行匹配（暂定Harrise角点和灰度匹配）
void GetFeaturePoint();
int Feature_Detection(char* imagepath,int ch,int thresh,std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y);

//#include "../src/AeTri.cpp"
#include "../src/prePhotogrammetry.cpp"
#endif 