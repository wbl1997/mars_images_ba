#ifndef _MYGDAL_H_
#define _MYGDAL_H_

#include <fstream>
#include <iostream>
#include "base.h"

#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"

//2%灰度线性拉伸
void Linear2(char* pszSrcFile, char* pszDstFile);
void Linear(uchar *ImageMat, int rows, int cols, uchar *ImageResult);

//读取影像并转存
bool WriteImageData(char* strDestFilePath,unsigned char* pImageData,int nWidth,int nHeight,int nChannels,int nNewChannels);

//二进制文件读取
void LoadData(char* data_fname,int rows,int cols, int dsr_size, int PY, int Initial_JD, int Center_row, int Center_col, int ImageR, int ImageC, uchar *ImageData);

//列灰度均衡化
void Col_balance(uchar *ImageMat, int rows, int cols, uchar *ImageResult);

//pds格式影像读取及转存
void pds2tif();
void pds2tif2();

//同一CCD影像拼接
void mosaic1(char* pszSrcFile1, char* pszSrcFile2, char* pszDstFile2);


#include "../src/mygdal.cpp"
#endif 