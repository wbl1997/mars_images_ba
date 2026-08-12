#ifndef _MYGDAL_H_
#define _MYGDAL_H_

#include <fstream>
#include <iostream>

#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"
#include "gdal_alg.h"

typedef unsigned char uchar; 

bool WriteImageData(const char* strDestFilePath,unsigned char* pImageData,int nWidth,int nHeight,int nChannels,int nNewChannels);
void jichu();
void Linear2(char* pszSrcFile, char* pszDstFile);
void Linear(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
void LoadData(char* data_fname,int rows,int cols, int dsr_size, int PY, int Initial_JD, int Center_row, int Center_col, int ImageR, int ImageC, uchar *ImageData);
void Col_balance(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
void Col_balance1(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
void Col_balance2(char *img_path,char* cb_path);
void pds2tif(char* pszFile, char* dstPath,int ImageR,int ImageC);
void pds2tif2();
void mosaic1(char* pszSrcFile1, char* pszSrcFile2, char* pszDstFile,int overlap);
int Down_sample(char* pszSrcFile1, int batchsize, char* pszDstFile);
int Hijitreg_gdal(char* imagepathL, char* imagepathR, int OverlapSamples, int* dr, int* dc);
int CCDmosaic_gdal(char* imagepathL, char* imagepathR, int OverlapSamples, char* outpath);
void My_rec2NEH(const char *projRef,double X,double Y,double Z,double* N,double* E,double* H);
void My_NEH2rec(const char *projRef,double N,double E,double H,double* X,double* Y,double* Z);
void LoadDEM(char* imagepath, float X, float Y, float* Z);
void xulie_process();
void xulie_process1(char* filepath,char* outfilepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2);
void tif_load(char* imagepath1,uchar* data1);
void ds_mosaic();
void gdal_test();


//#include "../src/mygdal.cpp"
#endif 