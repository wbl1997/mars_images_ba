#ifndef _IMAGEPROCESS_H_
#define _IMAGEPROCESS_H_

#include <fstream>
#include <iostream>

#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"
#include "gdal_alg.h"

#include "SpiceUsr.h"
#include "PipelineConfig.h"

typedef unsigned char uchar; 
using namespace std;

class ImageProcess{

public:
	static void pds_load(char* data_fname,int rows,int cols, int dsr_size, int PY, int Initial_JD, int Center_row, int Center_col, int ImageR, int ImageC, uchar *ImageData);
	static void pds_load(char* data_fname,int rows,int cols, uchar *ImageData, int Begin_row=0, int Begin_col=0, int ImageR=0, int ImageC=0, int dsr_size=1058, int PY=18, int Initial_JD=227757);
	static void tif_load(char* imagepath1,uchar* data1);
	static void pds2tif(char* pszFile, char* dstPath,int ImageR,int ImageC);
	static void pds2tif(char* imagepath1,char* outpath);

	static void Linear(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
	static void Linear1(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
	static void Linear2(char* pszSrcFile, char* pszDstFile,int mark=0);
	static void ImageType2_8(char* pszSrcFile, char* pszDstFile,int mark=1);
	static void Col_balance(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
	static void Col_balance1(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
	static void Col_balance2(char *img_path,char* cb_path);
	static void Col_balance3(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
	static void Col_balance4(uchar *ImageMat, int rows, int cols, uchar *ImageResult);
	static void IntraCCD_mosaic(char* pszSrcFile1, char* pszSrcFile2, char* pszDstFile,int overlap);
	static void BetweenCCD_mosaic(char* imagepathL, char* imagepathR, int OverlapSamples, char* outpath);
	static int Down_sample(char* pszSrcFile1, int batchsize, char* pszDstFile);
	static int Hijitreg_gdal(char* imagepathL, char* imagepathR, int OverlapSamples, int* dr, int* dc);


	static void xulie_process(const PipelineConfig& cfg);
	static void intra_ccd_mosaic(const PipelineConfig& cfg);
	static void ds_mosaic();
	static bool createDirectoryRecursive(const std::string& path);


	static void My_rec2NEH(const char *projRef,double X,double Y,double Z,double* N,double* E,double* H);
	static void My_NEH2rec(const char *projRef,double N,double E,double H,double* X,double* Y,double* Z);
	static void LoadDEM(char* imagepath, float X, float Y, float* Z);
	static uchar int2uchar(int temp);


	static bool WriteImageData(const char* strDestFilePath,unsigned char* pImageData,int nWidth,int nHeight,int nChannels,int nNewChannels);
	static void jichu();
	static void gdal_test();
};

#endif 
