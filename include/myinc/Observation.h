#ifndef _OBSERVATION_H_
#define _OBSERVATION_H_

#include "EO.h"

#include <math.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include <Eigen/Dense>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "mygdal.h"
#include "ImageProcess.h"
#include "ImageMatch.h"
#include "PipelineConfig.h"


using namespace cv;
using namespace std;
using namespace Eigen;


class Observation{
public:
	PipelineConfig cfg_;

	template <typename T>
	static void rec2geo(T rec[3],T re,T f,T* lon,T* lat,T* H);

	void Polynomial3_EO(char* EO_txt,double* Poly_C, char* polyCC_txt);
	void Polynomial3_EO1(char* EO_txt,double* Poly_C, char* polyCC_txt);
	void Get_PolyEO(double et,double* Poly_C,float* EO);
	void Get_PolyEO1(double et,double* Poly_C,float* EO);

	void Eul2R(float phi,float w,float k,float* R);
	void IO_correct(int sample,int BIN,int TDI,float* IO,float* xp,float* yp);
	void IO_correct(float sample,int BIN,int TDI,float* IO,float* xp,float* yp);
	void Get_groundtruth(float xp ,float yp, float* EO, float f, float rMars, float* GC);
	void Get_groundtruth(float xp ,float yp, float* EO, MatrixXf dR, float* dS, float f, float rMars, float* GC);
	void Get_groundtruth(char* DEMdoc, int sample, int line,int BIN, int TDI, float meanZ, float* EO, float* IO, float* GC);
	void Get_imageCoor(float* EO, MatrixXf dR, float* dS, float* IO, float* GC, float* IC);
	void G2I(float* GC, float* RIO, double* RPoly_C, int* Rrc, double RbeginT, double LR, MatrixXf dR, float* dS, int Rrows=40000, int BIN=1, int TDI=128);


	void xulie_mosaic1(char* src_path, char* prefix, int OverlapSamples, int seq_index=0);
	void xulie_downsample();
	void prepare_feature_match_mosaic(int have_mosaic0=0);
	void fenfu_extract(int mark=0);
	void fenfu_match1(float* fs_c,int mark=0);
	void ImageAllMatch(char* imagepath1,char* imagepath2);

	void Compute_fsc(float* fs_c,int mark=0);
	void Compute_MI(float* MI_table);
	void Tichu_CX(int mark, float* fs_c,int LRmark=0);
	void Tichu_CX_ByGlobalControl(float* fs_c,int LRmark=0);
	void MatchGet(int mark, float* fs_c);

	void SemiDenseGrid_match1(float* fs_c);
	void SemiDenseGrid_match2(float* fs_c,int mark=0);
	void Iterative_refinement(float* fs_c,int mark=0);

	void Generate_matchPoint_Between_CCD(float* fs_c);
	void JitterShow();

	void Img_Preprcess(int seq_index=0);
	void ground_match();


	void Obs_main();

	void EpipolarImage();



	// 成功返回 true；EO/观测文件准备失败返回 false（调用方应中止 BA）
	bool prepare_for_BA(int mark=0);

	void SetObserveTxT_feaBA(char* matchfile,char* EOfile,char* IOtxt,char* xulie_ID1,char* xulie_ID2,char* observetxt,int* base);
	void SetObserveTxT_gridFI(char* matchfile,char* EOfile,char* IOtxt,char* xulie_ID1,char* xulie_ID2,char* observetxt,int* base);
	void SetObserveTxT2(char* matchfile,char* EOfile,char* IOtxt,char* xulie_ID1,char* xulie_ID2,char* observetxt,int* base);
	void GetImageGroundRange(char* EOfile,char* IOtxt,char* xulie_ID1,char* outfile,int rows, int cols, float H);


	//XYZתNEH
	void GetLocalNEH(char* ProjFile,char* XYZfile,char* NEHfile);
	void Match_Result_Show(char* featurepoint1,int j,float* fs_c,int mark=0);
	void Draw_FeaturePoint_betweenCCD(int j,int seq_index=0);
	void Draw_InterCCDMatch_OnMosaic(int seq_index=0);
	void Feature_Show(char* featurepoint1,int j);
	void Res_Show();

	ImageProcess IP_;
};



#endif
