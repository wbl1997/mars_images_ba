#ifndef _EO_H_
#define _EO_H_

#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>

#include "SpiceUsr.h"
#include <Eigen/Geometry>

using namespace std;


class EO{
public:
	EO();
	~EO();

	struct EOele{
		double Xs;
		double Ys;
		double Zs;
		double phi;
		double w;
		double k;
	};

	//��ȡ�ⷽλԪ�ز����
	static void GetEO(char * out, char *SCLK, double DLINE, double BIN, double TDI, int LINE);
	static void GetEO_(char * out, char *SCLK, double DLINE, double BIN, double TDI, int LINE);
	static bool GetEO(char * out, double et0, double DLINE, double BIN, double TDI, int LINE, const int sample_rate);
	
	//�������Ӱ����ⷽλ����
	static void OutEO2txt(char *inputFile, char *kernelFile, const int dT=200);
	void get_EO_main();
	void EO_poly_test();


	/********************************************************
	*	@brief       : ��дspice����
	********************************************************/
	//ŷ����ת����
	static void Eul2R_(float phi,float w,float k,float* R);
	static void Eul2R__(float phi,float w,float k,float* R);
	void sclkch2et(char* sclkch, double sclk0, char separator, double cons, double rate,double* et);
	void sclkch2sclkdp(char* sclkch, char separator, double uniticks, double* sclkdp);
	void sclkdp2et(double sclkdp,double sclk0, double cons, double rate,double* et);
	void get_truesclkdp(double ticks, int nparts, double *pstart, double *pstop, double *sclkdp);
	double get_truesclkdp(double ticks, int nparts, double *pstart, double *pstop);
	void TDT2TDB(double TDT, double K, double EB, double *M, double *TDB);
	void get_pos(char* obs, double et, double* pos);
	void get_v(char* obs, double et, double* v);
	void get_LineEle(char* obs, char* targ, double et, char* frame, int iters, int IsSac, double* pos);
	double get_et0(char* SCLK);


	/********************************************************
	*	@brief       : base����
	********************************************************/
	void split(char *src,char separator,char** dest,int *num);
	void mxm(double *m1, double *m2, double *re);
	void m2v(double **m1, int rows, int cols, double *m2);


public:
	double Xs;
	double Ys;
	double Zs;
	double phi;
	double w;
	double k;
};

#endif 