#ifndef _EO_H_
#define _EO_H_

#include "base.h"
#include <fstream>
#include "SpiceUsr.h"

using namespace std;

void GetEO( char * out, char *SCLK, double DLINE, double BIN, double TDI, int LINE );
void GetEO1( char * out, double et0, double DLINE, double BIN, double TDI, int LINE );

void sclkch2et(char* sclkch, double SCLK00, char separator, double base,double rate,double* et);
void sclkch2sclkdp(char* sclkch, char separator, double uniticks, double* sclkdp);
void sclkdp2et(double sclkdp,double SCLK00, double cons, double rate,double* et);

void get_truesclkdp(double ticks, int nparts, double *pstart, double *pstop, double *sclkdp);
double get_truesclkdp1(double ticks, int nparts, double *pstart, double *pstop);
void TDT2TDB(double TDT, double K, double EB, double *M, double *TDB);

void get_pos(char* obs, double et, char* frame);
void get_v(char* obs, double et, double* v);
void get_LineEle(char* obs, char* targ, double et, char* frame, int iters,int IsSac, double* pos);

double get_et0(char* SCLK);

void get_EO_main();


#include "../src/EO.cpp"
#endif 