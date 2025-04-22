#include <fstream>
#include "SpiceUsr.h"
#include "EO.h"
#include "AeTri.h"
#include "base.h"
#include "AeTri.h"

double get_et0(char* SCLK){
	//求解初始时刻对应的TDB星历时间：et0
	double sclkdp=0;double et0=0;double sclkdp0=0;

	/***************************************************************
	//spice函数求解

	scencd_c(-74999, SCLK, &sclkdp0 );
	printf("spice函数转换sclkdp：%lf\n",sclkdp0);
	scs2e_c(-74999, SCLK, &et0);
	printf("spice函数转换et0结果为：%lf\n", et0);           //spice函数结果
	**************************************************************/

	//自编函数求解
	sclkch2sclkdp(SCLK, ':', 65536, &sclkdp);  //分割字符串，求解初始ticks
	printf("自编函数转换ticks：%lf\n",sclkdp);

	int nparts=19;
	double *pstart = new double[19];
	double *pstop = new double[19];     //存储sclk各分区ticks

	int count=0;
	char* a = new char[30];
	char *b[2];
	int num=2;

	FILE * fp1=fopen("E:\\Mars_KF\\data\\jz_coef\\pstart.txt","r");
	while( fscanf(fp1, "%s\n", a ) == 1 )
	{
		split(a,'E',b,&num);
		pstart[count]=atof(b[0])*pow(10.0,atoi(b[1]));
		count++;
	}
	//printf("%d\n",count);
	fclose(fp1);

	count=0;
	FILE * fp2=fopen("E:\\Mars_KF\\data\\jz_coef\\pstop.txt","r");
	while( fscanf(fp2, "%s\n", a ) == 1 )
	{
		split(a,'E',b,&num);
		pstop[count]=atof(b[0])*pow(10.0,atoi(b[1]));
		count++;
	}
	//printf("%d\n",count);
	fclose(fp2);

	double sclkdp1=0;
	sclkdp1 = get_truesclkdp1(sclkdp, nparts, pstart, pstop);
	printf("自编函数转换sclkdp：%lf\n",sclkdp1);  
	double coef[3]={7.9410475499462E+13, 5.8051252123000E+08, 9.9999988100041E-01};   //start_ticks;cons;rate
	sclkdp2et(sclkdp1, coef[0], coef[1], coef[2], &et0);       //sclkdp转TDT（MRO默认并行时间系统TDT）
	double TDT = et0;
	//TDB = unitim_c ( TDT, "TDT", "ET" );                     //spice函数TDT转TDB

	double TDB=0;
	double DELTA_T_A = 32.184; double K = 1.657E-3; double EB = 1.671E-2; double M[2] = {6.239996E0, 1.99096871E-7};
	TDT2TDB(TDT, K, EB, M, &TDB);                              //自编函数TDT转TDB
	printf("自编函数转换et0为：%lf\n",TDB);
	et0=TDB;

	return et0;
}

int main(int argc, char* argv[])
{
	/*
	int i, j, count, LINE;
	char path[50],  name[50] , out[90], SCLK[50];
	double DLINE,  BIN, TDI;
	double et0=0;
	char* utc;
	double sclkdp0;

	furnsh_c( "./HiRISE1.txt" );

	FILE *fin = fopen( ".\\input1.txt", "r" );
	fscanf( fin, "%d%s",  &count, path );

	for(i=0;i<count;i++)
	{
		fscanf( fin, "%s%s%lf%lf%lf%d", name, SCLK,  &DLINE,  &BIN,  &TDI,  &LINE );
		sprintf( out, "%s%s", path, name );

		double et0;
		et0 = get_et0(SCLK);
		
		//根据et0和线扫描速度计算每行对应的星历时间并找到对应的外方位元素
		//GetEO( out, SCLK,  DLINE,  BIN, TDI, LINE );
		GetEO1( out, et0,  DLINE,  BIN, TDI, LINE );
	}
	fclose(fin);
	*/

	cvtest();
	return 0;
}