#include <fstream>
#include "SpiceUsr.h"
#include "EO.h"

void GetEO( char * out, char *SCLK, double DLINE, double BIN, double TDI, int LINE )
{
	int i,j,k,found;
	char utcstr[30];
	double LR,et0,et1,et,encsclk,clkout,lt,a1,a2,a3;
	double pos[3], R[3][3], R1[3][3], R2[3][3], Rt[3][3];
	char out1[80];
	sprintf( out1, "%s%s.txt", out, "_1" );
	FILE *fout1 = fopen(out1, "w");
	FILE *fout = fopen(out, "w");
	scs2e_c(-74999, SCLK, &et0);
	LR = ( 74.0 + DLINE/16.0 )/1000000;
	et1 = et0 + LR*(BIN-TDI)/2;

	SpiceChar  * centername="i am aaaa";
	printf("%s",centername);
	SpiceChar  * scname="bbbbb";
	SpiceChar  * iau= new char(30);
	strcpy(iau,centername);
	strcat(iau,scname);
	printf("%s",iau);
	for(i=0;i<LINE;i++)
	{
		et = et1 + i*LR*BIN;

		spkpos_c("MRO", et, "IAU_Mars", "CN+S", "MARS", pos, &lt);
		for(j=0; j<3; j++) pos[j] *= 1000;

		pxform_c( "J2000", "IAU_MARS", et, R1 );
        pxform_c( "MRO_HIRISE_OPTICAL_AXIS", "J2000", et, R2 );

		mxm_c( R1, R2, Rt ); //Rt为MRO_HIRISE_OPTICAL_AXIS-->IAU，和摄影测量一致，无需再转置
		for(j=0;j<3;j++)
			for(k=0;k<3;k++)
				R[j][k] = Rt[k][j];
		m2eul_c( R, 3, 1, 2, &a3, &a2, &a1 );

		fprintf(fout, "%.12f\t%.12f\t%.12f\t", pos[0],  pos[1], pos[2]);
        fprintf(fout, "%.17f\t%.17f\t%.17f\n", a1, a2,  a3);
		fprintf(fout1, "%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\n",
			    R[0][0], R[0][1],  R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2] );
	}
	fclose(fout);
	fclose(fout1);
}
