#include "EO.h"

namespace {

bool consume_spice_error(const char* stage, const char* detail = nullptr){
	if(!failed_c()){
		return false;
	}

	SpiceChar shortMsg[1841];
	SpiceChar longMsg[1841];
	getmsg_c("SHORT", sizeof(shortMsg), shortMsg);
	getmsg_c("LONG", sizeof(longMsg), longMsg);

	if(detail != nullptr){
		fprintf(stderr, "[SPICE][%s] %s\n", stage, detail);
	}
	fprintf(stderr, "[SPICE][%s] SHORT: %s\n", stage, shortMsg);
	fprintf(stderr, "[SPICE][%s] LONG : %s\n", stage, longMsg);

	reset_c();
	return true;
}

}

EO::EO(){
}

EO::~EO(){
}

//输入：输出路径；起始航天器时间；TDI线路延迟；窗口大小；TDI模式（8，32，64，128）；采样行数
void EO::GetEO(char * out, char *SCLK, double DLINE, double BIN, double TDI, int LINE)
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

		//spkpos_c("MRO", et, "IAU_Mars", "CN+S", "MARS", pos, &lt);
		spkpos_c( "MARS", et, "IAU_Mars", "CN+S","MRO", pos, &lt);
		for(j=0; j<3; j++) pos[j] *= -1000;

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

void EO::GetEO_(char * out, char *SCLK, double DLINE, double BIN, double TDI, int LINE)
{
	int i,j,k,found;
	char utcstr[30];
	double LR,et0,et1,et,encsclk,clkout,lt,a1,a2,a3;
	double pos[3], R[3][3], R1[3][3], R2[3][3], Rt[3][3];
	char out1[80];
	sprintf( out1, "%s%s.txt", out, "_1" );
	FILE *fout1 = fopen(out1, "w");
	FILE *fout = fopen(out, "w");

	SpiceInt sc;
	SpiceBoolean found0;
	bodn2c_c("MGS",&sc,&found0);
	printf("%d\n",int(sc));

	//scs2e_c(-74999, SCLK, &et0);
	scs2e_c(-94, SCLK, &et0);
	//scs2e_c(sc, SCLK, &et0);
	//LR = ( 74.0 + DLINE/16.0 )/1000000;
	//LR = 7.03/double(LINE); //M01
	LR = 7.78/double(LINE); //E02
	et1 = et0 + LR*(BIN-TDI)/2;

	//SpiceChar  * centername="i am aaaa";
	//printf("%s",centername);
	//SpiceChar  * scname="bbbbb";
	//SpiceChar  * iau= new char(30);
	//strcpy(iau,centername);
	//strcat(iau,scname);
	//printf("%s",iau);
	for(i=0;i<LINE;i++)
	{
		et = et1 + i*LR*BIN;

		//spkpos_c("MRO", et, "IAU_Mars", "CN+S", "MARS", pos, &lt);
		//spkpos_c( "MARS", et, "IAU_Mars", "CN+S","MRO", pos, &lt);
		spkpos_c( "MARS", et, "IAU_Mars", "CN+S","MGS", pos, &lt);
		for(j=0; j<3; j++) pos[j] *= -1000;

		pxform_c( "J2000", "IAU_MARS", et, R1 );
		//pxform_c( "MRO_HIRISE_OPTICAL_AXIS", "J2000", et, R2 );
		pxform_c( "MGS_SPACECRAFT", "J2000", et, R2 );

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

//输入：输出路径；起始星历时间；TDI线路延迟；窗口大小；TDI模式（8，32，64，128）；采样行数
bool EO::GetEO(char * out, double et0, double DLINE, double BIN, double TDI, int LINE, const int sample_rate=1)
{
	int i,j,k,found;
	char utcstr[30];
	double LR,et1,et,encsclk,clkout,lt,a1,a2,a3;
	double pos[3], R[3][3], R1[3][3], R2[3][3], Rt[3][3];

	LR = ( 74.0 + DLINE/16.0 )/1000000;
	et1 = et0 -LR*(TDI/2-0.5)+ LR*(BIN-1)/2;

	FILE *fout = fopen(out, "w");
	fprintf(fout, "%.12f %.12f\n", et1,LR);
	for(i=0;i<LINE;i+=sample_rate)
	{
		et = et1 + i*LR*BIN;

		spkpos_c("MRO", et, "IAU_Mars", "CN+S", "MARS", pos, &lt);
		if(consume_spice_error("GetEO/spkpos_c", out)){
			fclose(fout);
			remove(out); // 避免留下仅表头的残缺 EO，导致后续误判“文件存在”
			return false;
		}
		for(j=0; j<3; j++) pos[j] *= 1000;

		pxform_c( "J2000", "IAU_MARS", et, R1 );
		if(consume_spice_error("GetEO/pxform_c(J2000->IAU_MARS)", out)){
			fclose(fout);
			remove(out);
			return false;
		}
		pxform_c( "MRO_HIRISE_OPTICAL_AXIS", "J2000", et, R2 );
		if(consume_spice_error("GetEO/pxform_c(MRO_HIRISE_OPTICAL_AXIS->J2000)", out)){
			fclose(fout);
			remove(out);
			return false;
		}
		mxm_c( R1, R2, Rt);             //Rt为MRO_HIRISE_OPTICAL_AXIS-->IAU，和摄影测量一致，无需再转置

		pxform_c("MRO_HIRISE_OPTICAL_AXIS", "IAU_MARS", et, Rt);
		if(consume_spice_error("GetEO/pxform_c(MRO_HIRISE_OPTICAL_AXIS->IAU_MARS)", out)){
			fclose(fout);
			remove(out);
			return false;
		}

		for(j=0;j<3;j++){
			int temp=abs(j-2);
			temp=j;
			for(k=0;k<3;k++){
				R[j][k] = Rt[temp][k];
			}
		}
		//R[j][k] = Rt[k][j];
		Eigen::Matrix3d R_ei;
		float * RR = new float[9];
		R_ei << R[0][0],R[0][1],R[0][2],R[1][0],R[1][1],R[1][2],R[2][0],R[2][1],R[2][2];
		//printf("%.3lf %.3lf %.3lf\n%.3lf %.3lf %.3lf\n%.3lf %.3lf %.3lf\n\n",R[0][0],R[0][1],R[0][2],R[1][0],R[1][1],R[1][2],R[2][0],R[2][1],R[2][2]);


		//用它
		Eigen::Matrix<double,3,1> EulA = R_ei.eulerAngles(1,0,2);   //yxz内旋，右手系，用eigen分解，合成用Eul2R_
		a1=EulA[0];a2=EulA[1];a3=EulA[2];      //phi,w,k
		Eul2R_(float(a3),float(a2),float(a1),RR);
		//cout<<a1<<" "<<a2<<" "<<a3<<endl;
		//printf("%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",RR[0],RR[1],RR[2],RR[3],RR[4],RR[5],RR[6],RR[7],RR[8]);



		/*m2eul_c( R, 2, 1, 3, &a1, &a2, &a3);         //spice:累计旋转是左乘(y轴取负) //spice分解,左手系,调用Eul2R__合成
		Eul2R__(float(a3),float(a2),float(a1),RR);
		cout<<a1<<" "<<a2<<" "<<a3<<endl;
		printf("%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",RR[0],RR[1],RR[2],RR[3],RR[4],RR[5],RR[6],RR[7],RR[8]);//*/




		/*R_ei=Eigen::AngleAxisd(EulA[0], Eigen::Vector3d::UnitZ()) *         //eigen合成
             Eigen::AngleAxisd(EulA[1], Eigen::Vector3d::UnitX()) *
             Eigen::AngleAxisd(EulA[2], Eigen::Vector3d::UnitY());
		printf("%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",RR[0],RR[1],RR[2],RR[3],RR[4],RR[5],RR[6],RR[7],RR[8]);//*/

 		fprintf(fout, "%.12f\t", et);
		fprintf(fout, "%.12f\t%.12f\t%.12f\t", pos[0], pos[1], pos[2]);
		fprintf(fout, "%.17f\t%.17f\t%.17f\n", a1, a2, a3);
	}
	fclose(fout);
	return true;
}

//输入配置文件，输出外方位元素到txt
void EO::OutEO2txt(char *inputFile, char *kernelFile, const int dT){
	int count, LINE;
	char path[50],  name[50] , out[90], SCLK[50];
	double DLINE,  BIN, TDI;

	// 避免 SPICE 默认 ABORT，改为可恢复错误并输出详细日志。
	erract_c("SET", 0, "RETURN");
	errprt_c("SET", 0, "NONE");
	reset_c();
	kclear_c();

	//配置核文件
	furnsh_c(kernelFile);
	if(consume_spice_error("OutEO2txt/furnsh_c", kernelFile)){
		fprintf(stderr, "[SPICE][OutEO2txt] 内核加载失败，已跳过: %s\n", inputFile);
		return;
	}

	//逐影像输出外方位元素到txt
	FILE *fin = fopen(inputFile, "r" );
	if(fin == NULL){
		printf("未找到EO输入文件：%s\n", inputFile);
		kclear_c();
		return;
	}
	fscanf( fin, "%d%s",  &count, path );
	for(int i=0;i<count;i++)
	{
		fscanf( fin, "%s%s%lf%lf%lf%d", name, SCLK,  &DLINE,  &BIN,  &TDI,  &LINE);
		sprintf( out, "%s%s%s", path, name, ".txt");

		double et0;
		//et0 = get_et0(SCLK);
		scs2e_c( -74999, SCLK, &et0);

		//根据et0和线扫描速度计算每行对应的星历时间并找到对应的外方位元素
		//GetEO( out, SCLK,  DLINE,  BIN, TDI, LINE );
		if(!GetEO(out, et0,  DLINE,  BIN, TDI, LINE, dT)){
			fprintf(stderr, "[SPICE][OutEO2txt] 计算EO失败，输入文件: %s\n", inputFile);
			break;
		}
		//GetEO_(out,SCLK,DLINE,BIN,TDI,LINE);
	}
	fclose(fin);
	kclear_c();
}

void EO::get_EO_main(){
	int i, j, count, LINE;
	char path[50],  name[50] , out[90], SCLK[50];
	double DLINE,  BIN, TDI;
	double et0=0;
	char* utc;
	double sclkdp0;

	furnsh_c( "../data/EO/HiRISE1.txt" );

	FILE *fin = fopen( "../data/EO/input1.txt", "r" );
	fscanf( fin, "%d%s",  &count, path );

	for(i=0;i<count;i++)
	{
		fscanf( fin, "%s%s%lf%lf%lf%d", name, SCLK,  &DLINE,  &BIN,  &TDI,  &LINE );
		sprintf( out, "%s%s", path, name );

		double et0;
		et0 = get_et0(SCLK);

		//根据et0和线扫描速度计算每行对应的星历时间并找到对应的外方位元素
		//GetEO( out, SCLK,  DLINE,  BIN, TDI, LINE );
		if(!GetEO(out, et0,  DLINE,  BIN, TDI, LINE)){
			fprintf(stderr, "[SPICE][get_EO_main] 计算EO失败，停止继续处理。\n");
			break;
		}
	}
	fclose(fin);
}
void EO::EO_poly_test(){
	OutEO2txt("E:\\Mars_VS\\Mars../data/EO/input1.txt", "E:\\Mars_VS\\Mars../data/EO/HiRISE1.txt");
}



/********************************************************
*	@brief       : 重写spice函数
********************************************************/
//欧拉角计算旋转矩阵
void EO::Eul2R_(float phi,float w,float k,float* R){
	phi=-phi;
	R[0]=cos(phi)*cos(k)-sin(phi)*sin(w)*sin(k);
	R[1]=-cos(phi)*sin(k)-sin(phi)*sin(w)*cos(k);
	R[2]=-sin(phi)*cos(w);
	R[3]=cos(w)*sin(k);
	R[4]=cos(w)*cos(k);
	R[5]=-sin(w);
	R[6]=sin(phi)*cos(k)+cos(phi)*sin(w)*sin(k);
	R[7]=-sin(phi)*sin(k)+cos(phi)*sin(w)*cos(k);
	R[8]=cos(phi)*cos(w);
}

void EO::Eul2R__(float phi,float w,float k,float* R){
	R[0]=cos(phi)*cos(k)-sin(phi)*sin(w)*sin(k);
	R[1]=-cos(phi)*sin(k)-sin(phi)*sin(w)*cos(k);
	R[1]=-R[1];
	R[2]=-sin(phi)*cos(w);
	R[3]=cos(w)*sin(k);
	R[3]=-R[3];
	R[4]=cos(w)*cos(k);
	R[5]=-sin(w);
	R[5]=-R[5];
	R[6]=sin(phi)*cos(k)+cos(phi)*sin(w)*sin(k);
	R[7]=-sin(phi)*sin(k)+cos(phi)*sin(w)*cos(k);
	R[7]=-R[7];
	R[8]=cos(phi)*cos(w);
}

//输入：sclkch="4294967295.255";separator=":";base=19876.76;rate=0.02;
//将字符串类型的航天器时间转换为双精度ticks，再转换为TDB标准星历时间。
void EO::sclkch2et(char* sclkch, double sclk0, char separator, double cons, double rate,double* et){
	double sclkdp;
	//sclkch-->sclkdp
	double uniticks = 65536;
	sclkch2sclkdp(sclkch, separator, uniticks, &sclkdp);

	//sclkdh-->et
	sclkdp2et(sclkdp, sclk0, cons, rate, et);
}

//sclkch-->sclkdp
void EO::sclkch2sclkdp(char* sclkch, char separator, double uniticks, double* sclkdp){
	char *revbuf[2]; //存放分割后的子字符串 
	int num=0;

	//printf("字符串为：%s\n",sclkch);
	//sclkch为char*的字符串，可以将vector先转为char*，具体操作：先获取vector长度l，然后char* sclkch=new char[l];再for循环赋值。
	//separator为分割字符，这里为空格:' '
	//revbuf存放分割后的子字符串,需要提前定义长度：char *revbuf[100];
	//num直接定义输进去就好：int num=0；
	split(sclkch,separator,revbuf, &num); //调用函数进行分割

	double ticks = atof(revbuf[0])*uniticks + atoi(revbuf[1]);
	//printf("SCLK对应的ticks：%lf\n",ticks);

	*sclkdp=ticks;
}

//sclkdh-->et
void EO::sclkdp2et(double sclkdp,double sclk0, double cons, double rate,double* et){
	*et = cons + rate/65536*(sclkdp-sclk0);
}

//减去分区间多余ticks
void EO::get_truesclkdp(double ticks, int nparts, double *pstart, double *pstop, double *sclkdp){
	int i__, i__1, i__2, i__3, i__4, i__5;
	i__1 = nparts;

	double* ptotls = new double(nparts);
	for (i__ = 1; i__ <= i__1; ++i__) {
		pstop[(i__2 = i__ - 1)]                 //pstop[(i__2 = i__ - 1)]
		= 
			double(floor(pstop[(i__3 = i__ - 1)]+0.5));    //d_nint(&pstop[(i__3 = i__ -  1)])                                          //对每个开始结束点滴答进行四舍五入

		pstart[(i__2 = i__ - 1)]               //pstart[(i__2 = i__ - 1)]
		= 
			double(floor(pstart[(i__3 = i__ - 1)]+0.5));   //d_nint(&pstart[(i__3 = i__ -  1)])
	}

	/* For each partition, compute the total number of ticks in that partition plus all preceding partitions. */

	double d__1 = pstop[0] - pstart[0];
	ptotls[0] = floor(d__1+0.5);            //四舍五入函数
	i__1 = nparts;
	for (i__ = 2; i__ <= i__1; ++i__) {
		d__1 = ptotls[(i__3 = i__ - 2)] 
		+ pstop[(i__4 = i__ - 1)] 
		- pstart[(i__5 = i__ - 1)];          //把各分区中间空的时间除去，算出到每个分区的总ticks
		//printf("i_3:%d       i_4:%d        d_1:%lf\n",i__3,i__4,floor(d__1+0.5));
		ptotls[(i__2 = i__ - 1)] 
		= floor(d__1+0.5);
	}

	int Biaoji=10000;
	for(int i=0;i<nparts;i++){
		if(ticks>=pstart[i] && ticks<=pstop[i]){
			Biaoji=i;
		}
	}
	//printf("%lf %d\n",ticks,Biaoji);

	if(Biaoji>0){
		*sclkdp = ptotls[Biaoji-1] + ticks - pstart[Biaoji];
	}
	else{
		*sclkdp = ticks;
	}
}

double EO::get_truesclkdp(double ticks, int nparts, double *pstart, double *pstop){
	double sclkdp=0;
	int i__, i__1, i__2, i__3, i__4, i__5;
	i__1 = nparts;

	double* ptotls = new double[nparts];
	for (i__ = 1; i__ <= i__1; ++i__) {
		pstop[(i__2 = i__ - 1)]                 //pstop[(i__2 = i__ - 1)]
		= 
			double(floor(pstop[(i__3 = i__ - 1)]+0.5));    //d_nint(&pstop[(i__3 = i__ -  1)])                                          //对每个开始结束点滴答进行四舍五入

		pstart[(i__2 = i__ - 1)]               //pstart[(i__2 = i__ - 1)]
		= 
			double(floor(pstart[(i__3 = i__ - 1)]+0.5));   //d_nint(&pstart[(i__3 = i__ -  1)])
	}

	/* For each partition, compute the total number of ticks in that partition plus all preceding partitions. */

	double d__1 = pstop[0] - pstart[0];
	ptotls[0] = floor(d__1+0.5);            //四舍五入函数
	i__1 = nparts;
	for (i__ = 2; i__ <= i__1; ++i__) {
		d__1 = ptotls[(i__3 = i__ - 2)] 
		+ pstop[(i__4 = i__ - 1)] 
		- pstart[(i__5 = i__ - 1)];          //把各分区中间空的时间除去，算出到每个分区的总ticks

		ptotls[(i__2 = i__ - 1)] = floor(d__1 + 0.5);
	}

	int Biaoji=10000;
	for(int i=0;i<nparts;i++){
		if(ticks>=pstart[i] && ticks<=pstop[i]){
			Biaoji=i;
		}
	}
	//printf("%lf %d\n",ticks,Biaoji);

	if(Biaoji>0){
		sclkdp = ptotls[Biaoji-1] + ticks - pstart[Biaoji];
	}
	else{
		sclkdp = ticks;
	}/**/
	return sclkdp;
}

//TDT转TDB
void EO::TDT2TDB(double TDT, double K, double EB, double *M, double *TDB){
	double t = TDT;
	double m = M[0]+M[1]*t;
	*TDB = t + K * sin(m + EB * sin(m));
}

//获取指定时刻，物体（飞行器或星体）在J2000坐标系下相对于SSB的位置矢量。
void EO::get_pos(char* obs, double et, double* pos){
	double lt=0;
	spkpos_c(obs, et, "J2000", "NONE", "SSB", pos, &lt);
}

//获取J2000下指定时刻的航天器速度
void EO::get_v(char* obs, double et, double* v){
	SpiceBoolean found;
	SpiceInt code;

	bodn2c_c( obs, &code, &found );
	spkssb_c ( code, et, "J2000", v ); 
}

//求et时刻的线元素：包括单向光时间校正、恒星像差校正
void EO::get_LineEle(char* obs, char* targ, double et, char* frame, int iters, int IsSac, double* pos){
	double *pos1,*pos2;
	pos1 = new double[3];
	pos2 = new double[3];
	get_pos(obs, et, pos1);
	get_pos(targ, et, pos2);

	//单向光时间校正
	double lt=0;
	double clight = 2.99792458E+05;   //光速km/s
	for(int i=0; i<iters; i++){
		lt=sqrt((pos2[0]-pos1[0])*(pos2[0]-pos1[0])+(pos2[1]-pos1[1])*(pos2[1]-pos1[1])+(pos2[2]-pos1[2])*(pos2[2]-pos1[2]))/clight;
		get_pos(targ, et-lt, pos2);
	}

	double ptarg[3];
	for(int j=0; j<3; j++){ 
		ptarg[j] =pos2[j]-pos1[j];
		ptarg[j] *= -1000; //m为单位
	}
	clight *= 1000;

	double *vv = new double[3];
	//恒星像差校正
	double appobj[3];
	if(IsSac==1){
		get_v(obs, et, vv);
		//v[0]=0;v[1]=0;v[2]=0;
		vv[0]=vv[0]/clight;vv[1]=vv[1]/clight;vv[2]=vv[2]/clight;      //光速归一化
		//printf("v[2]=%lf\n",v[2]);
		double len = sqrt(ptarg[0]*ptarg[0] + ptarg[1]*ptarg[1] + ptarg[2]*ptarg[2]);
		double r[3]={ptarg[0]/len,ptarg[1]/len,ptarg[2]/len};    //单位向量
		//printf("r[2]=%lf\n",r[2]);
		double h[3];
		vcrss_c(r, vv, h);                   //求叉积：h=r×v，spice函数
		double sinphi = sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]);
		//printf("h[2]=%lf\n",h[2]);
		SpiceDouble phi = asin(sinphi);
		vrotv_c(ptarg, h, phi, appobj);    //ptarg绕h旋转phi，spice函数
	}
	else{
		appobj[0]=ptarg[0];appobj[1]=ptarg[1];appobj[2]=ptarg[2];
	}

	//从j2000旋转到对应坐标系
	double R1[3][3];
	pxform_c( "J2000", frame, et-lt, R1 ); 
	//pos = new double[3];
	mxv_c(R1, appobj, pos);
	pos;
}

//获取起始星历时间
double EO::get_et0(char* SCLK){
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

	FILE * fp1=fopen("E:\\Mars_KF../data/jz_coef\\pstart.txt","r");
	while( fscanf(fp1, "%s\n", a ) == 1 )
	{
		split(a,'E',b,&num);
		pstart[count]=atof(b[0])*pow(10.0,atoi(b[1]));
		count++;
	}
	//printf("%d\n",count);
	fclose(fp1);

	count=0;
	FILE * fp2=fopen("E:\\Mars_KF../data/jz_coef\\pstop.txt","r");
	while( fscanf(fp2, "%s\n", a ) == 1 )
	{
		split(a,'E',b,&num);
		pstop[count]=atof(b[0])*pow(10.0,atoi(b[1]));
		count++;
	}
	//printf("%d\n",count);
	fclose(fp2);

	double sclkdp1=0;
	sclkdp1 = get_truesclkdp(sclkdp, nparts, pstart, pstop);
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




/********************************************************
*	@brief       : base函数
********************************************************/
void EO::split(char *src,char separator,char** dest,int *num) {
	/*
	src 源字符串的首地址(buf的地址) 
	separator 指定的分割字符
	dest 接收子字符串的数组
	num 分割后子字符串的个数
	*/

	char *pNext;
	int count = 0;
	if (src == NULL || strlen(src) == 0){ //如果传入的地址为空或长度为0，直接终止 
		return;
	}

	char* p=src;
	int i = 0, j = 0;    
	char tmp[32][32] = {0};    
	char *p1 = (char *)malloc(1024); 
	while((p1 = strchr(p, separator)) != NULL)    
	{        
		strncpy(tmp[i], p, strlen(p) - strlen(p1));       
		p = p1 + 1;        
		i++;    
	}    
	strncpy(tmp[i], p, strlen(p)); 
	for(j = 0; j <= i; j++){ 
		dest[j]=tmp[j];
		//printf("temp[%d] = %s\n", j, dest[j]);
	}
	*num = i+1;
	//printf("分割后字符串个数：%d\n",*num);
} 	
void EO::mxm(double *m1, double *m2, double *re){
	int i__1, i__2, i__3, i__4, i__5, i__6, i__7;
	int i__, j;
	double prodm[9];
	for (i__ = 1; i__ <= 3; ++i__) { //行
		for (j = 1; j <= 3; ++j) { //列
			re[(i__1 = j + i__ * 3 - 4) ] = m1[(i__2 = i__ * 3 - 3)] * m2[(i__3 = j - 1)]
			+ m1[(i__4 = i__ * 3 - 2)] * m2[(i__5 = j + 2)] 
			+ m1[(i__6 = i__ * 3 - 1)] * m2[(i__7 = j + 5)];
		}
	}
}
void EO::m2v(double **m1, int rows, int cols, double *m2){
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			m2[i*rows+j]=m1[i][j];
		}
	}
}


