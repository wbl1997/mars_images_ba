#include "mainprocess.h"


//原始分辨率相邻CCD影像匹配、拼接，生成框幅影像，并记录dr、dc
void xulie_mosaic(char* src_path, char* prefix, int OverlapSamples, char* mosaic_path0){
	int CCD_num=2;
	int *dr=new int[CCD_num-1];
	int *dc=new int[CCD_num-1];
	memset(dr,0,sizeof(int)*(CCD_num-1));
	memset(dc,0,sizeof(int)*(CCD_num-1));

	for(int i=0;i<CCD_num-1;i++){
		char* img_path1 = new char[80];
		sprintf( img_path1, "%s%s%s%d%s", src_path, "\\", prefix, i, ".tif");
		char* img_path2 = new char[80];
		sprintf( img_path2, "%s%s%s%d%s", src_path, "\\", prefix, i+1, ".tif");
		Hijitreg_gdal(img_path1, img_path2, OverlapSamples, &(dr[i]), &(dc[i]));
	}

	int rows=40000;
	int cols=2048;

	int *beginR=new int[CCD_num];
	int *endR=new int[CCD_num];
	int *beginC=new int[CCD_num];
	int *endC=new int[CCD_num];
	memset(beginR,0,sizeof(int)*(CCD_num));
	memset(endR,0,sizeof(int)*(CCD_num));
	memset(beginC,0,sizeof(int)*(CCD_num));
	memset(endC,0,sizeof(int)*(CCD_num));

	int minR=65535;
	int maxR=-65535;
	int minC=65535;
	int maxC=-65535;

	for(int i=0;i<CCD_num;i++){
		if(i==0){
			beginR[i] = 0;
			endR[i] = rows;
			beginC[i]=0;
			endC[i]=cols;
			if(beginR[i]<minR) minR=beginR[i];
			if(endR[i]>maxR) maxR=endR[i];
			if(beginC[i]<minC) minC=beginC[i];
			if(endC[i]>maxC) maxC=endC[i];
		}
		else{
			beginR[i]=beginR[i-1]+dr[i-1];
			endR[i]=beginR[i]+rows;
			beginC[i]=endC[i-1]-OverlapSamples+dc[i-1];
			endC[i]=beginC[i]+cols;
			if(beginR[i]<minR) minR=beginR[i];
			if(endR[i]>maxR) maxR=endR[i];
			if(beginC[i]<minC) minC=beginC[i];
			if(endC[i]>maxC) maxC=endC[i];
		}


	}

	char* mosaictxt = new char[80];
	sprintf( mosaictxt, "%s%s", src_path, "\\mosaic.txt");
	FILE *fp=fopen(mosaictxt,"w");
	for(int i=0;i<CCD_num;i++){
		beginR[i]-= minR;
		endR[i]  -= minR;
		beginC[i]-= minC;
		endC[i]  -= minC;
		fprintf(fp,"%d %d %d %d %d\n",i,beginR[i],endR[i],beginC[i],endC[i]);
	}
	fclose(fp);

	rows=maxR-minR;
	cols=maxC-minC;

	//生成拼接框幅
	GDALAllRegister();
	GDALDataType Type0 = GDT_Byte;
	char* mosaic_path = new char[80];
	sprintf( mosaic_path, "%s%s", mosaic_path0, "\\mosaic.tif");
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	GDALDataset* dst = pDriver->Create(mosaic_path, cols, rows, 1, GDT_Byte, ppszOptions);

	//读取原始影像灰度并写入
	int rows1,cols1;
	int rows0,cols0;
	for(int i=0;i<CCD_num;i++){
		GDALDataset *poDataset;
		GDALRasterBand *poBand;
		char* img_path1 = new char[80];
		sprintf( img_path1, "%s%s%s%d%s", src_path, "\\", prefix, i, ".tif");
		poDataset = (GDALDataset*) GDALOpen( img_path1,GA_ReadOnly );
		if( poDataset == NULL || poDataset->GetRasterCount()<1)
		{
			printf( "File1: %s不能打开！\n",img_path1);
			return;
		}

		poBand = poDataset->GetRasterBand(1);
		Type0 = poBand->GetRasterDataType();
		cols0 = poBand->GetXSize();
		rows0 = poBand->GetYSize();

		int npart=4;
		int rows1=rows0/npart;
		int cols1=cols0/npart;
		for(int j=0;j<npart;j++){
			for(int k=0;k<npart;k++){
				uchar* imgData1 = new uchar[rows1*cols1];
				//uchar* imgData1 = (uchar*) CPLMalloc(sizeof(uchar)*rows1*cols1); 
				poBand->RasterIO(GF_Read, cols1*j, rows1*k, cols1, rows1, 
					imgData1, cols1, rows1, Type0, 0, 0 );

				dst->GetRasterBand(1)->RasterIO(GF_Write, 
					beginC[i]+cols1*j, 
					beginR[i]+rows1*k, 
					cols1, 
					rows1, 
					imgData1, 
					cols1, 
					rows1,
					Type0, 
					0, 
					0); 
				if (dst == nullptr)
				{
					printf("Can't Write Image!");
					return;
				}
				delete []imgData1;
				imgData1=NULL;
				//CPLFree(imgData1);
			}
		}
		GDALClose(poBand);
		GDALClose(poDataset);
	}

	GDALClose(dst);
	char* mosaic_path1 = new char[80];
	sprintf( mosaic_path1, "%s%s", mosaic_path0, "\\mosaic1.tif");
	Col_balance2(mosaic_path, mosaic_path1);
}

void xulie_mosaic1(char* src_path, char* prefix, int OverlapSamples,int CCD_num,int rows,int cols){
	int *dr=new int[CCD_num-1];
	int *dc=new int[CCD_num-1];
	memset(dr,0,sizeof(int)*(CCD_num-1));
	memset(dc,0,sizeof(int)*(CCD_num-1));

	for(int i=0;i<CCD_num-1;i++){
		char* img_path1 = new char[80];
		sprintf( img_path1, "%s%s%s%d%s", src_path, "\\", prefix, i, ".tif");
		char* img_path2 = new char[80];
		sprintf( img_path2, "%s%s%s%d%s", src_path, "\\", prefix, i+1, ".tif");
		Hijitreg_gdal(img_path1, img_path2, OverlapSamples, &(dr[i]), &(dc[i]));
	}

	int *beginR=new int[CCD_num];
	int *endR=new int[CCD_num];
	int *beginC=new int[CCD_num];
	int *endC=new int[CCD_num];
	memset(beginR,0,sizeof(int)*(CCD_num));
	memset(endR,0,sizeof(int)*(CCD_num));
	memset(beginC,0,sizeof(int)*(CCD_num));
	memset(endC,0,sizeof(int)*(CCD_num));

	int minR=65535;
	int maxR=-65535;
	int minC=65535;
	int maxC=-65535;

	for(int i=0;i<CCD_num;i++){
		if(i==0){
			beginR[i] = 0;
			endR[i] = rows;
			beginC[i]=0;
			endC[i]=cols;
			if(beginR[i]<minR) minR=beginR[i];
			if(endR[i]>maxR) maxR=endR[i];
			if(beginC[i]<minC) minC=beginC[i];
			if(endC[i]>maxC) maxC=endC[i];
		}
		else{
			beginR[i]=beginR[i-1]+dr[i-1];
			endR[i]=beginR[i]+rows;
			beginC[i]=endC[i-1]-OverlapSamples+dc[i-1];
			endC[i]=beginC[i]+cols;
			if(beginR[i]<minR) minR=beginR[i];
			if(endR[i]>maxR) maxR=endR[i];
			if(beginC[i]<minC) minC=beginC[i];
			if(endC[i]>maxC) maxC=endC[i];
		}


	}

	char* mosaictxt = new char[80];
	sprintf( mosaictxt, "%s%s", src_path, "\\mosaic.txt");
	FILE *fp=fopen(mosaictxt,"w");
	for(int i=0;i<CCD_num;i++){
		beginR[i]-= minR;
		endR[i]  -= minR;
		beginC[i]-= minC;
		endC[i]  -= minC;
		fprintf(fp,"%d %d %d %d %d\n",i,beginR[i],endR[i],beginC[i],endC[i]);
	}
	fclose(fp);

	rows=maxR-minR;
	cols=maxC-minC;

	//生成拼接框幅
	GDALAllRegister();
	GDALDataType Type0 = GDT_Byte;
	char* mosaic_path = new char[80];
	sprintf( mosaic_path, "%s%s", src_path, "\\mosaic.tif");
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	GDALDataset* dst = pDriver->Create(mosaic_path, cols, rows, 1, GDT_Byte, ppszOptions);

	//读取原始影像灰度并写入
	int rows1,cols1;
	int rows0,cols0;
	for(int i=0;i<CCD_num;i++){
		GDALDataset *poDataset;
		GDALRasterBand *poBand;
		char* img_path1 = new char[80];
		sprintf( img_path1, "%s%s%s%d%s", src_path, "\\", prefix, i, ".tif");
		poDataset = (GDALDataset*) GDALOpen( img_path1,GA_ReadOnly );
		if( poDataset == NULL || poDataset->GetRasterCount()<1)
		{
			printf( "File1: %s不能打开！\n",img_path1);
			return;
		}

		poBand = poDataset->GetRasterBand(1);
		Type0 = poBand->GetRasterDataType();
		cols0 = poBand->GetXSize();
		rows0 = poBand->GetYSize();

		int npart=4;
		int rows1=rows0/npart;
		int cols1=cols0/npart;
		for(int j=0;j<npart;j++){
			for(int k=0;k<npart;k++){
				uchar* imgData1 = new uchar[rows1*cols1];
				//uchar* imgData1 = (uchar*) CPLMalloc(sizeof(uchar)*rows1*cols1); 
				poBand->RasterIO(GF_Read, cols1*j, rows1*k, cols1, rows1, 
					imgData1, cols1, rows1, Type0, 0, 0 );

				dst->GetRasterBand(1)->RasterIO(GF_Write, 
					beginC[i]+cols1*j, 
					beginR[i]+rows1*k, 
					cols1, 
					rows1, 
					imgData1, 
					cols1, 
					rows1,
					Type0, 
					0, 
					0); 
				if (dst == nullptr)
				{
					printf("Can't Write Image!");
					return;
				}
				delete []imgData1;
				imgData1=NULL;
				//CPLFree(imgData1);
			}
		}
		GDALClose(poBand);
		GDALClose(poDataset);
	}

	GDALClose(dst);
	char* mosaic_path1 = new char[80];
	sprintf( mosaic_path1, "%s%s", src_path, "\\mosaic_col.tif");
	Col_balance2(mosaic_path, mosaic_path1);
	char* ds_path1 = new char[80];
	sprintf( ds_path1, "%s%s", src_path, "\\mosaic_ds4.tif");
	Down_sample(mosaic_path1, 16, ds_path1);
}

//影像降采样并存储（分四级：2，4，8，16）
void xulie_downsample(char* src_path, char* prefix,char* downsample_path){
	int CCD_num=10;
	for(int j=1;j<5;j++){
		//新建文件夹
		char* ds_path = new char[80];
		sprintf( ds_path, "%s%s%d", downsample_path, "\\", j);
		int status = mkdir(ds_path); 

		//降采样
		int batchsize=pow(double(2),double(j));
		for(int i=0;i<CCD_num;i++){
			char* img_path = new char[80];
			sprintf( img_path, "%s%s%s%s%d%s", src_path, "\\", prefix, "_RED", i, ".tif");
			char* dst_path = new char[80];
			sprintf( dst_path, "%s%s%s%s%d%s", ds_path, "\\", prefix, "_RED", i, ".tif");
			Down_sample(img_path, batchsize, dst_path);
			delete []img_path;
			delete []dst_path;
		}
	}
}

void fenfu_match(char* filepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2){
	int cols = 2048;
	int CCD_num=10;
	char* filepath1=filepath;

	/*
	//原始分辨率拼接&输出每级拼接参数及影像到downsample文件夹
	for(int i=0;i<2;i++){
	char* xulie_ID;
	int rows;
	if(i==0){
	xulie_ID=xulie_ID1;
	rows=rows1;
	}
	else if(i==1){
	xulie_ID=xulie_ID2;
	rows=rows2;
	}

	char* src_path = new char[80];
	sprintf( src_path, "%s%s%s%s", filepath,"\\", xulie_ID, "\\downsample\\0");
	char* prefix = new char[80];
	sprintf( prefix, "%s%s", xulie_ID, "_RED");
	int OverlapSamples=48;
	xulie_mosaic1(src_path, prefix, OverlapSamples,CCD_num, rows, cols);

	char* mosaictxt = new char[80];
	sprintf(mosaictxt, "%s%s%s%s%d%s", filepath1, "\\", xulie_ID, "\\downsample\\",0,"\\mosaic.txt");
	//mosaictxt="G:\\mosaic.txt";
	FILE *fp0=fopen(mosaictxt,"r");
	if(fp0==NULL){
	return;
	}
	int CCD_id,beginR,endR,beginC,endC;
	CCD_id=beginR=endR=beginC=endC=0;

	for(int j=1;j<5;j++){
	char* ds_path_mosaictxt = new char[80];
	sprintf( ds_path_mosaictxt, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID, "\\downsample\\",j,"\\mosaic.txt");
	FILE *fp=fopen(ds_path_mosaictxt,"w");
	int ds_r=pow(double(2),double(j));
	while(fp0 && !feof(fp0)){
	fscanf(fp0,"%d %d %d %d %d\n",&CCD_id,&beginR,&endR,&beginC,&endC);
	fprintf(fp,"%d %d %d %d %d\n",CCD_id,beginR/ds_r,endR/ds_r,beginC/ds_r,endC/ds_r);
	}
	fclose(fp);
	rewind(fp0);
	}
	fclose(fp0); 
	}


	char* mosaic_path = new char[80];
	sprintf( mosaic_path, "%s%s%s%s", filepath,"\\", xulie_ID2, "\\downsample\\0\\mosaic_col.tif");
	char* ds_path = new char[80];
	sprintf( ds_path, "%s%s%s%s", filepath,"\\", xulie_ID2, "\\downsample\\0\\mosaic_ds4.tif");
	//Down_sample(mosaic_path, 16, ds_path);
	


	//提取特征点并存储
	int ch=6;
	int Localmax_win[5]={129,161,81,49,37};
	for(int i=4;i>=0;i--){
		//提取特征点
		FILE *fp_f;
		for(int j=0;j<CCD_num;j++){
			printf("提取特征点：L-%d-%d……\n",i,j);
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
			std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
			Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			fp_f=fopen(featurepoint1,"w");
			for(int i=0;i<KeyPoint_x1.size();i++){
				fprintf(fp_f,"%d %d %d\n",0,KeyPoint_x1[i],KeyPoint_y1[i]);
			}
			KeyPoint_x1.swap(vector<int>());
			KeyPoint_y1.swap(vector<int>());
			fclose(fp_f);
		}

		for(int j=0;j<CCD_num;j++){
			printf("提取特征点：R-%d-%d……\n",i,j);
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", j, ".txt");

			std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
			Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			fp_f=fopen(featurepoint1,"w");
			for(int i=0;i<KeyPoint_x1.size();i++){
				fprintf(fp_f,"%d %d %d\n",0,KeyPoint_x1[i],KeyPoint_y1[i]);
			}
			KeyPoint_x1.swap(vector<int>());
			KeyPoint_y1.swap(vector<int>());
			fclose(fp_f);
		}
	}
	*/



	//逐层匹配
	float* fs_c = new float[6];
	memset(fs_c,1.0,sizeof(float)*6);
	int CCD_id;
	int BJ=0;
	for(int i=4;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char* ds_path_mosaictxt1 = new char[80];
		sprintf( ds_path_mosaictxt1, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		int* mosaic_c1=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]);
		}
		fclose(fpm1);
		char* ds_path_mosaictxt2 = new char[80];
		sprintf( ds_path_mosaictxt2, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		int* mosaic_c2=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]);
		}
		fclose(fpm2);


		//匹配
		for(int j=0;j<CCD_num;j++){
			printf("%d-%d%s\n",i,j,"……");
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
			char* outpoint1 = new char[80];
			sprintf( outpoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_.txt");
			for(int k=0;k<CCD_num;k++){
				char* imgR_path = new char[80];
				sprintf( imgR_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", k, ".tif");
				char* featurepoint2 = new char[80];
				sprintf( featurepoint2, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", k, ".txt");
				if(i==4){
					CC_match2(imgL_path, imgR_path, 15, 0.95, 2, fs_c, k, featurepoint1, featurepoint2, outpoint1);
				}
				else{
					float* fs_c1 = new float[6];
					fs_c1[0]=fs_c[0];
					fs_c1[1]=fs_c[1];
					fs_c1[2]=fs_c[2]*2+fs_c[0]*mosaic_c1[j*4+0]+fs_c[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
					fs_c1[3]=fs_c[3];
					fs_c1[4]=fs_c[4];
					fs_c1[5]=fs_c[5]*2+fs_c[3]*mosaic_c1[j*4+0]+fs_c[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];
					//beginR、beginC 
					if(i==1&&j==1&&k==1){
						limit_match1(imgL_path, imgR_path, 15, 30*(5-i), 0.9, 2, fs_c1, k, featurepoint1, featurepoint2, outpoint1);
					}
					else if(i==0){
						limit_match1(imgL_path, imgR_path, 21, 30*(5-i), 0.8, 2, fs_c1, k, featurepoint1, featurepoint2, outpoint1);
					}
					else{
						limit_match1(imgL_path, imgR_path, 7*(5-i), 30*(5-i), 0.8, 2, fs_c1, k, featurepoint1, featurepoint2, outpoint1);
					}
				}
				int rem=remove(featurepoint1);
				int ren=rename(outpoint1,featurepoint1);
				//printf("%d %d\n",rem,ren);
			}
		}

		//i结束后，进行匹配点筛选和重排，加载拼接系数并计算fs系数。
		//仿射系数计算
		std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
		std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
		for(int j=0;j<CCD_num;j++){
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
			FILE *fp=fopen(featurepoint1,"r");
			int bj,row,col,imgID,mrow,mcol;
			float m_score;
			double sfr=pow(double(2),double(i))/pow(double(2),double(4));
			while(!feof(fp)){
				fscanf(fp,"%d ",&bj);
				if(bj==1){
					fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score);
					KeyPoint_x1.push_back(row+mosaic_c1[j*4+0]);
					KeyPoint_y1.push_back(col+mosaic_c1[j*4+2]);
					KeyPoint_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
					KeyPoint_y2.push_back(mcol+mosaic_c2[imgID*4+2]);
					KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
					KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
					KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[imgID*4+0])*sfr));
					KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[imgID*4+2])*sfr));
				}
				else{
					fscanf(fp,"%d %d\n",&row,&col);
				}
			}
			fclose(fp);
		}
		int * match = new int[KeyPoint_x1.size()];
		memset(match,-1,sizeof(int)*KeyPoint_x1.size());
		RANSAC_fs2(KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,50*(5-i),2000,match,fs_c);  //剔除粗差

		char* imgL_path1 = new char[80];
		sprintf( imgL_path1, "%s%s%s%s%d%s%s", filepath, "\\", xulie_ID1, "\\downsample\\",0,"\\","mosaic_ds4.tif");
		char* imgR_path1 = new char[80];
		sprintf( imgR_path1, "%s%s%s%s%d%s%s", filepath, "\\", xulie_ID2, "\\downsample\\",0,"\\","mosaic_ds4.tif");
		drawMatch3(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,1);
		drawMatch2(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,1);

		KeyPoint_x1.swap(vector<int>());
		KeyPoint_y1.swap(vector<int>());
		KeyPoint_x2.swap(vector<int>());
		KeyPoint_y2.swap(vector<int>());
		KeyPoint_x11.swap(vector<int>());
		KeyPoint_y11.swap(vector<int>());
		KeyPoint_x22.swap(vector<int>());
		KeyPoint_y22.swap(vector<int>());

		//存储结果并跳出
		if(BJ==1){
			//std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
			//std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
			for(int j=0;j<CCD_num;j++){
				char* featurepoint1 = new char[80];
				sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
				char* featurepoint2 = new char[80];
				sprintf( featurepoint2, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_match.txt");
				FILE *fp=fopen(featurepoint1,"r");
				FILE *fp_re=fopen(featurepoint2,"w");
				int bj,row,col,imgID,mrow,mcol;
				float m_score;
				double sfr=pow(double(2),double(i))/pow(double(2),double(4));
				int count=0;
				while(!feof(fp)){
					fscanf(fp,"%d ",&bj);
					if(bj==1){
						fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score);
						KeyPoint_x1.push_back(row+mosaic_c1[j*4+0]);
						KeyPoint_y1.push_back(col+mosaic_c1[j*4+2]);
						KeyPoint_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
						KeyPoint_y2.push_back(mcol+mosaic_c2[imgID*4+2]);
						//KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
						//KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
						//KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[imgID*4+0])*sfr));
						//KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[imgID*4+2])*sfr));
						float vx=KeyPoint_x1[count]*fs_c[0]+KeyPoint_y1[count]*fs_c[1]+fs_c[2]-KeyPoint_x2[count];
						float vy=KeyPoint_x1[count]*fs_c[3]+KeyPoint_y1[count]*fs_c[4]+fs_c[5]-KeyPoint_y2[count];
						if(sqrt(vx*vx+vy*vy)<250 && abs(vx)<150 && abs(vy)<150){
							fprintf(fp_re,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,m_score);
						}
						count++;
					}
					else{
						fscanf(fp,"%d %d\n",&row,&col);
					}
				}
				fclose(fp);
				fclose(fp_re);
			}

			break;
		}

		KeyPoint_x1.swap(vector<int>());
		KeyPoint_y1.swap(vector<int>());
		KeyPoint_x2.swap(vector<int>());
		KeyPoint_y2.swap(vector<int>());
		KeyPoint_x11.swap(vector<int>());
		KeyPoint_y11.swap(vector<int>());
		KeyPoint_x22.swap(vector<int>());
		KeyPoint_y22.swap(vector<int>());
	}
	//根据低层次匹配关系，预测高层次匹配范围

	//将匹配范围对应到每张影像上去

	//遍历所有预测范围影像块，进行特征点提取，进行预测位置，判断是否在右影像块范围内，
	//是的话就进一步找预测位置附近的特征点，进行匹配。

	//匹配对进行输出


	//先对融合影像提取特征点
}

void fenfu_match1(char* filepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2,float* fs_c){
	int cols = 2048;
	int CCD_num=10;
	char* filepath1=filepath;

	/*
	//原始分辨率拼接&输出每级拼接参数及影像到downsample文件夹
	for(int i=0;i<2;i++){
	char* xulie_ID;
	int rows;
	if(i==0){
	xulie_ID=xulie_ID1;
	rows=rows1;
	}
	else if(i==1){
	xulie_ID=xulie_ID2;
	rows=rows2;
	}

	char* src_path = new char[80];
	sprintf( src_path, "%s%s%s%s", filepath,"\\", xulie_ID, "\\downsample\\0");
	char* prefix = new char[80];
	sprintf( prefix, "%s%s", xulie_ID, "_RED");
	int OverlapSamples=48;
	xulie_mosaic1(src_path, prefix, OverlapSamples,CCD_num, rows, cols);

	char* mosaictxt = new char[80];
	sprintf(mosaictxt, "%s%s%s%s%d%s", filepath1, "\\", xulie_ID, "\\downsample\\",0,"\\mosaic.txt");
	//mosaictxt="G:\\mosaic.txt";
	FILE *fp0=fopen(mosaictxt,"r");
	if(fp0==NULL){
	return;
	}
	int CCD_id,beginR,endR,beginC,endC;
	CCD_id=beginR=endR=beginC=endC=0;

	for(int j=1;j<5;j++){
	char* ds_path_mosaictxt = new char[80];
	sprintf( ds_path_mosaictxt, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID, "\\downsample\\",j,"\\mosaic.txt");
	FILE *fp=fopen(ds_path_mosaictxt,"w");
	int ds_r=pow(double(2),double(j));
	while(fp0 && !feof(fp0)){
	fscanf(fp0,"%d %d %d %d %d\n",&CCD_id,&beginR,&endR,&beginC,&endC);
	fprintf(fp,"%d %d %d %d %d\n",CCD_id,beginR/ds_r,endR/ds_r,beginC/ds_r,endC/ds_r);
	}
	fclose(fp);
	rewind(fp0);
	}
	fclose(fp0); 
	}


	char* mosaic_path = new char[80];
	sprintf( mosaic_path, "%s%s%s%s", filepath,"\\", xulie_ID2, "\\downsample\\0\\mosaic_col.tif");
	char* ds_path = new char[80];
	sprintf( ds_path, "%s%s%s%s", filepath,"\\", xulie_ID2, "\\downsample\\0\\mosaic_ds4.tif");
	//Down_sample(mosaic_path, 16, ds_path);
	


	//提取特征点并存储
	int ch=6;
	int Localmax_win[5]={129,161,81,49,37};
	for(int i=0;i>=0;i--){
		//提取特征点
		FILE *fp_f;
		for(int j=0;j<CCD_num;j++){
			printf("提取特征点：L-%d-%d……\n",i,j);
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
			std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
			Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			fp_f=fopen(featurepoint1,"w");
			for(int i=0;i<KeyPoint_x1.size();i++){
				fprintf(fp_f,"%d %d %d\n",0,KeyPoint_x1[i],KeyPoint_y1[i]);
			}
			KeyPoint_x1.swap(vector<int>());
			KeyPoint_y1.swap(vector<int>());
			fclose(fp_f);
		}

		for(int j=0;j<CCD_num;j++){
			printf("提取特征点：R-%d-%d……\n",i,j);
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", j, ".txt");

			std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
			Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			fp_f=fopen(featurepoint1,"w");
			for(int i=0;i<KeyPoint_x1.size();i++){
				fprintf(fp_f,"%d %d %d\n",0,KeyPoint_x1[i],KeyPoint_y1[i]);
			}
			KeyPoint_x1.swap(vector<int>());
			KeyPoint_y1.swap(vector<int>());
			fclose(fp_f);
		}
	}
	*/

	//逐层匹配
	//float* fs_c = new float[6];
	//memset(fs_c,1.0,sizeof(float)*6);
	int CCD_id;
	int BJ=0;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char* ds_path_mosaictxt1 = new char[80];
		sprintf( ds_path_mosaictxt1, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		int* mosaic_c1=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]);
		}
		fclose(fpm1);
		char* ds_path_mosaictxt2 = new char[80];
		sprintf( ds_path_mosaictxt2, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		int* mosaic_c2=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]);
		}
		fclose(fpm2);


		//匹配
		for(int j=0;j<CCD_num;j++){
			printf("%d-%d%s\n",i,j,"……");
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
			char* outpoint1 = new char[80];
			sprintf( outpoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_.txt");
			for(int k=0;k<CCD_num;k++){
				char* imgR_path = new char[80];
				sprintf( imgR_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", k, ".tif");
				char* featurepoint2 = new char[80];
				sprintf( featurepoint2, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", k, ".txt");
				if(i==4){
					CC_match2(imgL_path, imgR_path, 15, 0.95, 2, fs_c, k, featurepoint1, featurepoint2, outpoint1);
				}
				else{
					float* fs_c1 = new float[6];
					fs_c1[0]=fs_c[0];
					fs_c1[1]=fs_c[1];
					fs_c1[2]=fs_c[2]+fs_c[0]*mosaic_c1[j*4+0]+fs_c[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
					fs_c1[3]=fs_c[3];
					fs_c1[4]=fs_c[4];
					fs_c1[5]=fs_c[5]+fs_c[3]*mosaic_c1[j*4+0]+fs_c[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];
					//beginR、beginC 
					if(i==1&&j==1&&k==1){
						limit_match1(imgL_path, imgR_path, 15, 30*(5-i), 0.9, 2, fs_c1, k, featurepoint1, featurepoint2, outpoint1);
					}
					else if(i==0){
						limit_match1(imgL_path, imgR_path, 25, 70*(5-i), 0.9, 2, fs_c1, k, featurepoint1, featurepoint2, outpoint1);
					}
					else{
						limit_match1(imgL_path, imgR_path, 7*(5-i), 30*(5-i), 0.8, 2, fs_c1, k, featurepoint1, featurepoint2, outpoint1);
					}
				}
				int rem=remove(featurepoint1);
				int ren=rename(outpoint1,featurepoint1);
				//printf("%d %d\n",rem,ren);
			}
		}

		//i结束后，进行匹配点筛选和重排，加载拼接系数并计算fs系数。
		//仿射系数计算
		std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
		std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
		for(int j=0;j<CCD_num;j++){
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
			FILE *fp=fopen(featurepoint1,"r");
			int bj,row,col,imgID,mrow,mcol;
			float m_score;
			double sfr=pow(double(2),double(i))/pow(double(2),double(4));
			while(!feof(fp)){
				fscanf(fp,"%d ",&bj);
				if(bj==1){
					fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score);
					KeyPoint_x1.push_back(row+mosaic_c1[j*4+0]);
					KeyPoint_y1.push_back(col+mosaic_c1[j*4+2]);
					KeyPoint_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
					KeyPoint_y2.push_back(mcol+mosaic_c2[imgID*4+2]);
					KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
					KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
					KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[imgID*4+0])*sfr));
					KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[imgID*4+2])*sfr));
				}
				else{
					fscanf(fp,"%d %d\n",&row,&col);
				}
			}
			fclose(fp);
		}
		int * match = new int[KeyPoint_x1.size()];
		memset(match,-1,sizeof(int)*KeyPoint_x1.size());
		RANSAC_fs(KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,60,match,fs_c);  //剔除粗差

		char* imgL_path1 = new char[80];
		sprintf( imgL_path1, "%s%s%s%s%d%s%s", filepath, "\\", xulie_ID1, "\\downsample\\",0,"\\","mosaic_ds4.tif");
		char* imgR_path1 = new char[80];
		sprintf( imgR_path1, "%s%s%s%s%d%s%s", filepath, "\\", xulie_ID2, "\\downsample\\",0,"\\","mosaic_ds4.tif");
		drawMatch3(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,1);
		drawMatch2(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,1);

		KeyPoint_x1.swap(vector<int>());
		KeyPoint_y1.swap(vector<int>());
		KeyPoint_x2.swap(vector<int>());
		KeyPoint_y2.swap(vector<int>());
		KeyPoint_x11.swap(vector<int>());
		KeyPoint_y11.swap(vector<int>());
		KeyPoint_x22.swap(vector<int>());
		KeyPoint_y22.swap(vector<int>());

		//存储结果并跳出
		if(BJ==1){
			//std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
			//std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
			for(int j=0;j<CCD_num;j++){
				char* featurepoint1 = new char[80];
				sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
				char* featurepoint2 = new char[80];
				sprintf( featurepoint2, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_match.txt");
				FILE *fp=fopen(featurepoint1,"r");
				FILE *fp_re=fopen(featurepoint2,"w");
				int bj,row,col,imgID,mrow,mcol;
				float m_score;
				double sfr=pow(double(2),double(i))/pow(double(2),double(4));
				int count=0;
				while(!feof(fp)){
					fscanf(fp,"%d ",&bj);
					if(bj==1){
						fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score);
						KeyPoint_x1.push_back(row+mosaic_c1[j*4+0]);
						KeyPoint_y1.push_back(col+mosaic_c1[j*4+2]);
						KeyPoint_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
						KeyPoint_y2.push_back(mcol+mosaic_c2[imgID*4+2]);
						//KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
						//KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
						//KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[imgID*4+0])*sfr));
						//KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[imgID*4+2])*sfr));
						float vx=KeyPoint_x1[count]*fs_c[0]+KeyPoint_y1[count]*fs_c[1]+fs_c[2]-KeyPoint_x2[count];
						float vy=KeyPoint_x1[count]*fs_c[3]+KeyPoint_y1[count]*fs_c[4]+fs_c[5]-KeyPoint_y2[count];
						if(sqrt(vx*vx+vy*vy)<250 && abs(vx)<150 && abs(vy)<150){
							fprintf(fp_re,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,m_score);
						}
						count++;
					}
					else{
						fscanf(fp,"%d %d\n",&row,&col);
					}
				}
				fclose(fp);
				fclose(fp_re);
			}

			break;
		}

		KeyPoint_x1.swap(vector<int>());
		KeyPoint_y1.swap(vector<int>());
		KeyPoint_x2.swap(vector<int>());
		KeyPoint_y2.swap(vector<int>());
		KeyPoint_x11.swap(vector<int>());
		KeyPoint_y11.swap(vector<int>());
		KeyPoint_x22.swap(vector<int>());
		KeyPoint_y22.swap(vector<int>());
	}
	//根据低层次匹配关系，预测高层次匹配范围

	//将匹配范围对应到每张影像上去

	//遍历所有预测范围影像块，进行特征点提取，进行预测位置，判断是否在右影像块范围内，
	//是的话就进一步找预测位置附近的特征点，进行匹配。

	//匹配对进行输出


	//先对融合影像提取特征点
}


//剔除粗差
void Compute_fsc(char* filepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2,float* fs_c){
	printf("Compute_fsc begin!\n");
	int cols = 2048;
	int CCD_num=10;
	char* filepath1=filepath;

	//逐层匹配
	//float* fs_c = new float[6];
	//memset(fs_c,1.0,sizeof(float)*6);
	int CCD_id;
	int BJ=0;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char* ds_path_mosaictxt1 = new char[80];
		sprintf( ds_path_mosaictxt1, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		int* mosaic_c1=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]);
		}
		fclose(fpm1);
		char* ds_path_mosaictxt2 = new char[80];
		sprintf( ds_path_mosaictxt2, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		int* mosaic_c2=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]);
		}
		fclose(fpm2);


		//i结束后，进行匹配点筛选和重排，加载拼接系数并计算fs系数。
		//仿射系数计算
		std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
		std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
		for(int j=0;j<CCD_num;j++){
			char* featurepoint1 = new char[80];
			//sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_grid.txt");
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
			FILE *fp=fopen(featurepoint1,"r");
			int bj,row,col,imgID,mrow,mcol;
			float m_score;
			double sfr=pow(double(2),double(i))/pow(double(2),double(4));
			while(!feof(fp)){
				fscanf(fp,"%d ",&bj);
				if(bj==1){
					fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score);
					KeyPoint_x1.push_back(row+mosaic_c1[j*4+0]);
					KeyPoint_y1.push_back(col+mosaic_c1[j*4+2]);
					KeyPoint_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
					KeyPoint_y2.push_back(mcol+mosaic_c2[imgID*4+2]);
					KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
					KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
					KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[imgID*4+0])*sfr));
					KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[imgID*4+2])*sfr));
				}
				else{
					fscanf(fp,"%d %d\n",&row,&col);
				}
			}
			fclose(fp);
		}
		int * match = new int[KeyPoint_x1.size()];
		memset(match,-1,sizeof(int)*KeyPoint_x1.size());
		RANSAC_fs1(KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,30,match,fs_c);  //剔除粗差

		char* imgL_path1 = new char[80];
		sprintf( imgL_path1, "%s%s%s%s%d%s%s", filepath, "\\", xulie_ID1, "\\downsample\\",0,"\\","mosaic_ds4.tif");
		char* imgR_path1 = new char[80];
		sprintf( imgR_path1, "%s%s%s%s%d%s%s", filepath, "\\", xulie_ID2, "\\downsample\\",0,"\\","mosaic_ds4.tif");
		drawMatch3(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,1);
		drawMatch2(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,1);

		KeyPoint_x1.swap(vector<int>());
		KeyPoint_y1.swap(vector<int>());
		KeyPoint_x2.swap(vector<int>());
		KeyPoint_y2.swap(vector<int>());
		KeyPoint_x11.swap(vector<int>());
		KeyPoint_y11.swap(vector<int>());
		KeyPoint_x22.swap(vector<int>());
		KeyPoint_y22.swap(vector<int>());
	}
}

void Tichu_CX(char* filepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2,int mark, float* fs_c){
	printf("Tichu_CX begin!\n");
	int cols = 2048;
	int CCD_num=10;
	char* filepath1=filepath;

	//逐层匹配
	//float* fs_c = new float[6];
	//memset(fs_c,1.0,sizeof(float)*6);
	int CCD_id;
	int BJ=0;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char* ds_path_mosaictxt1 = new char[80];
		sprintf( ds_path_mosaictxt1, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		int* mosaic_c1=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]);
		}
		fclose(fpm1);
		char* ds_path_mosaictxt2 = new char[80];
		sprintf( ds_path_mosaictxt2, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		int* mosaic_c2=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]);
		}
		fclose(fpm2);


		std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
		std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
		//存储结果并跳出
		if(BJ==1){
			//std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
			//std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
			int CX_count=0;
			int M_count=0;
			for(int j=0;j<CCD_num;j++){
				char* featurepoint1 = new char[80];
				if(mark==0){
					sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
				}
				else if(mark==1){
					sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_grid.txt");
				}
				else{
					sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
				}
				char* featurepoint2 = new char[80];
				sprintf( featurepoint2, "%s%s%s%s%d%s%s%s%d%s", filepath1, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_match.txt");
				FILE *fp=fopen(featurepoint1,"r");
				FILE *fp_re=fopen(featurepoint2,"w");
				int bj,row,col,imgID,mrow,mcol;
				float m_score;
				double sfr=pow(double(2),double(i))/pow(double(2),double(4));
				int count=0;
				while(!feof(fp)){
					fscanf(fp,"%d ",&bj);
					if(bj==1){
						fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score);
						KeyPoint_x1.push_back(row+mosaic_c1[j*4+0]);
						KeyPoint_y1.push_back(col+mosaic_c1[j*4+2]);
						KeyPoint_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
						KeyPoint_y2.push_back(mcol+mosaic_c2[imgID*4+2]);


						float vx=KeyPoint_x1[count]*fs_c[0]+KeyPoint_y1[count]*fs_c[1]+fs_c[2]-KeyPoint_x2[count];
						float vy=KeyPoint_x1[count]*fs_c[3]+KeyPoint_y1[count]*fs_c[4]+fs_c[5]-KeyPoint_y2[count];
						M_count++;

						if(sqrt(vx*vx+vy*vy)<250 && abs(vx)<150 && abs(vy)<150){
							fprintf(fp_re,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,m_score);
						}
						else{
							CX_count++;
						}
						count++;
					}
					else{
						fscanf(fp,"%d %d\n",&row,&col);
					}
				}
				fclose(fp);
				fclose(fp_re);
			}
			printf("M_count:%d    CX_count:%d\n",M_count,CX_count);
			printf("Tichu_CX end!\n\n");
			break;
		}

		KeyPoint_x1.swap(vector<int>());
		KeyPoint_y1.swap(vector<int>());
		KeyPoint_x2.swap(vector<int>());
		KeyPoint_y2.swap(vector<int>());
		KeyPoint_x11.swap(vector<int>());
		KeyPoint_y11.swap(vector<int>());
		KeyPoint_x22.swap(vector<int>());
		KeyPoint_y22.swap(vector<int>());
	}
}

//生成网格点并匹配
void SemiDenseGrid_match(char* filepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2,float* fs_c){
	printf("SemiDenseGrid_match begin!\n");
	int cols = 2048;
	int CCD_num=10;
	char* filepath1=filepath;

	int CCD_id;
	int BJ=0;
	int batch_size=64;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char* ds_path_mosaictxt1 = new char[80];
		sprintf( ds_path_mosaictxt1, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		int* mosaic_c1=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]);
		}
		fclose(fpm1);
		char* ds_path_mosaictxt2 = new char[80];
		sprintf( ds_path_mosaictxt2, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		int* mosaic_c2=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]);
		}
		fclose(fpm2);


		//生成网格点并匹配
		for(int j=0;j<CCD_num;j++){
			printf("limit_grid:%d%s\n",j,"……");
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_grid.txt");

			char* outpoint1 = new char[80];

			/*
			//生成格网点
			FILE *fp_grid=fopen(featurepoint1,"w");
			for(int rr=0;(rr+1)*batch_size<=rows1;rr++){
				for(int cc=0;(cc+1)*batch_size<=cols;cc++){
					fprintf(fp_grid,"%d %d %d\n",0,rr*batch_size+batch_size/2,cc*batch_size+batch_size/2);
				}
			}
			fclose(fp_grid);

			char* outpoint1 = new char[80];
			sprintf( outpoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_.txt");
			for(int k=0;k<CCD_num;k++){
				printf("limit_grid:%d-%d%s\n",j,k,"……");
				char* imgR_path = new char[80];
				sprintf( imgR_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", k, ".tif");

				float* fs_c1 = new float[6];
				fs_c1[0]=fs_c[0];
				fs_c1[1]=fs_c[1];
				fs_c1[2]=fs_c[2]+fs_c[0]*mosaic_c1[j*4+0]+fs_c[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
				fs_c1[3]=fs_c[3];
				fs_c1[4]=fs_c[4];
				fs_c1[5]=fs_c[5]+fs_c[3]*mosaic_c1[j*4+0]+fs_c[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];
				//beginR、beginC 

				//limit_grid
				if(j<3){
					limit_grid(imgL_path, imgR_path, 15, 128, 0.7, fs_c1, k, featurepoint1, outpoint1);
				}
				else{
					limit_grid(imgL_path, imgR_path, 13, 32, 0.75, fs_c1, k, featurepoint1, outpoint1);
				}

				int rem=remove(featurepoint1);
				int ren=rename(outpoint1,featurepoint1);
				//printf("%d %d\n",rem,ren);
			}
			*/

			//单独处理CCD点连接点
			int imgID,row,col,mimgID,mrow,mcol;
			float mscore;
			if(j<CCD_num-1){
				sprintf(outpoint1, "%s%s%s%s%d%s%s%s%d%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",0,"\\",xulie_ID1, "_RED", j, "_", j+1, "_intra.txt");
				sprintf(featurepoint1, "%s%s%s%s%d%s%s%s%d%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",0,"\\",xulie_ID1, "_RED", j, "_", j+1, "_intra__.txt");
				FILE *fp=fopen(outpoint1,"r");
				FILE *fp1=fopen(featurepoint1,"w");
				while(!feof(fp)){
					fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
					fprintf(fp1,"%d %d %d\n",0,row,col);
				}
				fclose(fp);
				fclose(fp1);

				sprintf( outpoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",0,"\\",xulie_ID1, "_RED", j, "_.txt");
				for(int k=0;k<CCD_num;k++){
					char* imgR_path = new char[80];
					sprintf( imgR_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID2, "\\downsample\\",0,"\\",xulie_ID2, "_RED", k, ".tif");

					float* fs_c1 = new float[6];
					fs_c1[0]=fs_c[0];
					fs_c1[1]=fs_c[1];
					fs_c1[2]=fs_c[2]+fs_c[0]*mosaic_c1[j*4+0]+fs_c[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
					fs_c1[3]=fs_c[3];
					fs_c1[4]=fs_c[4];
					fs_c1[5]=fs_c[5]+fs_c[3]*mosaic_c1[j*4+0]+fs_c[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];

					//limit_grid
					if(j<3){
						limit_grid(imgL_path, imgR_path, 15, 128, 0.7, fs_c1, k, featurepoint1, outpoint1);
					}
					else{
						limit_grid(imgL_path, imgR_path, 13, 32, 0.75, fs_c1, k, featurepoint1, outpoint1);
					}

					int rem=remove(featurepoint1);
					int ren=rename(outpoint1,featurepoint1);
					//printf("%d %d\n",rem,ren);
				}

			}
		}
		//i结束后，进行匹配点筛选和重排，加载拼接系数并计算fs系数。
		//仿射系数计算(由TichuCX代替)
	}
	printf("SemiDenseGrid_match end!\n");
}

//局部仿射变换约束格网点匹配
void LocalLimit_SemiDenseGrid_match(char* filepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2){
	printf("SemiDenseGrid_match begin!\n");
	int cols = 2048;
	int CCD_num=10;
	char* filepath1=filepath;

	int CCD_id;
	int BJ=0;
	int batch_size=128;

	float* fs_c = new float[6];
	memset(fs_c,1.0,sizeof(float)*6);
	int rowbatch=2000;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char* ds_path_mosaictxt1 = new char[80];
		sprintf( ds_path_mosaictxt1, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		int* mosaic_c1=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]);
		}
		fclose(fpm1);
		char* ds_path_mosaictxt2 = new char[80];
		sprintf( ds_path_mosaictxt2, "%s%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",i,"\\mosaic.txt");
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		int* mosaic_c2=new int[CCD_num*4];
		for(int ii=0;ii<CCD_num;ii++){
			fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]);
		}
		fclose(fpm2);


		//生成网格点并匹配
		for(int j=0;j<CCD_num;j++){
			printf("limit_grid:%d%s\n",j,"……");
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath, "\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".tif");
			char* featurepoint = new char[80];
			sprintf( featurepoint, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".txt");

			FILE *fp_feap=fopen(featurepoint,"r");

			//分块生成网格点
			int bj,row,col,imgID,mrow,mcol;
			float m_score;
			std::vector<int> match_feap_x1,match_feap_y1,match_feap_x2,match_feap_y2;
			for(int jj=0;jj<rows1/rowbatch;jj++){

				//读取分块范围内的特征点
				while(!feof(fp_feap)){
					fscanf(fp_feap,"%d ",&bj);
					if(bj==1){
						fscanf(fp_feap,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score);
						if(row<(jj+1)*rowbatch){
							match_feap_x1.push_back(row+mosaic_c1[j*4+0]);
							match_feap_y1.push_back(col+mosaic_c1[j*4+2]);
							match_feap_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
							match_feap_y2.push_back(mcol+mosaic_c2[imgID*4+2]);
						}
						else{
							break;
						}
					}
					else{
						fscanf(fp_feap,"%d %d\n",&row,&col);
					}
				}

				int * match = new int[match_feap_x1.size()];
				memset(match,-1,sizeof(int)*match_feap_x1.size());
				RANSAC_fs2(match_feap_x1,match_feap_y1,match_feap_x2,match_feap_y2,30,1000,match,fs_c);  //计算局部仿射系数

				//生成局部网格点并进行约束匹配
				char* temp_grid = new char[80];
				sprintf( temp_grid, "%s%s%s%s%d%s%s%s%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", "_temp.txt");
				FILE *fp_temp=fopen(temp_grid,"w");
				std::vector<int> local_grid_x1,local_grid_y1,local_grid_x2,local_grid_y2;
				for(int rr=jj*rowbatch;rr<(jj+1)*rowbatch;rr+=batch_size){
					for(int cc=0;cc<cols;cc+=batch_size){
						if(rr+batch_size<rows1 && cc+batch_size<cols){ 
							fprintf(fp_temp,"%d %d %d\n",0,rr+batch_size/2,cc+batch_size/2);
						}
					}
				}
				fclose(fp_temp);


				//约束匹配
				char* outtemp = new char[80];
				sprintf( outtemp, "%s%s%s%s%d%s%s%s%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", "_temp_.txt");
				for(int k=0;k<CCD_num;k++){
					char* imgR_path = new char[80];
					sprintf( imgR_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", k, ".tif");

					float* fs_c1 = new float[6];
					fs_c1[0]=fs_c[0];
					fs_c1[1]=fs_c[1];
					fs_c1[2]=fs_c[2]+fs_c[0]*mosaic_c1[j*4+0]+fs_c[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
					fs_c1[3]=fs_c[3];
					fs_c1[4]=fs_c[4];
					fs_c1[5]=fs_c[5]+fs_c[3]*mosaic_c1[j*4+0]+fs_c[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];

					limit_grid(imgL_path, imgR_path, 17, 32, 0.5, fs_c1, k, temp_grid, outtemp);

					int rem=remove(temp_grid);
					int ren=rename(outtemp,temp_grid);
				}


				//存储匹配网格点
				char* gridmatch = new char[80];
				sprintf(gridmatch, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, "_match.txt");
				FILE *fp_grid;
				if(jj==0){
					fp_grid=fopen(gridmatch,"w");
				}
				else{
					fp_grid=fopen(gridmatch,"a");
				}

				fp_temp=fopen(temp_grid,"r");
				while(!feof(fp_temp)){
					fscanf(fp_temp,"%d ",&bj);
					if(bj==1){
						fscanf(fp_temp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score);
						fprintf(fp_grid,"%d %d %d %d %d %d %f\n",1,row,col,imgID,mrow,mcol,m_score);
					}
					else{
						fscanf(fp_temp,"%d %d\n",&row,&col);
					}
				}
				fclose(fp_grid);
				fclose(fp_temp);
			}
		}
	}
	printf("SemiDenseGrid_match end!\n");
}

//CCD间连接点
void Generate_matchPoint_Between_CCD(char* filepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2){
	int cols = 2048;
	int CCD_num=10;
	char* filepath1=filepath;

	for(int j=0;j<CCD_num-1;j++){
		char* imgL_path = new char[80];
		sprintf( imgL_path, "%s%s%s%s%s%s%s%d%s", filepath, "\\", xulie_ID1, "\\downsample\\","0\\",xulie_ID1, "_RED", j, ".tif");
		char* imgR_path = new char[80];
		sprintf( imgR_path, "%s%s%s%s%s%s%s%d%s", filepath, "\\", xulie_ID1, "\\downsample\\","0\\",xulie_ID1, "_RED", j+1, ".tif");
		char* outpoint1 = new char[80];
		sprintf(outpoint1, "%s%s%s%s%d%s%s%s%d%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\",0,"\\",xulie_ID1, "_RED", j, "_", j+1, "_intra.txt");

		intra_CCD_match(imgL_path,imgR_path,15,10,0.9,10,j,outpoint1);
	}

	for(int j=0;j<CCD_num-1;j++){
		char* imgL_path = new char[80];
		sprintf( imgL_path, "%s%s%s%s%s%s%s%d%s", filepath, "\\", xulie_ID2, "\\downsample\\","0\\",xulie_ID2, "_RED", j, ".tif");
		char* imgR_path = new char[80];
		sprintf( imgR_path, "%s%s%s%s%s%s%s%d%s", filepath, "\\", xulie_ID2, "\\downsample\\","0\\",xulie_ID2, "_RED", j+1, ".tif");
		char* outpoint1 = new char[80];
		sprintf(outpoint1, "%s%s%s%s%d%s%s%s%d%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\",0,"\\",xulie_ID2, "_RED", j, "_", j+1, "_intra.txt");

		intra_CCD_match(imgL_path,imgR_path,15,10,0.85,10,j,outpoint1);
	}/**/
}


//基于约束逐级（从低分到高分）匹配：①生成仿射约束 ②存储匹配点对
void Zhuji_match(char* filepath,char* xulie_ID1,char* xulie_ID2){
	float* fs_c = new float[6];
	memset(fs_c,1.0,sizeof(float)*6);
	for(int i=4;i>0;i--){
		char* imagepath1 = new char[80];
		sprintf( imagepath1, "%s%s%s%d%s", filepath, xulie_ID1,"\\mosaic\\mosaic_ds", i, ".tif");
		char* imagepath2 = new char[80];
		sprintf( imagepath2, "%s%s%s%d%s", filepath, xulie_ID2,"\\mosaic\\mosaic_ds", i, ".tif");


		if(i==4){
			char* matchpointxt = new char[80];
			sprintf( matchpointxt, "%s%s%s%d%s", filepath, xulie_ID1,"\\mosaic\\mosaic_ds", i, ".txt");
			CC_match1(imagepath1, imagepath2, 13, 0.95, 2, fs_c,matchpointxt);
		}
		else{
			fs_c[2]=fs_c[2]*2;
			fs_c[5]=fs_c[5]*2;
			char* matchpointxt = new char[80];
			sprintf( matchpointxt, "%s%s%s%d%s", filepath, xulie_ID1,"\\mosaic\\mosaic_ds", i, ".txt");
			limit_match(imagepath1, imagepath2, 13, 0.95, 2, fs_c, matchpointxt);
		}
	}
	fs_c[2]=fs_c[2]*2;
	fs_c[5]=fs_c[5]*2;
	char* imagepath1 = new char[80];
	sprintf( imagepath1, "%s%s%s", filepath, xulie_ID1,"\\mosaic\\mosaic1.tif");
	char* imagepath2 = new char[80];
	sprintf( imagepath2, "%s%s%s", filepath, xulie_ID2,"\\mosaic\\mosaic1.tif");
	char* matchpointxt = new char[80];
	sprintf( matchpointxt, "%s%s%s", filepath, xulie_ID1,"\\mosaic\\mosaic.txt");
	limit_match(imagepath1, imagepath2, 13, 0.95, 2, fs_c, matchpointxt);
}

void Img_Preprcess(char* filepath,char* xulie_ID){
	//影像预处理
	char* src_path = new char[80];
	sprintf( src_path, "%s%s%s", filepath, xulie_ID, "\\src");

	char* prefix = new char[80];
	sprintf( prefix, "%s%s", xulie_ID, "_RED");
	//①降采样
	char* downsample_path = new char[80];
	sprintf( downsample_path, "%s%s%s", filepath, xulie_ID, "\\downsample");
	int status = mkdir(downsample_path); 
	xulie_downsample(src_path, prefix,downsample_path);

	//②拼接
	char* mosaic_path0 = new char[80];
	sprintf( mosaic_path0, "%s%s%s", filepath, xulie_ID, "\\mosaic");
	status = mkdir(mosaic_path0); 
	//(!status) ? (printf("Directory created\n")) : (printf("Unable to create directory\n")); 

	int OverlapSamples=48;
	xulie_mosaic(src_path, prefix, OverlapSamples, mosaic_path0);

	//③拼接影像降采样
	char* mosaic_path2 = new char[80];
	sprintf( mosaic_path2, "%s%s", mosaic_path0, "\\mosaic1.tif");

	for(int j=1;j<5;j++){
		char* ds_path = new char[80];
		sprintf( ds_path, "%s%s%d%s", mosaic_path0, "\\mosaic_ds", j, ".tif");

		//降采样
		int batchsize=pow(double(2),double(j));
		char* img_path = new char[80];
		Down_sample(mosaic_path2, batchsize, ds_path);
		delete []img_path;
		delete []ds_path;
	}
}

//物方匹配
void ground_match(char* filepath,char* xulie_ID1,char* xulie_ID2){
	char* filepath1=filepath;
	int CCD_num=10;

	//输出采样外方位元素进行拟合
	char* outeopath = new char[80];
	char* eopzpath = new char[80];
	sprintf( outeopath, "%s%s%s", "..\\data\\EO\\", xulie_ID1, ".txt");
	sprintf( eopzpath, "%s%s%s", "..\\data\\EO\\", xulie_ID1, "_pz.txt");
	OutEO2txt(outeopath, eopzpath);
	sprintf( outeopath, "%s%s%s", "..\\data\\EO\\", xulie_ID2, ".txt");
	sprintf( eopzpath, "%s%s%s", "..\\data\\EO\\", xulie_ID2, "_pz.txt");
	OutEO2txt(outeopath, eopzpath);

	//拟合外方位元素并存储
	double* Poly_C = new double[30];
	char* poly_path = new char[80];

	sprintf( outeopath, "%s%s%s%s%s", "..\\data\\EO\\", xulie_ID1,"\\", xulie_ID1,"_RED5_0.txt");
	sprintf( poly_path, "%s%s%s", "..\\data\\EO\\", xulie_ID1, "\\polyCC.txt");
	Polynomial3_EO(outeopath,Poly_C,poly_path);

	sprintf( outeopath, "%s%s%s%s%s", "..\\data\\EO\\", xulie_ID2,"\\", xulie_ID2,"_RED5_0.txt");
	sprintf( poly_path, "%s%s%s", "..\\data\\EO\\", xulie_ID2, "\\polyCC.txt");
	Polynomial3_EO(outeopath,Poly_C,poly_path);

	//预测左影像对应的地面点
	//float Z;
	//LoadDEM("E:\\Mars_VS\\Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",10,10,&Z);

	float* EO = new float[6];
	float* IO = new float[10];
	float* GC = new float[3];
	//double* Poly_C = new double[30];

	///////////////////////////////////////////////////////
	//计算每个兴趣点的地面坐标
	int ch=6;
	int Localmax_win[5]={513,129,65,49,37};
	for(int i=4;i>=4;i--){
		FILE *fp_f,*fp_eo,*fp_io,*fp_poly;

		//提取EO拟合参数
		//char* poly_path = new char[80];
		sprintf( poly_path, "%s%s%s", "..\\data\\EO\\", xulie_ID1, "\\polyCC.txt");
		fp_poly = fopen(poly_path, "r");
		for(int i=0;i<6;i++){
			for(int j=0;j<5;j++){
				fscanf(fp_poly,"%lf ",&Poly_C[i*5+j]);
			}
			fscanf(fp_poly,"\n");
		}
		fclose(fp_poly);

		//读取内方位参数
		fp_io = fopen("..\\data\\IO\\IO.txt", "r");

		//提取特征点并求出其对应地面坐标
		for(int j=0;j<CCD_num;j++){
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID1, "\\downsample\\",i,"\\",xulie_ID1, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID1, "\\downsample\\ground\\",i,"\\",xulie_ID1, "_RED", j, ".txt");
			std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
			Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			//读取对应的内方位元素
			int temp;
			float temp1;
			for(int k=0;k<11;k++){
				if(k==0){
					fscanf(fp_io,"%d ",&temp);
				}
				else if(k==8||k==9){
					fscanf(fp_io,"%e ",&IO[k-1]);
				}
				else{
					fscanf(fp_io,"%f ",&IO[k-1]);
				}
			}
			fscanf(fp_io,"\n");

			//读取对应CCD影像的起始采样时间
			double beginET,LR;
			char* EOpath = new char[80];
			sprintf( EOpath, "%s%s%s%s%s%d%s", "..\\data\\EO\\",xulie_ID1,"\\",xulie_ID1, "_RED", j, "_0.txt");
			fp_eo=fopen(EOpath,"r");
			fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
			fclose(fp_eo);

			fp_f=fopen(featurepoint1,"w");
			for(int ii=0;ii<KeyPoint_x1.size();ii++){
				KeyPoint_x1[ii]=KeyPoint_x1[ii]*16;
				KeyPoint_y1[ii]=KeyPoint_y1[ii]*16;
				double et=beginET+KeyPoint_x1[ii]*1*LR;
				Get_PolyEO(et,Poly_C,EO);
				//Get_PolyEO1(et,KeyPoint_x1[ii],Poly_C,EO);
				Get_groundtruth("E:\\Mars_VS\\Mars_MGS_MOLA_DEM_mosaic_global_463m.tif", KeyPoint_y1[ii], KeyPoint_x1[ii], 1, 128, 0, EO, IO, GC);
				fprintf(fp_f,"%d %d %d %f %f %f\n",0,KeyPoint_x1[ii],KeyPoint_y1[ii],GC[0],GC[1],GC[2]);
			}
			KeyPoint_x1.swap(vector<int>());
			KeyPoint_y1.swap(vector<int>());
			fclose(fp_f);
		}


		rewind(fp_io);
		//提取EO拟合参数
		sprintf( poly_path, "%s%s%s", "..\\data\\EO\\", xulie_ID2, "\\polyCC.txt");
		fp_poly = fopen(poly_path, "r");
		for(int i=0;i<6;i++){
			for(int j=0;j<5;j++){
				fscanf(fp_poly,"%lf ",&Poly_C[i*5+j]);
			}
			fscanf(fp_poly,"\n");
		}
		fclose(fp_poly);
		for(int j=0;j<CCD_num;j++){
			char* imgL_path = new char[80];
			sprintf( imgL_path, "%s%s%s%s%d%s%s%s%d%s", filepath,"\\", xulie_ID2, "\\downsample\\",i,"\\",xulie_ID2, "_RED", j, ".tif");
			char* featurepoint1 = new char[80];
			sprintf( featurepoint1, "%s%s%s%s%d%s%s%s%d%s", filepath1,"\\", xulie_ID2, "\\downsample\\ground\\",i,"\\",xulie_ID2, "_RED", j, ".txt");

			std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
			Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			//读取对应的内方位元素
			int temp;
			float temp1;
			for(int k=0;k<11;k++){
				if(k==0){
					fscanf(fp_io,"%d ",&temp);
				}
				else if(k==8||k==9){
					fscanf(fp_io,"%e ",&IO[k-1]);
				}
				else{
					fscanf(fp_io,"%f ",&IO[k-1]);
				}
			}
			fscanf(fp_io,"\n");

			//读取对应CCD影像的起始采样时间
			double beginET,LR;
			char* EOpath = new char[80];
			sprintf( EOpath, "%s%s%s%s%s%d%s", "..\\data\\EO\\",xulie_ID2,"\\",xulie_ID2, "_RED", j, "_0.txt");
			//sprintf( EOpath, "%s%d%s", "..\\data\\EO\\1650\\PSP_001777_1650_RED", j, "_0.txt");
			fp_eo=fopen(EOpath,"r");
			fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
			fclose(fp_eo);

			fp_f=fopen(featurepoint1,"w");
			for(int ii=0;ii<KeyPoint_x1.size();ii++){
				KeyPoint_x1[ii]=KeyPoint_x1[ii]*16;
				KeyPoint_y1[ii]=KeyPoint_y1[ii]*16;
				double et=beginET+KeyPoint_x1[ii]*1*LR;
				Get_PolyEO(et,Poly_C,EO);
				//Get_PolyEO1(et,KeyPoint_x1[ii],Poly_C,EO);
				Get_groundtruth("E:\\Mars_VS\\Mars_MGS_MOLA_DEM_mosaic_global_463m.tif", KeyPoint_y1[ii], KeyPoint_x1[ii], 1, 128, 4000, EO, IO, GC);
				fprintf(fp_f,"%d %d %d %f %f %f\n",0,KeyPoint_x1[ii],KeyPoint_y1[ii],GC[0],GC[1],GC[2]);
			}
			KeyPoint_x1.swap(vector<int>());
			KeyPoint_y1.swap(vector<int>());
			fclose(fp_f);
		}
		fclose(fp_io);
	}


	//Get_PolyEO(et,Poly_C,EO);
	Get_groundtruth("E:\\Mars_VS\\Mars_MGS_MOLA_DEM_mosaic_global_463m.tif", 0, 0, 1, 128, 0, EO, IO, GC);
}

void main_process(){
	char* filepath="E:\\Mars_VS\\data\\";
	char* xulie_ID1="PSP_001777_1650";
	char* xulie_ID2="PSP_001513_1655";

	//Img_Preprcess(filepath,xulie_ID1);
	//Img_Preprcess(filepath,xulie_ID2);

	//Zhuji_match(filepath, xulie_ID1, xulie_ID2);
	fenfu_match(filepath, xulie_ID1, xulie_ID2,40000,80000);
}