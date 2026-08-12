#include "mygdal.h"
#include "SpiceUsr.h"


bool WriteImageData(const char* strDestFilePath,unsigned char* pImageData,int nWidth,int nHeight,int nChannels,int nNewChannels)
{
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8","NO");

	GDALAllRegister();

	//QString strType;
	//GetDriverType(strDestFilePath,strType);

	GDALDriver *pMemDriver = NULL;
	pMemDriver = GetGDALDriverManager()->GetDriverByName("MEM");
	if( pMemDriver == NULL ) { return false; }

	GDALDataset * pMemDataSet = pMemDriver->Create("",nWidth,nHeight,nNewChannels,GDT_Byte,NULL);
	GDALRasterBand *pBand = NULL;

	for (int i = 1; i <= nNewChannels; i++)
	{
		if (i==nNewChannels && nNewChannels >nChannels)
		{
			unsigned char *pTempImageData = new unsigned char[nNewChannels*nWidth*nHeight];
			memset(pTempImageData,0xFE,nNewChannels*nWidth*nHeight);

			pBand = pMemDataSet->GetRasterBand(i);
			pBand->RasterIO(GF_Write, 
				0, 
				0, 
				nWidth, 
				nHeight, 
				pTempImageData, 
				nWidth, 
				nHeight, 
				GDT_Byte, 
				nNewChannels, 
				0); 
		}
		else
		{
			pBand = pMemDataSet->GetRasterBand(i);
			pBand->RasterIO(GF_Write, 
				0, 
				0, 
				nWidth, 
				nHeight, 
				pImageData+i-1 , 
				nWidth, 
				nHeight, 
				GDT_Byte, 
				nChannels, 
				0); 
		}
	}

	GDALDriver *pDstDriver = NULL;
	//pDstDriver = (GDALDriver *)GDALGetDriverByName(strType.toStdString().c_str());
	pDstDriver = GetGDALDriverManager()->GetDriverByName("GTIFF");
	if (pDstDriver == NULL)
	{ 
		GDALClose(pMemDataSet); 

		return false;
	}

	GDALDataset * pDataSet = pDstDriver->CreateCopy(strDestFilePath,pMemDataSet,FALSE, NULL, NULL, NULL);
	if (pDataSet == NULL)
	{
		GDALClose(pMemDataSet); 

		return false;
	}

	GDALClose(pDataSet); 
	GDALClose(pMemDataSet); 
	return true;
}

//常规操作
void jichu(){
	//注册文件格式
	GDALAllRegister();

	const char* pszFile = "E:\\Mars_VS\\Mars_MGS_MOLA_DEM_mosaic_global_463m.tif";
	GDALDataset *poDataset;
	//使用只读方式打开图像
	poDataset = (GDALDataset*) GDALOpen( pszFile,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File: %s不能打开！\n",pszFile);
		return;
	}

	//输出图像的格式信息

	printf( "Driver:%s/%s\n",
		poDataset->GetDriver()->GetDescription(),
		poDataset->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME) );

	//输出图像的大小和波段个数
	printf( "Size is%dx%dx%d\n",
		poDataset->GetRasterXSize(),poDataset->GetRasterYSize(),
		poDataset->GetRasterCount());

	//输出图像的投影信息
	if( poDataset->GetProjectionRef() != NULL )
		printf( "Projectionis `%s'\n", poDataset->GetProjectionRef() );

		//if( poDataset != NULL )
		//printf( "Projectionis `%s'\n", poDataset->GetProjectionRef() );

	OGRSpatialReference fRef;
	char *tmp = NULL;
	const char *projRef =poDataset->GetProjectionRef();
	/** 获得projRef的一份拷贝 **/
	/** 由于projRef是const char*,下面的一个函数不接受，所以需要转换成非const **/
	tmp = (char *)malloc(strlen(projRef) + 1);
	strcpy(tmp, projRef);


	/** 设置原始的坐标参数，和test.tif一致 **/
	fRef.importFromWkt(&tmp);
	/** 设置转换后的坐标 **/
	OGRSpatialReference *tRef;
	tRef=fRef.CloneGeogCS();  //只复制出地理坐标那部分
	//tRef->SetWellKnownGeogCS("WGS84");

	/** 下面进行坐标转换，到此为止都不需要proj，但是下面的内容如果不安装proj将会无法编译 **/
	double x=916925.21;double y=9502679.47;double z=1000;
	OGRCoordinateTransformation *coordTrans;
	coordTrans = OGRCreateCoordinateTransformation(&fRef, tRef);
	coordTrans->Transform(1, &x, &y, &z);
	printf("lat:%lf lon:%lf H:%lf\n",x,y,z);

	//输出图像的坐标和分辨率信息
	double adfGeoTransform[6];
	if( poDataset->GetGeoTransform(adfGeoTransform) == CE_None )
	{
		printf( "Origin =(%.6f,%.6f)\n",
			adfGeoTransform[0], adfGeoTransform[3]);

		printf( "PixelSize = (%.6f,%.6f)\n",
			adfGeoTransform[1], adfGeoTransform[5]);
	}

	GDALRasterBand *poBand;
	int            nBlockXSize, nBlockYSize;
	int            bGotMin, bGotMax;
	double         adfMinMax[2];

	//读取第一个波段
	poBand = poDataset->GetRasterBand( 1 );

	//获取图像的块大小并输出
	poBand->GetBlockSize(&nBlockXSize, &nBlockYSize );
	printf( "Block=%dx%dType=%s, ColorInterp=%s\n",
		nBlockXSize, nBlockYSize,
		GDALGetDataTypeName(poBand->GetRasterDataType()),
		GDALGetColorInterpretationName(
		poBand->GetColorInterpretation()));

	//获取该波段的最大值最小值，如果获取失败，则进行统计
	adfMinMax[0] = poBand->GetMinimum( &bGotMin);
	adfMinMax[1] = poBand->GetMaximum( &bGotMax);

	if( ! (bGotMin&& bGotMax) )
		GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);

	printf( "Min=%.3fd,Max=%.3f\n", adfMinMax[0], adfMinMax[1] );

	//输出图像的金字塔信息
	if( poBand->GetOverviewCount() > 0 )
		printf( "Band has%d overviews.\n", poBand->GetOverviewCount() );

	//输出图像的颜色表信息
	if( poBand->GetColorTable() != NULL)
		printf( "Band hasa color table with %d entries.\n",
		poBand->GetColorTable()->GetColorEntryCount() );

	float *pafScanline;
	int   nXSize = poBand->GetXSize();

	//读取图像的第一行数据
	pafScanline = (float*) CPLMalloc(sizeof(float)*nXSize);
	poBand->RasterIO(GF_Read, 0, 0, nXSize,1, 
		pafScanline, nXSize,1, GDT_Float32, 0, 0 );//GDT_Float32

	int m=*(pafScanline+34);
	int n=GDALGetDataTypeSize(GDT_CInt16);



	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	const char* dstPath = "D:\\dst.tif";
	int bufWidth = 200;
	int bufHeight = 500;
	GDALDataset* dst = pDriver->Create(dstPath, bufWidth, bufHeight, 1, GDT_Byte, ppszOptions);
	if (dst == nullptr)
	{
		printf("Can't Write Image!");
		return;
	}


	CPLFree(pafScanline);//记得要释放空间

	//关闭文件
	GDALClose((GDALDatasetH)poDataset);
}

//2%拉伸
void Linear2(char* pszSrcFile, char* pszDstFile){

	GDALDataset *dataSet = (GDALDataset*)GDALOpen(pszSrcFile,GA_ReadOnly);//输入文件路径自己改

	if( dataSet == NULL )
	{
		printf( "File: %s不能打开！\n",pszSrcFile);
		return;
	}

	int width = dataSet->GetRasterXSize();
	int height = dataSet->GetRasterYSize();
	int channels = dataSet->GetRasterCount();
	GDALDataType dataType = dataSet->GetRasterBand(1)->GetRasterDataType();

	//创建输出对象
	GDALDriver *poDriver = (GDALDriver*)GDALGetDriverByName("GTiff");
	GDALDataset *poDstDS = poDriver->Create(pszDstFile,width,height,channels,dataType,NULL);//输出文件路径自己改
	for(int i=1;i<=channels;i++){
		//统计直方图
		unsigned long long HistBand1[256] = {0};

		//累计直方图
		float HistR[256] = {0};

		GDALRasterBand *poBand1;
		GDALRasterBand *pDstband1 = poDstDS->GetRasterBand(i);

		//R波段累计直方图
		poBand1 = dataSet->GetRasterBand(i);
		int pmax=poBand1->GetMaximum();
		int pmin=poBand1->GetMinimum();
		poBand1->GetHistogram(-0.5,255.5,256, HistBand1,FALSE,FALSE,NULL,NULL);
		HistR[0] = (float)HistBand1[0]/(width*height);
		for(int i=1;i<=255;i++)
		{
			HistR[i] = HistR[i-1] + (float)HistBand1[i]/(width*height);
		}

		unsigned char *pBuf1 = new unsigned char[width*height];
		unsigned char *BufR = new unsigned char[width*height];
		poBand1->RasterIO(GF_Read,0,0,width,height,pBuf1,width,height,GDT_Byte,1,0);

		//R波段赋值
		int minR,maxR;//2%处的灰度值和98%处的灰度值
		minR=0;maxR=255;
		for(int i=0;i<=255;i++)
		{
			if(HistR[i]<=0.015)
				minR = i;
			if(HistR[i]<=0.985)
				maxR = i;
		}
		//minR=minR+(0.02-HistR[minR])/(HistR[minR+1]-HistR[minR]);
		//maxR=maxR+(0.98-HistR[maxR])/(HistR[maxR+1]-HistR[maxR]);

		int count1=0;int count2=0;int count3=0;
		int Gmin=0;//minR/2;
		int Gmax=255;//(255-maxR)/2+maxR;
		for(int i=0;i<width*height;i++)
		{
			if(pBuf1[i]<=minR){
				//BufR[i] = (Gmin - 0)/(minR - 0)*(pBuf1[i] - 0) + 0;
				BufR[i] = 0;
				count1+=1;
			}
			else if(pBuf1[i]>=maxR){
				//BufR[i] = (255 - Gmax)/(255 - maxR)*(pBuf1[i]-maxR) + Gmax;
				BufR[i] = 255;
				count2+=1;
			}
			else{
				BufR[i] = (Gmax - Gmin)/(maxR - minR)*(pBuf1[i]-minR) + Gmin;
				//BufR[i] = 255/(maxR - minR)*(pBuf1[i] - minR);
				count3+=1;
			}
		}
		pDstband1->RasterIO(GF_Write,0,0,width,height,BufR,width,height,GDT_Byte,1,0);
		delete [] pBuf1;
		delete [] BufR;
	}
	GDALClose((GDALDatasetH)poDstDS);
	GDALClose((GDALDatasetH)dataSet);
}

//普通灰度线性拉伸
void Linear(uchar *ImageMat, int rows, int cols, uchar *ImageResult){
	float maxI=-10000;
	int minI=10000;
	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			if(ImageMat[j*cols+i]<minI)
				minI = ImageMat[j*cols+i];
			if(ImageMat[j*cols+i]>maxI)
				maxI = ImageMat[j*cols+i];
		}
	}

	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			ImageResult[j*cols+i] = uchar((float(ImageMat[j*cols+i])-minI)/(maxI-minI)*255);
		}
	}
}

//二进制文件读取
//影像路径；影像行；影像列；每行字节数；每行偏移量；文件头字节数；输出影像中心坐标（位于原始影像）；输出影像行列；输出影像data
void LoadData(char* data_fname,int rows,int cols, int dsr_size, int PY, int Initial_JD, int Center_row, int Center_col, int ImageR, int ImageC, uchar *ImageData){
	FILE *file_id = fopen(data_fname, "r");
	if( file_id == NULL ){
		printf("Cannot open file, press any key to exit!\n");
		return;
	}

	uchar *ImageMat = new uchar[ImageR*ImageC];
	uchar *rowdata = new uchar[ImageC];
	memset(ImageMat,0,sizeof(uchar)*ImageC*ImageR);
	memset(rowdata,0,sizeof(uchar)*ImageC);
	int count=0;

	for(int i=Center_row-ImageR/2;i<Center_row+ImageR/2;i++){
		int jump_distance = Initial_JD + dsr_size*i + PY + (Center_col-ImageC/2);
		fseek(file_id, jump_distance * sizeof(uchar), 0);
		if((feof(file_id)==0) && (fread(rowdata, sizeof(uchar), ImageC, file_id))){
			//memcpy(ImageMat,rowdata,sizeof(uchar)*ImageC);
			for(int ii=0;ii<ImageC;ii++){
				ImageMat[count*ImageC+ii]=rowdata[ii];
			}
			count += 1;
		}
	}
	memcpy(ImageData,ImageMat,sizeof(uchar)*ImageC*ImageR);
	delete []ImageMat;
	delete []rowdata;
}

//影像列间灰度均衡化
void Col_balance(uchar *ImageMat, int rows, int cols, uchar *ImageResult){
	float* meanCol = new float[cols];
	memset(meanCol,0,sizeof(float)*cols);
	float meanImage = 0;
	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			meanCol[i] += float(ImageMat[j*cols+i])/rows;
		}
		meanImage += meanCol[i]/cols;
	}
	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			ImageResult[j*cols+i] = uchar(float(ImageMat[j*cols+i])*meanImage/meanCol[i]+0.5);
		}
	}
	delete []meanCol;
}
void Col_balance1(uchar *ImageMat, int rows, int cols, uchar *ImageResult){
	float* meanCol = new float[cols];
	memset(meanCol,0,sizeof(float)*cols);
	float meanImage = 0;
	for(int i=0;i<cols;i++){
		for(int j=0;j<rows;j++){
			meanCol[i] += float(ImageMat[j*cols+i])/rows;
		}
		meanImage += meanCol[i]/cols;
	}

	//计算列间梯度，及其均值、方差
	float* Col_gradient = new float[cols-2];
	memset(Col_gradient,0,sizeof(float)*(cols-2));
	float mean_gradient = 0;
	for(int i=0;i<cols-2;i++){
		Col_gradient[i]=(abs(meanCol[i+2]-meanCol[i+1])+abs(meanCol[i+1]-meanCol[i]))/2;
		mean_gradient += Col_gradient[i]/(cols-2);
	}
	float sigma_gradient = 0;
	for(int i=0;i<cols-2;i++){
		sigma_gradient += (Col_gradient[i]-mean_gradient)*(Col_gradient[i]-mean_gradient)/(cols-2);
	}
	sigma_gradient = sqrt(sigma_gradient);

	//对梯度变化异常的列进行处理
	memcpy(ImageMat,ImageResult,sizeof(uchar)*rows*cols);
	for(int i=1;i<cols-1;i++){
		if(abs(mean_gradient-Col_gradient[i-1])>2*sigma_gradient){
			for(int j=0;j<rows;j++){
				ImageResult[j*cols+i] = uchar(float(ImageMat[j*cols+i])*(meanCol[i-10])/meanCol[i]+0.5);
			}
		}
	}
	delete []meanCol;
	delete []Col_gradient;
}
void Col_balance2(char *img_path,char* cb_path){
	GDALAllRegister();
	GDALDataType Type0 = GDT_Byte;

	GDALDataset *poDataset;
	GDALRasterBand *poBand;
	int rows0,cols0;

	poDataset = (GDALDataset*) GDALOpen( img_path,GA_ReadOnly );
	if( poDataset == NULL || poDataset->GetRasterCount()<1)
	{
		printf( "File1: %s不能打开！\n",img_path);
		return;
	}

	poBand = poDataset->GetRasterBand(1);
	Type0 = poBand->GetRasterDataType();
	cols0 = poBand->GetXSize();
	rows0 = poBand->GetYSize();

	float* meanCol = new float[cols0];
	memset(meanCol,0,sizeof(float)*cols0);
	float meanImage = 0;
	int npart=16;
	int cols1=cols0/npart;
	for(int i=0;i<=npart;i++){
		if(i<npart){
			uchar* imgData1 = new uchar[rows0*cols1];
			poBand->RasterIO(GF_Read, i*cols1, 0, cols1, rows0, 
				imgData1, cols1, rows0, Type0, 0, 0 );
			for(int k=0;k<cols1;k++){
				for(int j=0;j<rows0;j++){
					meanCol[i*cols1+k] += float(imgData1[j*cols1+k])/rows0;
				}
				meanImage += meanCol[i*cols1+k]/cols0;
			}
			delete []imgData1;
			imgData1=NULL;
		}
		else if(i==npart && cols1*npart<cols0){
			int temp=cols0-cols1*npart;
			uchar* imgData1 = new uchar[rows0*temp];
			poBand->RasterIO(GF_Read, i*cols1, 0, temp, rows0, 
				imgData1, temp, rows0, Type0, 0, 0 );
			for(int k=0;k<temp;k++){
				for(int j=0;j<rows0;j++){
					meanCol[i*cols1+k] += float(imgData1[j*temp+k])/rows0;
				}
				meanImage += meanCol[i*cols1+k]/cols0;
			}
			delete []imgData1;
			imgData1=NULL;
		}
	}

	//计算列间梯度，及其均值、方差
	/*
	float* Col_gradient = new float[cols0-2];
	memset(Col_gradient,0,sizeof(float)*(cols0-2));
	float mean_gradient = 0;
	for(int i=0;i<cols0-2;i++){
	Col_gradient[i]=(abs(meanCol[i+2]-meanCol[i+1])+abs(meanCol[i+1]-meanCol[i]))/2;
	mean_gradient += Col_gradient[i]/(cols0-2);
	}
	float sigma_gradient = 0;
	for(int i=0;i<cols0-2;i++){
	sigma_gradient += (Col_gradient[i]-mean_gradient)*(Col_gradient[i]-mean_gradient)/(cols0-2);
	}
	sigma_gradient = sqrt(sigma_gradient);
	*/

	//对梯度变化异常的列进行处理
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	GDALDataset* dst = pDriver->Create(cb_path, cols0, rows0, 1, GDT_Byte, ppszOptions);

	for(int i=0;i<=npart;i++){
		if(i<npart){
			uchar* imgData1 = new uchar[rows0*cols1];
			poBand->RasterIO(GF_Read, i*cols1, 0, cols1, rows0, 
				imgData1, cols1, rows0, Type0, 0, 0 );
			for(int k=0;k<cols1;k++){
				for(int j=0;j<rows0;j++){
					imgData1[j*cols1+k] = uchar(float(imgData1[j*cols1+k])*meanImage/meanCol[i*cols1+k]+0.5);
				}
			}
			/*if(abs(mean_gradient-Col_gradient[i-1])>2*sigma_gradient && i>10 && i<cols0-10){
			for(int j=0;j<rows0;j++){
			imgData1[j] = uchar(float(imgData1[j])*(meanCol[i-10])/meanCol[i]+0.5);
			}
			}*/
			dst->GetRasterBand(1)->RasterIO(GF_Write, i*cols1, 0, cols1, rows0, 
				imgData1, cols1, rows0, Type0, 0, 0);
			delete []imgData1;
			imgData1=NULL;
		}
		else if(i==npart && cols1*npart<cols0){
			int temp=cols0-cols1*npart;
			uchar* imgData1 = new uchar[rows0*temp];
			poBand->RasterIO(GF_Read, i*cols1, 0, temp, rows0, 
				imgData1, temp, rows0, Type0, 0, 0 );
			for(int k=0;k<temp;k++){
				for(int j=0;j<rows0;j++){
					imgData1[j*temp+k] = uchar(float(imgData1[j*temp+k])*meanImage/meanCol[i*cols1+k]+0.5);
				}
			}
			dst->GetRasterBand(1)->RasterIO(GF_Write, i*cols1, 0, temp, rows0, 
				imgData1, temp, rows0, Type0, 0, 0);
			delete []imgData1;
			imgData1=NULL;
		}
	}
	delete []meanCol;
	//delete []Col_gradient;
	GDALClose(poBand);
	GDALClose(poDataset);
	GDALClose(dst);
}

//pds格式影像读取及转存
void pds2tif(char* pszFile, char* dstPath,int ImageR,int ImageC){
	//注册文件格式
	GDALAllRegister();

	GDALDataType Type0 = GDT_Byte;

	int CenterR=ImageR/2;int CenterC=ImageC/2;
	uchar *ImageData = new uchar[ImageR*ImageC];
	memset(ImageData,0,sizeof(uchar)*ImageC*ImageR);
	uchar *ImageResult = new uchar[ImageR*ImageC];
	memset(ImageResult,0,sizeof(uchar)*ImageC*ImageR);

	LoadData(pszFile, ImageR, ImageC, 1058, 18, 227757, CenterR, CenterC, ImageR, ImageC, ImageData);
	memcpy(ImageResult,ImageData,sizeof(uchar)*ImageR*ImageC);

	//Col_balance(ImageResult,ImageR,ImageC,ImageData);
	//memcpy(ImageResult,ImageData,sizeof(uchar)*ImageR*ImageC);

	Linear(ImageResult,ImageR,ImageC,ImageData);
	memcpy(ImageResult,ImageData,sizeof(uchar)*ImageR*ImageC);

	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	//char* dstPath = "..\\out\\dst.tif";
	int bufWidth = ImageC;
	int bufHeight = ImageR;
	unsigned char *pBuf1 = new unsigned char[bufWidth*bufHeight];
	int aa=0;
	for(int i=0;i<bufWidth;i++){ 
		for(int j=0;j<bufHeight;j++){
			aa=ImageResult[j*ImageC+i];
			pBuf1[j*bufWidth+i]=ImageResult[j*ImageC+i];
		}
	}
	GDALDataset* dst = pDriver->Create(dstPath, bufWidth, bufHeight, 1, GDT_Byte, ppszOptions);
	dst->GetRasterBand(1)->RasterIO(GF_Write, 
		0, 
		0, 
		bufWidth, 
		bufHeight, 
		pBuf1, 
		bufWidth, 
		bufHeight,
		Type0, 
		1, 
		0); ;
	if (dst == nullptr)
	{
		printf("Can't Write Image!");
		return;
	}

	delete []ImageData;
	delete []ImageResult;
	delete []pBuf1;
	GDALClose((GDALDatasetH)dst); 
	//Linear2(dstPath,"..\\out\\dst1.tif");
}
void pds2tif2(){
	//注册文件格式
	GDALAllRegister();

	GDALDataType Type0 = GDT_Byte;
	char* pszFile = "../data/1650\\PSP_001777_1650_RED0_0.IMG";

	int CenterR=20000;int CenterC=512;
	int ImageR=40000;int ImageC=1024;
	uchar *ImageData = new uchar[ImageR*ImageC];
	memset(ImageData,0,sizeof(uchar)*ImageC*ImageR);
	uchar *ImageResult = new uchar[ImageR*ImageC];
	memset(ImageResult,0,sizeof(uchar)*ImageC*ImageR);

	LoadData(pszFile, 40000, 1024, 1058, 18, 227757, CenterR, CenterC, ImageR, ImageC, ImageData);
	memcpy(ImageResult,ImageData,sizeof(uchar)*ImageR*ImageC);

	Col_balance(ImageResult,ImageR,ImageC,ImageData);
	memcpy(ImageResult,ImageData,sizeof(uchar)*ImageR*ImageC);

	Linear(ImageResult,ImageR,ImageC,ImageData);
	memcpy(ImageResult,ImageData,sizeof(uchar)*ImageR*ImageC);

	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	char* dstPath = "..\\out\\dst.tif";
	int bufWidth = 1024;//ImageC;
	int bufHeight = 1024;//ImageR;
	unsigned char *pBuf1 = new unsigned char[bufWidth*bufHeight];
	int aa=0;
	for(int i=0;i<bufWidth;i++){ 
		for(int j=0;j<bufHeight;j++){
			aa=ImageResult[j*ImageC+i];
			pBuf1[j*bufWidth+i]=ImageResult[j*ImageC+i];
		}
	}
	GDALDataset* dst = pDriver->Create(dstPath, bufWidth, bufHeight, 1, GDT_Byte, ppszOptions);
	dst->GetRasterBand(1)->RasterIO(GF_Write, 
		0, 
		0, 
		bufWidth, 
		bufHeight, 
		pBuf1, 
		bufWidth, 
		bufHeight,
		Type0, 
		1, 
		0); ;
	if (dst == nullptr)
	{
		printf("Can't Write Image!");
		return;
	}
	GDALClose(dst); 

	Linear2(dstPath,"..\\out\\dst1.tif");
}

//同一CCD影像拼接
void mosaic1(char* pszSrcFile1, char* pszSrcFile2, char* pszDstFile,int overlap){
	//注册文件格式
	GDALAllRegister();

	//定义变量
	GDALDataType Type0 = GDT_Byte;
	GDALDataset *poDataset;
	GDALRasterBand *poBand;
	int nXSize1,nYSize1,nXSize2,nYSize2;

	uchar *paf1,*paf2;

	//左影像数据读取
	poDataset = (GDALDataset*) GDALOpen( pszSrcFile1,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",pszSrcFile1);
		return;
	}

	if(poDataset->GetRasterCount()<1){
		return;
	}

	poBand = poDataset->GetRasterBand(1);
	Type0 = poBand->GetRasterDataType();
	nXSize1 = poBand->GetXSize();
	nYSize1 = poBand->GetYSize();

	nXSize1 = nXSize1-overlap;

	paf1 = new unsigned char[nXSize1*nYSize1];
	poBand->RasterIO(GF_Read, 0, 0, nXSize1, nYSize1, 
		paf1, nXSize1, nYSize1, Type0, 1, 0 );

	//右影像数据读取
	poDataset = (GDALDataset*) GDALOpen( pszSrcFile2,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File2: %s不能打开！\n",pszSrcFile2);
		return;
	}

	if(poDataset->GetRasterCount()<1){
		return;
	}

	poBand = poDataset->GetRasterBand(1);
	Type0 = poBand->GetRasterDataType();
	nXSize2 = poBand->GetXSize();
	nYSize2 = poBand->GetYSize();

	paf2 = new unsigned char[nXSize2*nYSize2];
	poBand->RasterIO(GF_Read, 0, 0, nXSize2, nYSize2, 
		paf2, nXSize2, nYSize2, Type0, 1, 0 );

	//关闭文件
	GDALClose((GDALDatasetH)poDataset);

	if(nYSize1==nYSize2){ //nXSize1==nXSize2 && 
		uchar* paf = new unsigned char[(nXSize2+nXSize1)*nYSize1];
		uchar* puf = new unsigned char[(nXSize2+nXSize1)*nYSize1];
		for(int i=0;i<nXSize2+nXSize1;i++){
			for(int j=0;j<nYSize1;j++){
				if(i<nXSize1){
					paf[i+j*(nXSize2+nXSize1)]=paf1[i+j*nXSize1];
				}
				else{
					paf[i+j*(nXSize2+nXSize1)]=paf2[i-nXSize1+j*nXSize2];
				}
			}
		}
		memcpy(puf,paf,sizeof(uchar)*(nXSize2+nXSize1)*nYSize1);
		//Linear(puf,nYSize1,2*nXSize1,paf);
		Col_balance(paf,nYSize1,(nXSize2+nXSize1),puf);
		//memcpy(puf,paf,sizeof(uchar)*2*nXSize1*nYSize1);

		GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
		char** ppszOptions = NULL;
		ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
		GDALDataset* dst = pDriver->Create(pszDstFile, (nXSize2+nXSize1), nYSize1, 1, GDT_Byte, ppszOptions);
		dst->GetRasterBand(1)->RasterIO(GF_Write, 
			0, 
			0, 
			(nXSize2+nXSize1), 
			nYSize1, 
			puf, 
			(nXSize2+nXSize1), 
			nYSize1,
			Type0, 
			1, 
			0); ;
		if (dst == nullptr)
		{
			printf("Can't Write Image!");
			return;
		}
		GDALClose(dst);
		delete [] paf;
		delete [] puf;
	}
	delete [] paf1;
	delete [] paf2;
}

int Down_sample(char* pszSrcFile1, int batchsize, char* pszDstFile){
	//注册文件格式
	GDALAllRegister();

	//定义变量
	GDALDataType Type0 = GDT_Byte;
	GDALDataset *poDataset;
	GDALRasterBand *poBand;
	int nXSize1,nYSize1,nXSize2,nYSize2;

	uchar *paf1,*paf2;

	//左影像数据读取
	poDataset = (GDALDataset*) GDALOpen( pszSrcFile1,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",pszSrcFile1);
		return 0;
	}

	if(poDataset->GetRasterCount()<1){
		return 0;
	}

	poBand = poDataset->GetRasterBand(1);
	Type0 = poBand->GetRasterDataType();
	nXSize1 = poBand->GetXSize();
	nYSize1 = poBand->GetYSize();

	//定义输出图像的大小
	nXSize2=int(nXSize1/batchsize);
	nYSize2=int(nYSize1/batchsize);
	paf2 = new unsigned char[nXSize2*nYSize2];
	memset(paf2,0,sizeof(uchar)*nXSize2*nYSize2);

	int part_size=10;
	int cols1=batchsize*part_size;
	int npart=nXSize1/cols1;
	for(int i=0;i<=npart;i++){
		if(i<npart){
			paf1 = new unsigned char[cols1*nYSize1];
			poBand->RasterIO(GF_Read, i*cols1, 0, cols1, nYSize1, 
				paf1, cols1, nYSize1, Type0, 1, 0 );

			float temp;
			for(int ii=0;ii<cols1/batchsize;ii++){
				for(int j=0;j<nYSize2;j++){
					temp=0;
					for(int k=0;k<batchsize;k++){
						for(int m=0;m<batchsize;m++){
							temp += paf1[(ii*batchsize)+k+(j*batchsize+m)*cols1];
						}
					}
					paf2[part_size*i+ii+j*nXSize2] = uchar(temp/batchsize/batchsize);
				}
			}
		}
		else if(i==npart && cols1*npart<nXSize1){
			int res_col=nXSize1-cols1*npart;
			paf1 = new unsigned char[res_col*nYSize1];
			poBand->RasterIO(GF_Read, i*cols1, 0, res_col, nYSize1, 
				paf1, res_col, nYSize1, Type0, 1, 0 );

			float temp;
			for(int ii=0;ii<res_col/batchsize;ii++){
				for(int j=0;j<nYSize2;j++){
					temp=0;
					for(int k=0;k<batchsize;k++){
						for(int m=0;m<batchsize;m++){
							temp += paf1[(ii*batchsize)+k+(j*batchsize+m)*res_col];
						}
					}
					paf2[part_size*i+ii+j*nXSize2] = uchar(temp/batchsize/batchsize);
				}
			}
		}
	}

	//关闭文件
	GDALClose((GDALDatasetH)poDataset);

	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	GDALDataset* dst = pDriver->Create(pszDstFile, nXSize2, nYSize2, 1, GDT_Byte, ppszOptions);
	dst->GetRasterBand(1)->RasterIO(GF_Write, 
		0, 
		0, 
		nXSize2, 
		nYSize2, 
		paf2, 
		nXSize2, 
		nYSize2,
		Type0, 
		1, 
		0); ;
	if (dst == nullptr)
	{
		printf("Can't Write Image!");
		return 0;
	}
	GDALClose(dst);
	delete [] paf1;
	delete [] paf2;
	return 0;
}

int Hijitreg_gdal(char* imagepathL, char* imagepathR, int OverlapSamples, int* dr, int* dc){
	//注册文件格式
	GDALAllRegister();

	//读入影像
	GDALDataType Type0 = GDT_Byte;
	GDALDataset *poDataset;
	GDALRasterBand *poBand;

	uchar *imgData1,*imgData2;
	int rows1,cols1,rows2,cols2;

	//左影像
	poDataset = (GDALDataset*) GDALOpen( imagepathL,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",imagepathL);
		return 0;
	}

	if(poDataset->GetRasterCount()<1){
		return 0;
	}

	poBand = poDataset->GetRasterBand(1);
	Type0 = poBand->GetRasterDataType();
	cols1 = poBand->GetXSize();
	rows1 = poBand->GetYSize();

	imgData1 = new uchar[rows1*cols1];
	poBand->RasterIO(GF_Read, 0, 0, cols1, rows1, 
		imgData1, cols1, rows1, Type0, 1, 0 );

	//右影像
	poDataset = (GDALDataset*) GDALOpen( imagepathR,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",imagepathR);
		return 0;
	}

	if(poDataset->GetRasterCount()<1){
		return 0;
	}

	poBand = poDataset->GetRasterBand(1);
	Type0 = poBand->GetRasterDataType();
	cols2 = poBand->GetXSize();
	rows2 = poBand->GetYSize();

	imgData2 = new uchar[rows2*cols2];
	poBand->RasterIO(GF_Read, 0, 0, cols2, rows2, 
		imgData2, cols2, rows2, Type0, 1, 0 );

	//关闭文件
	GDALClose((GDALDatasetH)poDataset);


	//生成格网点
	int SampleRateLine = 10;
	int SampleRateSample = 4;

	//格网点灰度匹配
	int dl,ds;
	int w_size=5;
	int N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int dl_max=0;int ds_max=0;
	float maxCC=-1000;
	float CC;
	for(dl=-100;dl<100;dl++){
		for(ds=-5;ds<5;ds++){
			CC=0;
			int count=0;
			for(int i=0+w_size/2;i<rows1-w_size/2;i++){
				for(int j=0+w_size/2;j<OverlapSamples-w_size/2;j++){
					int ii=i-dl;int jj=j-ds;
					if(i%SampleRateLine==0 && j%SampleRateSample==0 && ii>=w_size/2 && ii<=rows2-w_size/2 && jj>=w_size/2 && jj<=cols2-w_size/2){
						s12=0;s11=0;s22=0;s1=0;s2=0;
						int sum1=0;int sum2=0;
						int grey1=imgData1[i*cols1+cols1-OverlapSamples+j];
						int grey2=imgData2[(ii)*cols2+jj];
						for(int k=-w_size/2;k<w_size/2;k++){
							for(int m=-w_size/2;m<w_size/2;m++){
								r1=i+k;r2=ii+k;
								c1=cols1-OverlapSamples+j+m;c2=jj+m;
								sum1 += imgData1[r1*cols1+c1]-grey1;
								sum2 += imgData2[r2*cols2+c2]-grey2;
								s12 += (imgData1[r1*cols1+c1]-grey1)*(imgData2[r2*cols2+c2]-grey2);
								s22 += (imgData2[r2*cols2+c2]-grey2)*(imgData2[r2*cols2+c2]-grey2);
								s11 += (imgData1[r1*cols1+c1]-grey1)*(imgData1[r1*cols1+c1]-grey1);
								s1  += (imgData1[r1*cols1+c1]-grey1);
								s2  += (imgData2[r2*cols2+c2]-grey2);
							}
						}
						if(sum1==0) sum1=1;
						if(sum2==0) sum2=1;
						s12 = s12/sum1/sum2;
						s22 = s22/sum2/sum2;
						s11 = s11/sum1/sum1;
						s1 = s1/sum1;
						s2 = s2/sum2;
						float temp=(s11-s1*s1/N)*(s22-s2*s2/N);
						if(temp<=0) temp=0.0001;
						CC += (s12-s1*s2/N)/sqrt(temp);
						count++;
					}
				}
			}
			CC=CC/count;
			if(CC>maxCC){
				maxCC=CC;
				dl_max=dl;
				ds_max=ds;
			}
		}
	}
	//最小二乘思想（dx，dy）精确对齐

	*dr=dl_max;
	*dc=ds_max;
	delete []imgData1;
	delete []imgData2;
	return 0;
}

int CCDmosaic_gdal(char* imagepathL, char* imagepathR, int OverlapSamples, char* outpath){
	//注册文件格式
	GDALAllRegister();

	//读入影像
	GDALDataType Type0 = GDT_Byte;
	GDALDataset *poDataset;
	GDALRasterBand *poBand;

	uchar *imgData1,*imgData2;
	int rows1,cols1,rows2,cols2;

	//左影像
	poDataset = (GDALDataset*) GDALOpen( imagepathL,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",imagepathL);
		return 0;
	}

	if(poDataset->GetRasterCount()<1){
		return 0;
	}

	poBand = poDataset->GetRasterBand(1);
	Type0 = poBand->GetRasterDataType();
	cols1 = poBand->GetXSize();
	rows1 = poBand->GetYSize();

	imgData1 = new uchar[rows1*cols1];
	poBand->RasterIO(GF_Read, 0, 0, cols1, rows1, 
		imgData1, cols1, rows1, Type0, 1, 0 );

	//右影像
	poDataset = (GDALDataset*) GDALOpen( imagepathR,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",imagepathR);
		return 0;
	}

	if(poDataset->GetRasterCount()<1){
		return 0;
	}

	poBand = poDataset->GetRasterBand(1);
	Type0 = poBand->GetRasterDataType();
	cols2 = poBand->GetXSize();
	rows2 = poBand->GetYSize();

	imgData2 = new uchar[rows2*cols2];
	poBand->RasterIO(GF_Read, 0, 0, cols2, rows2, 
		imgData2, cols2, rows2, Type0, 1, 0 );

	//关闭文件
	GDALClose((GDALDatasetH)poDataset);


	//生成格网点
	int SampleRateLine = 10;
	int SampleRateSample = 4;

	//格网点灰度匹配
	int dl,ds;
	int w_size=5;
	int N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int dl_max=0;int ds_max=0;
	float maxCC=-1000;
	float CC;
	for(dl=-100;dl<100;dl++){
		for(ds=-5;ds<5;ds++){
			CC=0;
			int count=0;
			for(int i=0+w_size/2;i<rows1-w_size/2;i++){
				for(int j=0+w_size/2;j<OverlapSamples-w_size/2;j++){
					int ii=i-dl;int jj=j-ds;
					if(i%SampleRateLine==0 && j%SampleRateSample==0 && ii>=w_size/2 && ii<=rows2-w_size/2 && jj>=w_size/2 && jj<=cols2-w_size/2){
						s12=0;s11=0;s22=0;s1=0;s2=0;
						int sum1=0;int sum2=0;
						int grey1=imgData1[i*cols1+cols1-OverlapSamples+j];
						int grey2=imgData2[(ii)*cols2+jj];
						for(int k=-w_size/2;k<w_size/2;k++){
							for(int m=-w_size/2;m<w_size/2;m++){
								r1=i+k;r2=ii+k;
								c1=cols1-OverlapSamples+j+m;c2=jj+m;
								sum1 += imgData1[r1*cols1+c1]-grey1;
								sum2 += imgData2[r2*cols2+c2]-grey2;
								s12 += (imgData1[r1*cols1+c1]-grey1)*(imgData2[r2*cols2+c2]-grey2);
								s22 += (imgData2[r2*cols2+c2]-grey2)*(imgData2[r2*cols2+c2]-grey2);
								s11 += (imgData1[r1*cols1+c1]-grey1)*(imgData1[r1*cols1+c1]-grey1);
								s1  += (imgData1[r1*cols1+c1]-grey1);
								s2  += (imgData2[r2*cols2+c2]-grey2);
							}
						}
						if(sum1==0) sum1=1;
						if(sum2==0) sum2=1;
						s12 = s12/sum1/sum2;
						s22 = s22/sum2/sum2;
						s11 = s11/sum1/sum1;
						s1 = s1/sum1;
						s2 = s2/sum2;
						float temp=(s11-s1*s1/N)*(s22-s2*s2/N);
						if(temp<=0) temp=0.0001;
						CC += (s12-s1*s2/N)/sqrt(temp);
						count++;
					}
				}
			}
			CC=CC/count;
			if(CC>maxCC){
				maxCC=CC;
				dl_max=dl;
				ds_max=ds;
			}
		}
	}
	//最小二乘思想（dx，dy）精确对齐

	//拼接
	int beginR = 0<=dl_max ? 0:dl_max;
	int endR = rows1>rows2+dl_max ? rows1:rows2+dl_max;

	int rows=endR-beginR;
	int cols=cols1+cols2-OverlapSamples+ds_max;

	uchar* paf = new unsigned char[rows*cols];

	if(beginR==0){
		for(int i=0;i<rows1;i++){
			for(int j=0;j<cols1;j++){
				paf[i*cols+j]=imgData1[i*cols1+j];
			}
		}
		for(int i=0;i<rows2;i++){
			for(int j=0;j<cols2;j++){
				paf[(i+dl_max)*cols+j+cols1-OverlapSamples+ds_max]=imgData2[i*cols2+j];
			}
		}
	}
	else{
		for(int i=0;i<rows1;i++){
			for(int j=0;j<cols1;j++){
				paf[(i-dl_max)*cols+j]=imgData1[i*cols1+j];
			}
		}
		for(int i=0;i<rows2;i++){
			for(int j=0;j<cols2;j++){
				paf[i*cols+j+cols1-OverlapSamples+ds_max]=imgData2[i*cols2+j];
			}
		}
	}
	delete[] imgData1;
	delete[] imgData2;

	//uchar* puf = new unsigned char[rows*cols];
	//memcpy(puf,paf,sizeof(uchar)*rows*cols);
	//Col_balance(paf,rows,cols,puf);
	//delete[] paf;

	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	GDALDataset* dst = pDriver->Create(outpath, cols, rows, 1, GDT_Byte, ppszOptions);
	dst->GetRasterBand(1)->RasterIO(GF_Write, 
		0, 
		0, 
		cols, 
		rows, 
		paf, 
		cols, 
		rows,
		Type0, 
		1, 
		0); ;
	if (dst == nullptr)
	{
		printf("Can't Write Image!");
		return 0;
	}
	GDALClose(dst);
	delete[] paf;
}

void My_rec2NEH(const char *projRef,double X,double Y,double Z,double* N,double* E,double* H){
	//recgeo
	ConstSpiceDouble rectan[3]={X,Y,Z};
	//ConstSpiceDouble rectan[3]={-3396190,0,0};
	//recgeo_c(rectan,3396190,0.00736,E,N,H); 
	recgeo_c(rectan,3396190,0.00736,E,N,H); 

	//gdal:geo2neh
	OGRSpatialReference fRef;
	char *tmp = NULL;
	//const char *projRef =poDataset->GetProjectionRef();
	/** 获得projRef的一份拷贝 **/
	/** 由于projRef是const char*,下面的一个函数不接受，所以需要转换成非const **/
	tmp = (char *)malloc(strlen(projRef) + 1);
	strcpy(tmp, projRef);


	/** 设置原始的坐标参数，和test.tif一致 **/
	fRef.importFromWkt(&tmp);           //NEH
	/** 设置转换后的坐标 **/
	OGRSpatialReference *tRef;
	tRef=fRef.CloneGeogCS();  //只复制出地理坐标那部分(lat,lon,H)

	/** 下面进行坐标转换，到此为止都不需要proj，但是下面的内容如果不安装proj将会无法编译 **/
	OGRCoordinateTransformation *coordTrans;
	coordTrans = OGRCreateCoordinateTransformation(tRef, &fRef);
	
	*N=(*N)*180/3.1425926;
	*E=(*E)*180/3.1425926;
	//std::cout<<*N<<" "<<*E<<std::endl;
	//coordTrans->Transform(1, E, N, H);
	//std::cout<<*N<<" "<<*E<<std::endl;
	//printf("lat:%lf lon:%lf H:%lf\n",*N,*E,*H);
}

void My_NEH2rec(const char *projRef,double N,double E,double H,double* X,double* Y,double* Z){
	//gdal:NEH2geo
	OGRSpatialReference fRef;
	char *tmp = NULL;
	//const char *projRef =poDataset->GetProjectionRef();
	/** 获得projRef的一份拷贝 **/
	/** 由于projRef是const char*,下面的一个函数不接受，所以需要转换成非const **/
	tmp = (char *)malloc(strlen(projRef) + 1);
	strcpy(tmp, projRef);


	/** 设置原始的坐标参数，和test.tif一致 **/
	fRef.importFromWkt(&tmp);           //NEH
	/** 设置转换后的坐标 **/
	OGRSpatialReference *tRef;
	tRef=fRef.CloneGeogCS();  //只复制出地理坐标那部分(lat,lon,H)

	/** 下面进行坐标转换，到此为止都不需要proj，但是下面的内容如果不安装proj将会无法编译 **/
	OGRCoordinateTransformation *coordTrans;
	coordTrans = OGRCreateCoordinateTransformation(&fRef, tRef);
	coordTrans->Transform(1, &N, &E, &H);  //角度输出
	N=N*3.1415926/180;
	E=E*3.1415926/180;
	//printf("lat:%lf lon:%lf H:%lf\n",N,E,H);

	//georec
	SpiceDouble* rectan = new double[3];
	//georec_c(E,N,H,3396190,0.00736,rectan);
	georec_c(E,N,H,3396190,0,rectan);  //弧度输入
	*X=rectan[0];
	*Y=rectan[1];
	*Z=rectan[2];
}

//读取MOLA DEM数据
void LoadDEM(char* imagepath, float X, float Y, float* Z){
	//注册文件格式
	GDALAllRegister();

	GDALDataset *poDataset;
	//使用只读方式打开图像
	poDataset = (GDALDataset*) GDALOpen( imagepath,GA_ReadOnly );
	if( poDataset == NULL )
	{
		printf( "File: %s不能打开！\n",imagepath);
		return;
	}

	//输出图像的左上角坐标和分辨率信息
	double adfGeoTransform[6];
	if( poDataset->GetGeoTransform( adfGeoTransform) == CE_None )
	{
		//printf( "Origin =(%.6f,%.6f)\n",adfGeoTransform[0], adfGeoTransform[3]);
		//printf( "PixelSize = (%.6f,%.6f)\n",adfGeoTransform[1], adfGeoTransform[5]);
	}

	GDALRasterBand *poBand = poDataset->GetRasterBand(1);
	GDALDataType Type0 = poBand->GetRasterDataType();
	int nXSize = poBand->GetXSize();
	int nYSize = poBand->GetYSize();


	/*
	switch (Type0)
	{
	case GDT_Byte:
		{
			unsigned char *pafScanblock1;
			pafScanblock1 = (unsigned char *)CPLMalloc(sizeof(unsigned char)*(1)*(1));//建议能小则小否则会造成内存不足的情况
			poBand1->RasterIO(GF_Read, dx, dy, 1, 1, pafScanblock1, 1, 1, GDALDataType(poBand1->GetRasterDataType()), 0, 0);
			cout << Xgeo << "  " << Ygeo << "  " << *pafScanblock1 << endl;
			break;
		}
	case GDT_UInt16:
		{
			unsigned short *pafScanblock1;
			pafScanblock1 = (unsigned short *)CPLMalloc(sizeof(unsigned short)*(1)*(1));
			poBand1->RasterIO(GF_Read, dx, dy, 1, 1, pafScanblock1, 1, 1, GDALDataType(poBand1->GetRasterDataType()), 0, 0);
			cout << Xgeo << "  " << Ygeo << "  " << *pafScanblock1 << endl;
			break;
		}
	case GDT_Int16:
		{
			short int *pafScanblock1;
			pafScanblock1 = (short int *)CPLMalloc(sizeof(short int)*(1)*(1));
			poBand1->RasterIO(GF_Read, dx, dy, 1, 1, pafScanblock1, 1, 1, GDALDataType(poBand1->GetRasterDataType()), 0, 0);
			cout << Xgeo << "  " << Ygeo << "  " << *pafScanblock1 << endl;
			break;
		}
	case GDT_UInt32:
		{
			unsigned long *pafScanblock1;
			pafScanblock1 = (unsigned long *)CPLMalloc(sizeof(unsigned long)*(1)*(1));
			poBand1->RasterIO(GF_Read, dx, dy, 1, 1, pafScanblock1, 1, 1, GDALDataType(poBand1->GetRasterDataType()), 0, 0);
			cout << Xgeo << "  " << Ygeo << "  " << *pafScanblock1 << endl;
			break;
		}
	case GDT_Int32:
		{
			long *pafScanblock1;
			pafScanblock1 = (long *)CPLMalloc(sizeof(long)*(1)*(1));
			poBand1->RasterIO(GF_Read, dx, dy, 1, 1, pafScanblock1, 1, 1, GDALDataType(poBand1->GetRasterDataType()), 0, 0);
			cout << Xgeo << "  " << Ygeo << "  " << *pafScanblock1 << endl;
			break;
		}
	case GDT_Float32:
		{
			float *pafScanblock1;
			pafScanblock1 = (float *)CPLMalloc(sizeof(float)*(1)*(1));
			poBand1->RasterIO(GF_Read, dx, dy, 1, 1, pafScanblock1, 1, 1, GDALDataType(poBand1->GetRasterDataType()), 0, 0);
			cout << Xgeo << "  " << Ygeo << "  " << *pafScanblock1 << endl;
			break;
		}
	case GDT_Float64:
		{
			double *pafScanblock1;
			pafScanblock1 = (double *)CPLMalloc(sizeof(double)*(1)*(1));
			poBand1->RasterIO(GF_Read, dx, dy, 1, 1, pafScanblock1, 1, 1, GDALDataType(poBand1->GetRasterDataType()), 0, 0);
			cout << Xgeo << "  " << Ygeo << "  " << *pafScanblock1 << endl;
			break;
		}
	default: break;
	}
	*/

	short H;
	int row = int((Y - adfGeoTransform[3])/adfGeoTransform[5]+0.5);
	int col = int((X - adfGeoTransform[0])/adfGeoTransform[1]+0.5);
	poBand->RasterIO(GF_Read, col, row, 1, 1, 
				&H, 1, 1, Type0, 1, 0 );

	GDALClose(poBand);
	GDALClose(poDataset);

	*Z = float(H);
}

void xulie_process(){
	char* imagepath1="../data/1650\\PSP_001777_1650_RED";
	char* imagepath2="../data/1655\\PSP_001513_1655_RED";
	char* outpath1="G:\\Mars../data/1650\\src\\PSP_001777_1650_RED";
	char* outpath11="G:\\Mars../data/1650\\mosaic\\PSP_001777_1650_RED";
	char* outpath2="G:\\Mars../data/1655\\src\\PSP_001513_1655_RED";
	char* outpath22="G:\\Mars../data/1655\\mosaic\\PSP_001513_1655_RED";
	char* outpath="G:\\Mars../data/";

	for(int i=0;i<10;i++){
		char* out1 = new char[80];
		char* out2 = new char[80];
		char* dstpath1 = new char[80];
		char* dstpath2 = new char[80];
		char* dstpath = new char[80];
		sprintf( out1, "%s%d%s%d%s", imagepath1, i,"_", 0, ".img" );
		sprintf( out2, "%s%d%s%d%s", imagepath1, i,"_", 1, ".img" );
		sprintf( dstpath1, "%s%d%s%d%s", outpath1, i,"_", 0, ".tif" );
		sprintf( dstpath2, "%s%d%s%d%s", outpath1, i,"_", 1, ".tif" );
		sprintf( dstpath, "%s%d%s", outpath11, i, ".tif" );


		//拼接
		pds2tif(out1, dstpath1,40000,1024);
		pds2tif(out2, dstpath2,80000,1024);
		mosaic1(dstpath2, dstpath1, dstpath,0);/**/

		//降采样
		char* dstpathds = new char[80];
		sprintf( dstpathds, "%s%s%d%s", outpath,"/1650\\downsample/PSP_001777_1650_RED", i, "_ds.tif" );
		Down_sample(dstpath,8,dstpathds);

		delete [] out1,out2,dstpath1,dstpath2,dstpath,dstpathds;
	}

	for(int i=0;i<10;i++){
		char* out1 = new char[80];
		char* out2 = new char[80];
		char* dstpath1 = new char[80];
		char* dstpath2 = new char[80];
		char* dstpath = new char[80];
		sprintf( out1, "%s%d%s%d%s", imagepath2, i,"_", 0, ".img" );
		sprintf( out2, "%s%d%s%d%s", imagepath2, i,"_", 1, ".img" );
		sprintf( dstpath1, "%s%d%s%d%s", outpath2, i,"_", 0, ".tif" );
		sprintf( dstpath2, "%s%d%s%d%s", outpath2, i,"_", 1, ".tif" );
		sprintf( dstpath, "%s%d%s", outpath22, i, ".tif" );


		//拼接

		pds2tif(out1, dstpath1,40000,1024);
		pds2tif(out2, dstpath2,80000,1024);
		mosaic1(dstpath2, dstpath1, dstpath,0);/**/

		//降采样
		char* dstpathds = new char[80];
		sprintf( dstpathds, "%s%s%d%s", outpath,"/1655\\downsample/PSP_001513_1655_RED", i, "_ds.tif" );
		Down_sample(dstpath,8,dstpathds);

		delete [] out1,out2,dstpath1,dstpath2,dstpath,dstpathds;
	}

}

void xulie_process1(char* filepath,char* outfilepath,char* xulie_ID1,char* xulie_ID2,int rows1,int rows2){

	char* imagepath1 = new char[80];
	char* imagepath2 = new char[80];
	sprintf( imagepath1, "%s%s%s%s%s%s", filepath, "/", xulie_ID1, "/",xulie_ID1,"_RED");
	sprintf( imagepath2, "%s%s%s%s%s%s", filepath, "/", xulie_ID2, "/",xulie_ID2,"_RED");

	char* outpath1 = new char[80];
	char* outpath2 = new char[80];
	sprintf( outpath1, "%s%s%s%s%s%s", outfilepath, "/", xulie_ID1, "/downsample/0\\",xulie_ID1,"_RED");
	sprintf( outpath2, "%s%s%s%s%s%s", outfilepath, "/", xulie_ID2, "/downsample/0\\",xulie_ID2,"_RED");


	for(int i=0;i<10;i++){
		char* out1 = new char[80];
		char* out2 = new char[80];
		char* dstpath1 = new char[80];
		char* dstpath2 = new char[80];
		char* dstpath = new char[80];
		sprintf( out1, "%s%d%s%d%s", imagepath1, i,"_", 0, ".img" );
		sprintf( out2, "%s%d%s%d%s", imagepath1, i,"_", 1, ".img" );
		sprintf( dstpath1, "%s%d%s%d%s", outpath1, i,"_", 0, ".tif" );
		sprintf( dstpath2, "%s%d%s%d%s", outpath1, i,"_", 1, ".tif" );
		sprintf( dstpath, "%s%d%s", outpath1, i, ".tif" );


		//拼接
		pds2tif(out1, dstpath1,rows1,1024);
		pds2tif(out2, dstpath2,rows1,1024);
		mosaic1(dstpath2, dstpath1, dstpath,0);/**/

		delete [] out1,out2,dstpath1,dstpath2,dstpath;
	}

	for(int i=0;i<10;i++){
		char* out1 = new char[80];
		char* out2 = new char[80];
		char* dstpath1 = new char[80];
		char* dstpath2 = new char[80];
		char* dstpath = new char[80];
		sprintf( out1, "%s%d%s%d%s", imagepath2, i,"_", 0, ".img" );
		sprintf( out2, "%s%d%s%d%s", imagepath2, i,"_", 1, ".img" );
		sprintf( dstpath1, "%s%d%s%d%s", outpath2, i,"_", 0, ".tif" );
		sprintf( dstpath2, "%s%d%s%d%s", outpath2, i,"_", 1, ".tif" );
		sprintf( dstpath, "%s%d%s", outpath2, i, ".tif" );


		//拼接
		pds2tif(out1, dstpath1,rows2,1024);
		pds2tif(out2, dstpath2,rows2,1024);
		mosaic1(dstpath2, dstpath1, dstpath,0);/**/

		delete [] out1,out2,dstpath1,dstpath2,dstpath;
	}
}

void tif_load(char* imagepath1,uchar* data1){
	GDALAllRegister();
	//定义变量
	GDALDataType Type0 = GDT_Byte;
	GDALDataset *poDataset;
	int nXSize1,nYSize1,nBands;
	//uchar* data1;

	//左影像读取
	
	poDataset = (GDALDataset*) GDALOpen( imagepath1,GA_ReadOnly);
	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",imagepath1);
		return;
	}
	if(poDataset->GetRasterCount()<1){
		printf("视差图波段小于1！\n");
		return;
	}
	nXSize1 = poDataset->GetRasterBand(1)->GetXSize();
	nYSize1 = poDataset->GetRasterBand(1)->GetYSize();
	nBands = poDataset->GetRasterCount();
	int rows1=nYSize1;
	int cols1=nXSize1;
	Type0 = poDataset->GetRasterBand(1)->GetRasterDataType();

	//data1 = new uchar[nXSize1*nYSize1];
	poDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, nXSize1, nYSize1, 
		data1, nXSize1, nYSize1, Type0, 0, 0 );

	GDALClose(poDataset);
}

void ds_mosaic(){
	char* outpath="G:\\Mars\\data";

	char* dstpathds0 = new char[80];
	sprintf( dstpathds0, "%s%s%d%s", outpath,"/1650\\downsample/PSP_001777_1650_RED", 0, "_ds.tif" );
	char* mosaicpath = new char[80];
	sprintf( mosaicpath, "%s%s", outpath,"/1650\\downsample/PSP_001777_1650_RED.tif" );
	for(int i=1;i<10;i++){
		//降采样
		char* dstpathds = new char[80];
		sprintf( dstpathds, "%s%s%d%s", outpath,"/1650\\downsample/PSP_001777_1650_RED", i, "_ds.tif" );
		if(i==1){
			mosaic1(dstpathds0,dstpathds,mosaicpath,6);
		}
		else{
			mosaic1(mosaicpath,dstpathds,mosaicpath,6);
		}
	}


	sprintf( dstpathds0, "%s%s%d%s", outpath,"/1655\\downsample/PSP_001513_1655_RED", 0, "_ds.tif" );
	sprintf( mosaicpath, "%s%s", outpath,"/1655\\downsample/PSP_001513_1655_RED.tif" );
	for(int i=1;i<10;i++){
		//降采样
		char* dstpathds = new char[80];
		sprintf( dstpathds, "%s%s%d%s", outpath,"/1655\\downsample/PSP_001513_1655_RED", i, "_ds.tif" );
		if(i==1){
			mosaic1(dstpathds0,dstpathds,mosaicpath,6);
		}
		else{
			mosaic1(mosaicpath,dstpathds,mosaicpath,6);
		}
	}
}

void gdal_test(){

	//转存和拼接
	char* imagepath1="../data/1655\\PSP_001513_1655_RED0_0.IMG";
	char* imagepath2="../data/1655\\PSP_001513_1655_RED0_1.IMG";
	char* dstpath1="../data/1655tif\\PSP_001513_1655_RED0_0.tif";
	char* dstpath2="../data/1655tif\\PSP_001513_1655_RED0_1.tif";

	char* dstpath="../data/1655tif\\PSP_001513_1655_RED0.tif";
	char* dstpath_linear="../data/1655tif\\PSP_001513_1655_RED0_Linear.tif";

	pds2tif(imagepath1, dstpath1,40000,1024);
	pds2tif(imagepath2, dstpath2,80000,1024);
	mosaic1(dstpath2, dstpath1, dstpath,0);
	Linear2(dstpath, dstpath_linear);

	//降采样
	//Down_sample("../data/1655tif\\PSP_001513_1655_RED0.tif",8,"../data/1655tif\\PSP_001513_1655_RED0_ds8.tif");
}