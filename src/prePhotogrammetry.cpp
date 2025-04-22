#include "AeTri.h"

int cvtest(){
	Mat Image1 = imread("..\\data\\image\\1.jpg",0);
	Image1.setTo(255);
	imshow("cvtest" ,Image1);
	imwrite("../out/1.tif",Image1);
	waitKey(0);

	return 0;
}

int surf_feature(){
	Mat img_1 = imread("..\\data\\1650tif\\PSP_001777_1650_RED2.tif", 1);
	Mat img_2 = imread("..\\data\\1650tif\\PSP_001777_1650_RED3.tif", 1);

	if( !img_1.data || !img_2.data )
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 100;

	SurfFeatureDetector Detector(minHessian);
	//SiftFeatureDetector Detector;

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	Detector.detect( img_1, keypoints_1 );
	Detector.detect( img_2, keypoints_2 );

	//-- Step 2: Calculate descriptors (feature vectors)
	//SurfDescriptorExtractor extractor;
	SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute( img_1, keypoints_1, descriptors_1 );
	extractor.compute( img_2, keypoints_2, descriptors_2 );

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_1.rows; i++ )
	{ double dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;

	for( int i = 0; i < descriptors_1.rows; i++ )
	{ if( matches[i].distance < 2*min_dist )
	{ good_matches.push_back( matches[i]); }
	}

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//-- Show detected matches
	imshow( "Good Matches", img_matches );

	for( int i = 0; i < good_matches.size(); i++ )
	{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

	waitKey(0);

	return 0;
}

int HarrisDet(){
	//以灰度模式载入图像并显示
	Mat srcImage = imread("..\\data\\1650tif\\PSP_001777_1650_RED2.tif", 0);
	imshow("原始图", srcImage);

	//进行Harris角点检测找出角点
	Mat cornerStrength;
	cornerHarris(srcImage, cornerStrength, 2, 3, 0.01);

	//对灰度图进行阈值操作，得到二值图并显示
	Mat harrisCorner;
	threshold(cornerStrength, harrisCorner, 0.00001, 255, THRESH_BINARY);
	imshow("角点检测后的二值效果图", harrisCorner);
	waitKey(0);

	return 0;
}

int guu(uchar* pImg,int ii, int jj,int cols){
	return pImg[(ii+1)*cols+(jj+1)]-pImg[ii*cols+jj];
}
int gvv(uchar* pImg,int ii, int jj,int cols){
	return pImg[ii*cols+(jj+1)]-pImg[(ii+1)*cols+jj];
}

int Forstner(uchar* pImg,int rows,int cols,int base, std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y)
{
	//分配内存存储每点的两个方向梯度（Robert梯度）
	float *gu = new float[rows*cols];
	float *gv = new float[rows*cols];
	memset(gu, 0, sizeof(float)*rows*cols);
	memset(gv, 0, sizeof(float)*rows*cols);

	//逐点计算梯度
	for(int ii=1;ii<rows-1;ii++){
		for(int jj=1;jj<cols-1;jj++){
			int m = ii*cols+jj;
			gu[ii*cols+jj]=pImg[(ii+1)*cols+(jj+1)]-pImg[ii*cols+jj];
			gv[ii*cols+jj]=pImg[ii*cols+(jj+1)]-pImg[(ii+1)*cols+jj];
		} 
	}

	//分配内存并计算存储各点灰度协方差矩阵参数（以5*5窗口为例）
	float *Sguu = new float[rows*cols];
	float *Sgvv = new float[rows*cols];
	float *Sguv = new float[rows*cols];
	memset(Sguu, 0, sizeof(float)*rows*cols);
	memset(Sgvv, 0, sizeof(float)*rows*cols);
	memset(Sguv, 0, sizeof(float)*rows*cols);


	int dim=7;

	//最后一行列无梯度，所以-dim/2-1。
	for(int ii=dim/2;ii<rows-dim/2-1;ii++){
		for(int jj=dim/2;jj<cols-dim/2-1;jj++){
			int m = ii*cols+jj;

			//5*5窗口内计算（ii，jj）点灰度协方差参数
			Sguu[m]=Sgvv[m]=Sguv[m]=0;
			for(int kk=-dim/2;kk<=dim/2;kk++){
				for(int ll=-dim/2;ll<=dim/2;ll++){
					Sguu[m] += (gu[(ii+kk)*cols+(jj+ll)])*(gu[(ii+kk)*cols+(jj+ll)]);
					Sgvv[m] += (gv[(ii+kk)*cols+(jj+ll)])*(gv[(ii+kk)*cols+(jj+ll)]);
					Sguv[m] += gu[(ii+kk)*cols+(jj+ll)]*gv[(ii+kk)*cols+(jj+ll)];
					//Sguu[m] += guu(pImg,ii+kk,jj+ll,cols)*guu(pImg,ii+kk,jj+ll,cols);
					//Sgvv[m] += gvv(pImg,ii+kk,jj+ll,cols)*gvv(pImg,ii+kk,jj+ll,cols);
					//Sguv[m] += guu(pImg,ii+kk,jj+ll,cols)*gvv(pImg,ii+kk,jj+ll,cols);;
				}
			}
		} 
	}

	//分配存储空间并计算兴趣值q和w
	float *q = new float[rows*cols];
	float *w = new float[rows*cols];
	memset(q, 0, rows*cols);
	memset(w, 0, rows*cols);

	int count=0;                         //用于计数
	float weva=0;                       //平均权值
	for(int ii=2;ii<rows-3;ii++){
		for(int jj=2;jj<cols-3;jj++){
			int m = ii*cols+jj;
			if(Sguu[m]+Sgvv[m]!=0)
			{
				//Sguu[m]=0.0000001;
				//规避分母为0
				w[m] = (Sguu[m]*Sgvv[m]-Sguv[m]*Sguv[m])/(Sguu[m]+Sgvv[m]);
				q[m] = 4*(Sguu[m]*Sgvv[m]-Sguv[m]*Sguv[m])/((Sguu[m]+Sgvv[m])*(Sguu[m]+Sgvv[m]));
				weva += w[m];
				count=count+1;
			} 
		} 
	}

	weva=weva/count;
	//printf("平均权值：%lf 待选点总数：%d\n",weva,count);

	//给定阈值Tq,Tw选取待选点
	//一般Tq=0.5~0.75
	//Tw=fw(w为权平均值,f=0.5~1.5)或Tw=5*wc(wc为权中值）
	float Tq=0.93;
	float Tw=2*weva;

	Mat image1;
	image1.create(rows, cols, CV_8UC1);
	unsigned char *pImg1 = image1.data;
	memset(pImg1, 255, rows*cols);

	for(int ii=2;ii<rows-3;ii++){
		for(int jj=2;jj<cols-3;jj++){
			int m = ii*cols+jj;
			if(w[m]>Tw && q[m]>Tq){
				int n=w[m];

				//9*9窗口内搜索区域最大值(仅以w为参考)
				int w_s = 7;
				if(ii>=w_s && ii<rows-w_s && jj>=w_s && jj<cols-w_s){
					for(int a=-w_s;a<=w_s;a++){
						for(int b=-w_s;b<=w_s;b++){
							if(n<w[(ii+a)*cols+jj+b]){
								n=w[(ii+a)*cols+jj+b];
							}
						}
					}
					//判断是否区域极值
					if(w[ii*cols+jj]>=n){
						KeyPoint_x.push_back(base+ii);
						KeyPoint_y.push_back(jj);
						//pImg[ii*cols+jj]=0;
						//circle(image2,cvPoint(jj,ii),3,CV_RGB(255,0,0),0.5,8,0); 
					}
				}
			} 
		}	
	}
	//delete []gu;
	//delete []gv;
	delete []pImg;
	delete []Sguu;
	delete []Sgvv;
	delete []Sguv;
	delete []q;
	delete []w;
	//imshow("fig2",image2);
	//waitKey(0);
	return 0;
}

//2d高斯卷积核生成
void GaussianKernel(float sigma,int dim, float* kernel){
	float m= 1.0/(2.0 * CV_PI * sigma * sigma);  

	int c=dim/2;
	float count=0;
	for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){
			float p=m*exp(-((i-c)*(i-c)+(j-c)*(j-c))/(2.0 * sigma * sigma)); 
			kernel[i*dim+j] = p;
			count += p;
		}
	}
	for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){
			kernel[i*dim+j] = kernel[i*dim+j]/count;
		}
	}
}

//n维高斯滤波
void Gaussian_Juanji(float* InArray, float* OutArray, int rows, int cols, int dim, float sigma){
	float* kernel = new float[dim*dim];
	GaussianKernel(sigma,dim,kernel);
	int c=dim/2;
	for(int i=c;i<rows-c;i++){
		for(int j=c;j<cols-c;j++){
			float m=0;
			for(int k=0;k<dim;k++){
				for(int l=0;l<dim;l++){
					m += kernel[k*dim+l]*float(InArray[(i-c+k)*cols+j-c+l]);
				}
			}
			OutArray[i*cols+j]=m;
		}
	}
}

//快速排序
void fast_sort(int *a, int left, int right)
{
    if(left >= right)/*如果左边索引大于或者等于右边的索引就代表已经整理完成一个组了*/
    {
        return ;
    }
    int i = left;
    int j = right;
    int key = a[left];
     
    while(i < j)                               /*控制在当组内寻找一遍*/
    {
        while(i < j && key <= a[j])
        /*而寻找结束的条件就是，1，找到一个小于或者大于key的数（大于或小于取决于你想升
        序还是降序）2，没有符合条件1的，并且i与j的大小没有反转*/ 
        {
            j--;/*向前寻找*/
        }
         
        a[i] = a[j];
        /*找到一个这样的数后就把它赋给前面的被拿走的i的值（如果第一次循环且key是
        a[left]，那么就是给key）*/
         
        while(i < j && key >= a[i])
        /*这是i在当组内向前寻找，同上，不过注意与key的大小关系停止循环和上面相反，
        因为排序思想是把数往两边扔，所以左右两边的数大小与key的关系相反*/
        {
            i++;
        }
         
        a[j] = a[i];
    }
     
    a[i] = key;/*当在当组内找完一遍以后就把中间数key回归*/
    fast_sort(a, left, i - 1);/*最后用同样的方式对分出来的左边的小组进行同上的做法*/
    fast_sort(a, i + 1, right);/*用同样的方式对分出来的右边的小组进行同上的做法*/
                       /*当然最后可能会出现很多分左右，直到每一组的i = j 为止*/
}

int HarrisCorner(uchar* pImg,int rows,int cols,int Juanji_dim,float Juanji_sigma,int Localmax_dim,float threshold,std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y){
	int channels = 1;

	//定义数组存储梯度
	float *Grx  = new float[rows*cols];
	float *Gry  = new float[rows*cols];
	float *Grxx = new float[rows*cols];
	float *Gryy = new float[rows*cols];
	float *Grxy = new float[rows*cols];
	memset(Grx,  0, rows*cols*sizeof(float));
	memset(Gry,  0, rows*cols*sizeof(float));
	memset(Grxx, 0, rows*cols*sizeof(float));
	memset(Gryy, 0, rows*cols*sizeof(float));
	memset(Grxy, 0, rows*cols*sizeof(float));

	//遍历影像计算各点x，y方向的梯度
	for(int ii=1;ii<rows-1;ii++){
		for(int jj=1;jj<cols-1;jj++){
			Grx[ii*cols+jj]  = pImg[ii*cols+jj+1]-pImg[ii*cols+jj-1];
			Gry[ii*cols+jj]  = pImg[(ii+1)*cols+jj]-pImg[(ii-1)*cols+jj];
			Grxx[ii*cols+jj] = Grx[ii*cols+jj]*Grx[ii*cols+jj];
			Gryy[ii*cols+jj] = Gry[ii*cols+jj]*Gry[ii*cols+jj];
			Grxy[ii*cols+jj] = Grx[ii*cols+jj]*Gry[ii*cols+jj];
		} 
	}

	//存储M矩阵中元素{A,B;B,C}
	float *Grxx1 = new float[rows*cols];      //A
	float *Gryy1 = new float[rows*cols];      //C
	float *Grxy1 = new float[rows*cols];      //B
	memset(Grxx1, 0, rows*cols*sizeof(float));
	memset(Gryy1, 0, rows*cols*sizeof(float));
	memset(Grxy1, 0, rows*cols*sizeof(float));

	//高斯滤波处理梯度
	Gaussian_Juanji(Grxx,Grxx1,rows,cols,Juanji_dim,Juanji_sigma);
	Gaussian_Juanji(Gryy,Gryy1,rows,cols,Juanji_dim,Juanji_sigma);
	Gaussian_Juanji(Grxy,Grxy1,rows,cols,Juanji_dim,Juanji_sigma);

	delete []Grx;delete []Gry;delete []Grxx;delete []Gryy;delete []Grxy;

	//定义数组存储每点角点响应值
	float *I = new float[rows*cols];
	memset(I,0,sizeof(float)*rows*cols);           

	//计算角点响应值
	for(int ii=0;ii<rows;ii++){
		for(int jj=0;jj<cols;jj++){
			I[ii*cols+jj] = pow(Grxx1[ii*cols+jj]*Gryy1[ii*cols+jj]-Grxy1[ii*cols+jj]*Grxy1[ii*cols+jj],2)-0.04*pow(Grxx1[ii*cols+jj]+Gryy1[ii*cols+jj],2);
		}
	}

	delete []Grxx1;delete []Gryy1;delete []Grxy1;

	//局部区域取极大值（取9*9窗口）
	//int dim=9;
	for(int ii=Localmax_dim/2;ii<rows-Localmax_dim/2;ii+=Localmax_dim){
		for(int jj=Localmax_dim/2;jj<cols-Localmax_dim/2;jj+=Localmax_dim){
			float localmax=I[ii*cols+jj];
			int xx,yy;
			xx=ii;yy=jj;
			//搜索区域最大值
			for(int a=-Localmax_dim/2;a<=Localmax_dim/2;a++){
				for(int b=-Localmax_dim/2;b<=Localmax_dim/2;b++){
					if(I[(ii+a)*cols+jj+b]>localmax && I[(ii+a)*cols+jj+b]>=threshold){
						localmax=I[(ii+a)*cols+jj+b];
						xx=ii+a;
						yy=jj+b;
					}
				}
			}
			KeyPoint_x.push_back(xx);
			KeyPoint_y.push_back(yy);
		}	
	}
	delete []I;
	return 0;
}

int Feature_Detection(char* imagepath,int ch,int thresh,std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y){
	if(ch==1){//Sift
		Mat img_1 = imread(imagepath, 0);
		if( !img_1.data)
		{ 
			std::cout<< " --(!) Error reading images " << std::endl; 
			return -1; 
		}

		SiftFeatureDetector Detector;

		std::vector<KeyPoint> keypoints_1;
		Detector.detect( img_1, keypoints_1 );

		for(int i=0;i<keypoints_1.size();i++){
			KeyPoint_x.push_back(keypoints_1[i].pt.x);
			KeyPoint_y.push_back(keypoints_1[i].pt.y);
		}
	}
	else if(ch==2){//Surf
		Mat image = imread(imagepath, 0);
		if( !image.data)
		{ 
			std::cout<< " --(!) Error reading images " << std::endl; 
			return -1; 
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=20;
		for(int i=0;i<size;i++){
			Mat img_1;
			img_1.create(rows/size,cols,CV_8UC1);
			uchar* pImg1 = img_1.data;
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			int base=rows/size*i;

			int minHessian = 300;
			SurfFeatureDetector Detector(minHessian);

			std::vector<KeyPoint> keypoints_1;
			Detector.detect( img_1, keypoints_1 );
			for(int i=0;i<keypoints_1.size();i++){
				KeyPoint_x.push_back(keypoints_1[i].pt.y+base);
				KeyPoint_y.push_back(keypoints_1[i].pt.x);
			}
		}
	}
	else if(ch==3){//opencvHarris
		Mat image = imread(imagepath, 0);
		if( !image.data)
		{ 
			std::cout<< " --(!) Error reading images " << std::endl; 
			return -1; 
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=20;
		for(int i=0;i<size;i++){
			Mat img_1;
			img_1.create(rows/size,cols,CV_8UC1);
			uchar* pImg1 = img_1.data;
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			int base=rows/size*i;


			//进行Harris角点检测找出角点
			Mat cornerStrength;
			cornerHarris(img_1, cornerStrength, 2, 3, 0.01);

			//对灰度图进行阈值操作，得到二值图并显示
			Mat harrisCorner;
			threshold(cornerStrength, harrisCorner, 0.00001, 255, THRESH_BINARY);

			for(int i=0;i<harrisCorner.rows;i++){
				for(int j=0;j<harrisCorner.cols;j++){
					if(harrisCorner.data[i*harrisCorner.cols+j]>0){
						KeyPoint_x.push_back(i);
						KeyPoint_y.push_back(j);
					}
				}
			}
		}
	}
	else if(ch==4){
		//读取影像并获取行列信息
		Mat image = imread(imagepath,0); 
		//Mat image2 = imread(imageName,1); 
		if(image.empty()) 
		{
			printf( "Could not open or find the image. \n");
			return -1;
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=20;
		for(int i=0;i<size;i++){
			uchar* pImg1 = new uchar[cols*rows/size*sizeof(uchar)];
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			int base=rows/size*i;
			Forstner(pImg1,rows/size,cols,base,KeyPoint_x,KeyPoint_y);
			delete []pImg1;
		}

	}
	else if(ch==5){//Surf
		Mat image = imread(imagepath, 0);
		if( !image.data)
		{ 
			std::cout<< " --(!) Error reading images " << std::endl; 
			return -1; 
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=20;
		for(int i=0;i<size;i++){
			Mat img_1;
			img_1.create(rows/size,cols,CV_8UC1);
			uchar* pImg1 = img_1.data;
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			int base=rows/size*i;

			int minHessian = 600;
			SurfFeatureDetector Detector(minHessian);

			std::vector<KeyPoint> keypoints_1;
			Detector.detect( img_1, keypoints_1 );
			for(int i=0;i<keypoints_1.size();i++){
				KeyPoint_x.push_back(keypoints_1[i].pt.y+base);
				KeyPoint_y.push_back(keypoints_1[i].pt.x);
			}
		}
	}
	else if(ch==6){//myHarris
		//读取影像并获取行列信息
		Mat image = imread(imagepath,0); 
		//Mat image2 = imread(imageName,1); 
		if(image.empty()) 
		{
			printf( "Could not open or find the image. \n");
			return -1;
		}
		int cols = image.cols;
		int rows = image.rows;
		unsigned char * pImg = image.data;

		int size=40;
		for(int i=0;i<size;i++){
			uchar* pImg1 = new uchar[cols*rows/size*sizeof(uchar)];
			for(int j=rows/size*i;j<rows/size*(i+1);j++){
				for(int k=0;k<cols;k++){
					pImg1[(j-rows/size*i)*cols+k]=pImg[j*cols+k];
				}
			}
			std::vector<int> KeyPoint_xx,KeyPoint_yy;
			int base=rows/size*i;
			HarrisCorner(pImg1,rows/size,cols,9,0.5,thresh,400,KeyPoint_xx,KeyPoint_yy);

			if(KeyPoint_xx.size()>0){
				for(int k=0;k<KeyPoint_xx.size();k++){
					KeyPoint_x.push_back(KeyPoint_xx[k]+base);
					KeyPoint_y.push_back(KeyPoint_yy[k]);
				}
			}

			KeyPoint_xx.swap(vector<int>());
			KeyPoint_yy.swap(vector<int>());
			delete []pImg1;
		}
	}
	else
	{
		printf("输入的特征点提取方式无效！");
		return 0;
	}
	return 0;
}

//剔除粗差
void RANSAC_ty(std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,float sigma,int* match){
	//仿射变换系数求解
	int number = KeyPoint_x1.size();  //总点数
	int iter = number*20;

	if(number<3){
		printf("匹配点对数小于3！");
		return;
	}
	int pretotal=0;     //符合拟合模型的数据的个数
	VectorXf best(6);
	VectorXf mask0(number);
	VectorXf mask(number);
	for(int i=0;i<iter;i++){
		//随机选择三组点
		int *loc = new int[3];
		srand((unsigned)time(NULL)); 
		while(1){
			loc[0]=rand()%number;
			if(loc[0]>=0 && loc[0]<number) break;
		}
		while(loc[0]>=0 && loc[0]<number)   
		{       
			loc[1]=rand()%number;        
			if(loc[1]!=loc[0] && loc[1]>=0 && loc[1]<number) break;    
		}   
		while(loc[0]>=0 && loc[0]<number && loc[1]>=0 && loc[1]<number)
		{       
			loc[2]=rand()%number;        
			if((loc[2]!=loc[1]) && (loc[2]!=loc[0]) && loc[2]>=0 && loc[2]<number) break;  
		}

		float* sample = new float[3*4];
		for(int i=0;i<3;i++){
			sample[i*4+0]=KeyPoint_x1[loc[i]];
			sample[i*4+1]=KeyPoint_y1[loc[i]];
			sample[i*4+2]=KeyPoint_x2[loc[i]];
			sample[i*4+3]=KeyPoint_y2[loc[i]];
		}

		//拟合关系式：x2=a0x+a1y+a2;y2=b0x+b1y+b2; V=Ax-L
		float *A = new float[36];
		float *L = new float[6];
		memset(A,0,sizeof(float)*36);
		memset(L,0,sizeof(float)*6);
		for(int j=0;j<3;j++){
			A[(2*j)*6+0]=sample[4*j];
			A[(2*j)*6+1]=sample[4*j+1];
			A[(2*j)*6+2]=1;
			A[(2*j+1)*6+3]=sample[4*j];
			A[(2*j+1)*6+4]=sample[4*j+1];
			A[(2*j+1)*6+5]=1;

			L[2*j]=sample[4*j+2];
			L[2*j+1]=sample[4*j+3];
		}

		MatrixXf A_ = (Map<MatrixXf>(A,6,6)).transpose();
		MatrixXf L_ = Map<MatrixXf>(L,6,1);
		VectorXf x = (A_.transpose()*A_).inverse()*A_.transpose()*L_;
		//cout << x << endl;

		MatrixXf AA=MatrixXf::Zero(2*number,6);
		MatrixXf LL=MatrixXf::Zero(2*number,1);

		for(int j=0;j<number;j++){
			VectorXf temp1(6);
			temp1 << KeyPoint_x1[j],KeyPoint_y1[j],1,0,0,0;
			AA.row(2*j)=temp1;
			VectorXf temp2(6);
			temp2 << 0,0,0,KeyPoint_x1[j],KeyPoint_y1[j],1;
			AA.row(2*j+1)=temp2;
			LL(2*j)=KeyPoint_x2[j];
			LL(2*j+1)=KeyPoint_y2[j];
		}

		VectorXf v;
		v = AA*x-LL;

		VectorXf V(number);
		for(int j=0;j<number;j++){
			V(j)=sqrt(v(2*j)*v(2*j)+v(2*j+1)*v(2*j+1));  //求每个数据到拟合关系的残差
			//printf("%f\n",V(j));
		}

		int total=0;
		for(int i=0;i<number;i++){
			if(V(i)<sigma){
				total++;
				mask0(i)=1;
			}
			else{
				mask0(i)=0;
			}
		}

		if (total>pretotal){           //找到符合拟合数据最多的拟合关系
			pretotal=total;
			best=x;          //找到最好的拟合[a0,a1,a2,b0,b1,b2]
			mask=mask0;
		}
	}

	//找到符合最佳拟合的数据
	int count0=0;
	int count=0;
	for(int i=0;i<number;i++){
		if(mask(i)==0){
			//std::vector<int>::iterator it1,it2,it3,it4;
			//it1 = KeyPoint_x1.begin()+i-count0;  KeyPoint_x1.erase(it1);
			//it2 = KeyPoint_y1.begin()+i-count0;  KeyPoint_y1.erase(it2);
			//it3 = KeyPoint_x2.begin()+i-count0;  KeyPoint_x2.erase(it3);
			//it4 = KeyPoint_y2.begin()+i-count0;  KeyPoint_y2.erase(it4);
			match[i]=-1;
			count0++;
		}
		else{
			match[i]=i;
			count=count+1;
		}
	}
}

void RANSAC_fs(std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,float sigma,int* match, float* fs_c){
	//仿射变换系数求解
	int number = KeyPoint_x1.size();  //总点数
	int iter = 2000;//number*20;

	if(number<3){
		printf("匹配点对数小于3！");
		return;
	}
	int pretotal=0;     //符合拟合模型的数据的个数
	VectorXf best(6);
	VectorXf mask0(number);
	VectorXf mask(number);
	for(int i=0;i<iter;i++){
		//随机选择三组点
		int *loc = new int[3];
		srand((unsigned)time(NULL)); 
		while(1){
			loc[0]=rand()%number;
			if(loc[0]>=0 && loc[0]<number) break;
		}
		while(loc[0]>=0 && loc[0]<number)   
		{       
			loc[1]=rand()%number;        
			if(loc[1]!=loc[0] && loc[1]>=0 && loc[1]<number) break;    
		}   
		while(loc[0]>=0 && loc[0]<number && loc[1]>=0 && loc[1]<number)
		{       
			loc[2]=rand()%number;        
			if((loc[2]!=loc[1]) && (loc[2]!=loc[0]) && loc[2]>=0 && loc[2]<number) break;  
		}

		float* sample = new float[3*4];
		for(int i=0;i<3;i++){
			sample[i*4+0]=KeyPoint_x1[loc[i]];
			sample[i*4+1]=KeyPoint_y1[loc[i]];
			sample[i*4+2]=KeyPoint_x2[loc[i]];
			sample[i*4+3]=KeyPoint_y2[loc[i]];
		}

		//拟合关系式：x2=a0x+a1y+a2;y2=b0x+b1y+b2; V=Ax-L
		float *A = new float[36];
		float *L = new float[6];
		memset(A,0,sizeof(float)*36);
		memset(L,0,sizeof(float)*6);
		for(int j=0;j<3;j++){
			A[(2*j)*6+0]=sample[4*j];
			A[(2*j)*6+1]=sample[4*j+1];
			A[(2*j)*6+2]=1;
			A[(2*j+1)*6+3]=sample[4*j];
			A[(2*j+1)*6+4]=sample[4*j+1];
			A[(2*j+1)*6+5]=1;

			L[2*j]=sample[4*j+2];
			L[2*j+1]=sample[4*j+3];
		}

		MatrixXf A_ = (Map<MatrixXf>(A,6,6)).transpose();
		MatrixXf L_ = Map<MatrixXf>(L,6,1);
		VectorXf x = (A_.transpose()*A_).inverse()*A_.transpose()*L_;
		//cout << x << endl;

		MatrixXf AA=MatrixXf::Zero(2*number,6);
		MatrixXf LL=MatrixXf::Zero(2*number,1);

		for(int j=0;j<number;j++){
			VectorXf temp1(6);
			temp1 << KeyPoint_x1[j],KeyPoint_y1[j],1,0,0,0;
			AA.row(2*j)=temp1;
			VectorXf temp2(6);
			temp2 << 0,0,0,KeyPoint_x1[j],KeyPoint_y1[j],1;
			AA.row(2*j+1)=temp2;
			LL(2*j)=KeyPoint_x2[j];
			LL(2*j+1)=KeyPoint_y2[j];
		}

		VectorXf v;
		v = AA*x-LL;

		VectorXf V(number);
		for(int j=0;j<number;j++){
			V(j)=sqrt(v(2*j)*v(2*j)+v(2*j+1)*v(2*j+1));  //求每个数据到拟合关系的残差
			//printf("%f\n",V(j));
		}

		int total=0;
		for(int i=0;i<number;i++){
			if(V(i)<sigma){
				total++;
				mask0(i)=1;
			}
			else{
				mask0(i)=0;
			}
		}

		if (total>pretotal){           //找到符合拟合数据最多的拟合关系
			pretotal=total;
			best=x;          //找到最好的拟合[a0,a1,a2,b0,b1,b2]
			mask=mask0;
		}
	}

	//找到符合最佳拟合的数据
	int count0=0;
	int count=0;
	for(int i=0;i<number;i++){
		if(mask(i)==0){
			//std::vector<int>::iterator it1,it2,it3,it4;
			//it1 = KeyPoint_x1.begin()+i-count0;  KeyPoint_x1.erase(it1);
			//it2 = KeyPoint_y1.begin()+i-count0;  KeyPoint_y1.erase(it2);
			//it3 = KeyPoint_x2.begin()+i-count0;  KeyPoint_x2.erase(it3);
			//it4 = KeyPoint_y2.begin()+i-count0;  KeyPoint_y2.erase(it4);
			match[i]=-1;
			count0++;
		}
		else{
			match[i]=i;
			count=count+1;
		}
	}
	for(int i=0;i<6;i++){
		fs_c[i]=best[i];
	}
}

void RANSAC_fs1(std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,float sigma,int* match, float* fs_c){
	//仿射变换系数求解
	int number = KeyPoint_x1.size();  //总点数
	int iter = 100;//number*20;

	if(number<3){
		printf("匹配点对数小于3！");
		return;
	}
	int pretotal=0;     //符合拟合模型的数据的个数
	VectorXf best(6);
	VectorXf mask0(number);
	VectorXf mask(number);
	for(int i=0;i<iter;i++){
		//随机选择三组点
		int *loc = new int[3];
		srand((unsigned)time(NULL)); 
		while(1){
			loc[0]=rand()%number;
			if(loc[0]>=0 && loc[0]<number) break;
		}
		while(loc[0]>=0 && loc[0]<number)   
		{       
			loc[1]=rand()%number;        
			if(loc[1]!=loc[0] && loc[1]>=0 && loc[1]<number) break;    
		}   
		while(loc[0]>=0 && loc[0]<number && loc[1]>=0 && loc[1]<number)
		{       
			loc[2]=rand()%number;        
			if((loc[2]!=loc[1]) && (loc[2]!=loc[0]) && loc[2]>=0 && loc[2]<number) break;  
		}

		float* sample = new float[3*4];
		for(int i=0;i<3;i++){
			sample[i*4+0]=KeyPoint_x1[loc[i]];
			sample[i*4+1]=KeyPoint_y1[loc[i]];
			sample[i*4+2]=KeyPoint_x2[loc[i]];
			sample[i*4+3]=KeyPoint_y2[loc[i]];
		}

		//拟合关系式：x2=a0x+a1y+a2;y2=b0x+b1y+b2; V=Ax-L
		float *A = new float[36];
		float *L = new float[6];
		memset(A,0,sizeof(float)*36);
		memset(L,0,sizeof(float)*6);
		for(int j=0;j<3;j++){
			A[(2*j)*6+0]=sample[4*j];
			A[(2*j)*6+1]=sample[4*j+1];
			A[(2*j)*6+2]=1;
			A[(2*j+1)*6+3]=sample[4*j];
			A[(2*j+1)*6+4]=sample[4*j+1];
			A[(2*j+1)*6+5]=1;

			L[2*j]=sample[4*j+2];
			L[2*j+1]=sample[4*j+3];
		}

		MatrixXf A_ = (Map<MatrixXf>(A,6,6)).transpose();
		MatrixXf L_ = Map<MatrixXf>(L,6,1);
		VectorXf x = (A_.transpose()*A_).inverse()*A_.transpose()*L_;
		//cout << x << endl;

		MatrixXf AA=MatrixXf::Zero(2*number,6);
		MatrixXf LL=MatrixXf::Zero(2*number,1);

		for(int j=0;j<number;j++){
			VectorXf temp1(6);
			temp1 << KeyPoint_x1[j],KeyPoint_y1[j],1,0,0,0;
			AA.row(2*j)=temp1;
			VectorXf temp2(6);
			temp2 << 0,0,0,KeyPoint_x1[j],KeyPoint_y1[j],1;
			AA.row(2*j+1)=temp2;
			LL(2*j)=KeyPoint_x2[j];
			LL(2*j+1)=KeyPoint_y2[j];
		}

		VectorXf v;
		v = AA*x-LL;

		VectorXf V(number);
		for(int j=0;j<number;j++){
			V(j)=sqrt(v(2*j)*v(2*j)+v(2*j+1)*v(2*j+1));  //求每个数据到拟合关系的残差
			//printf("%f\n",V(j));
		}

		int total=0;
		for(int i=0;i<number;i++){
			if(V(i)<sigma){
				total++;
				mask0(i)=1;
			}
			else{
				mask0(i)=0;
			}
		}

		if (total>pretotal){           //找到符合拟合数据最多的拟合关系
			pretotal=total;
			best=x;          //找到最好的拟合[a0,a1,a2,b0,b1,b2]
			mask=mask0;
		}
	}

	//找到符合最佳拟合的数据
	int count0=0;
	int count=0;
	for(int i=0;i<number;i++){
		if(mask(i)==0){
			//std::vector<int>::iterator it1,it2,it3,it4;
			//it1 = KeyPoint_x1.begin()+i-count0;  KeyPoint_x1.erase(it1);
			//it2 = KeyPoint_y1.begin()+i-count0;  KeyPoint_y1.erase(it2);
			//it3 = KeyPoint_x2.begin()+i-count0;  KeyPoint_x2.erase(it3);
			//it4 = KeyPoint_y2.begin()+i-count0;  KeyPoint_y2.erase(it4);
			match[i]=-1;
			count0++;
		}
		else{
			match[i]=i;
			count=count+1;
		}
	}
	for(int i=0;i<6;i++){
		fs_c[i]=best[i];
	}
}

void RANSAC_fs2(std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,float sigma,int iter,int* match, float* fs_c){
	//仿射变换系数求解
	int number = KeyPoint_x1.size();  //总点数
	//int iter = 100;//number*20;

	if(number<3){
		printf("匹配点对数小于3！");
		return;
	}
	int pretotal=0;     //符合拟合模型的数据的个数
	VectorXf best(6);
	VectorXf mask0(number);
	VectorXf mask(number);
	for(int i=0;i<iter;i++){
		//随机选择三组点
		int *loc = new int[3];
		srand((unsigned)time(NULL)); 
		while(1){
			loc[0]=rand()%number;
			if(loc[0]>=0 && loc[0]<number) break;
		}
		while(loc[0]>=0 && loc[0]<number)   
		{       
			loc[1]=rand()%number;        
			if(loc[1]!=loc[0] && loc[1]>=0 && loc[1]<number) break;    
		}   
		while(loc[0]>=0 && loc[0]<number && loc[1]>=0 && loc[1]<number)
		{       
			loc[2]=rand()%number;        
			if((loc[2]!=loc[1]) && (loc[2]!=loc[0]) && loc[2]>=0 && loc[2]<number) break;  
		}

		float* sample = new float[3*4];
		for(int i=0;i<3;i++){
			sample[i*4+0]=KeyPoint_x1[loc[i]];
			sample[i*4+1]=KeyPoint_y1[loc[i]];
			sample[i*4+2]=KeyPoint_x2[loc[i]];
			sample[i*4+3]=KeyPoint_y2[loc[i]];
		}

		//拟合关系式：x2=a0x+a1y+a2;y2=b0x+b1y+b2; V=Ax-L
		float *A = new float[36];
		float *L = new float[6];
		memset(A,0,sizeof(float)*36);
		memset(L,0,sizeof(float)*6);
		for(int j=0;j<3;j++){
			A[(2*j)*6+0]=sample[4*j];
			A[(2*j)*6+1]=sample[4*j+1];
			A[(2*j)*6+2]=1;
			A[(2*j+1)*6+3]=sample[4*j];
			A[(2*j+1)*6+4]=sample[4*j+1];
			A[(2*j+1)*6+5]=1;

			L[2*j]=sample[4*j+2];
			L[2*j+1]=sample[4*j+3];
		}

		MatrixXf A_ = (Map<MatrixXf>(A,6,6)).transpose();
		MatrixXf L_ = Map<MatrixXf>(L,6,1);
		VectorXf x = (A_.transpose()*A_).inverse()*A_.transpose()*L_;
		//cout << x << endl;

		MatrixXf AA=MatrixXf::Zero(2*number,6);
		MatrixXf LL=MatrixXf::Zero(2*number,1);

		for(int j=0;j<number;j++){
			VectorXf temp1(6);
			temp1 << KeyPoint_x1[j],KeyPoint_y1[j],1,0,0,0;
			AA.row(2*j)=temp1;
			VectorXf temp2(6);
			temp2 << 0,0,0,KeyPoint_x1[j],KeyPoint_y1[j],1;
			AA.row(2*j+1)=temp2;
			LL(2*j)=KeyPoint_x2[j];
			LL(2*j+1)=KeyPoint_y2[j];
		}

		VectorXf v;
		v = AA*x-LL;

		VectorXf V(number);
		for(int j=0;j<number;j++){
			V(j)=sqrt(v(2*j)*v(2*j)+v(2*j+1)*v(2*j+1));  //求每个数据到拟合关系的残差
			//printf("%f\n",V(j));
		}

		int total=0;
		for(int i=0;i<number;i++){
			if(V(i)<sigma){
				total++;
				mask0(i)=1;
			}
			else{
				mask0(i)=0;
			}
		}

		if (total>pretotal){           //找到符合拟合数据最多的拟合关系
			pretotal=total;
			best=x;          //找到最好的拟合[a0,a1,a2,b0,b1,b2]
			mask=mask0;
		}
	}

	//找到符合最佳拟合的数据
	int count0=0;
	int count=0;
	for(int i=0;i<number;i++){
		if(mask(i)==0){
			//std::vector<int>::iterator it1,it2,it3,it4;
			//it1 = KeyPoint_x1.begin()+i-count0;  KeyPoint_x1.erase(it1);
			//it2 = KeyPoint_y1.begin()+i-count0;  KeyPoint_y1.erase(it2);
			//it3 = KeyPoint_x2.begin()+i-count0;  KeyPoint_x2.erase(it3);
			//it4 = KeyPoint_y2.begin()+i-count0;  KeyPoint_y2.erase(it4);
			match[i]=-1;
			count0++;
		}
		else{
			match[i]=i;
			count=count+1;
		}
	}
	for(int i=0;i<6;i++){
		fs_c[i]=best[i];
	}
}

//绘制匹配结果线对
void drawMatch(Mat imgL0,Mat imgR0,std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int sf){
	Mat imgL,imgR;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	imgR.create(imgR0.rows/sf,imgR0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			imgR.data[i*imgR.cols+j]=imgR0.data[sf*i*imgR0.cols+sf*j];
		}
	}
	int rows= imgL.rows>imgR.rows ? imgL.rows : imgR.rows;
	int cols= imgL.cols+imgR.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*imgL.cols+j)];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			img.data[3*(i*cols+imgL.cols+j)+0]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+1]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+2]=imgR.data[(i*imgR.cols+j)];
		}
	}
	int P_numL = KeyPoint_x1.size();
	for(int i=0;i<P_numL;i++){
		if(Match[i]!=-1){
			int r=int(float(rand())/RAND_MAX+0.5)*255;
			int g=int(float(rand())/RAND_MAX+0.5)*255;
			int b=int(float(rand())/RAND_MAX+0.5)*255;
			//printf("%d %d\n",kpL[i].row,kpL[i].col);
			circle(img,cvPoint(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),3,CV_RGB(r,g,b),0.5,8,0);
			circle(img,cvPoint(KeyPoint_y2[Match[i]]/sf+imgL.cols,KeyPoint_x2[Match[i]]/sf),3,CV_RGB(r,g,b),0.5,8,0);
			line(img,cvPoint(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),cvPoint(KeyPoint_y2[Match[i]]/sf+imgL.cols,KeyPoint_x2[Match[i]]/sf),CV_RGB(r,g,b),0.5,8,0);
			//line(img,cvPoint(0,0),cvPoint(200,200),CV_RGB(0,0,255),0.5,8,0);
		}
	}

	//imshow("Match",img);
	imwrite("../out/match.jpg",img);
	//waitKey(0);
}

void drawMatch1(Mat imgL0,Mat imgR0,std::vector<int> matched,int sf){
	Mat imgL,imgR;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	imgR.create(imgR0.rows/sf,imgR0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			imgR.data[i*imgR.cols+j]=imgR0.data[sf*i*imgR0.cols+sf*j];
		}
	}
	int rows= imgL.rows>imgR.rows ? imgL.rows : imgR.rows;
	int cols= imgL.cols+imgR.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*imgL.cols+j)];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			img.data[3*(i*cols+imgL.cols+j)+0]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+1]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+2]=imgR.data[(i*imgR.cols+j)];
		}
	}
	int P_numL = matched.size()/4;
	for(int i=0;i<P_numL;i++){
		int row1= matched[4*i];
		int col1 = matched[4*i+1];
		int row2 = matched[4*i+2];
		int col2 = matched[4*i+3];
		int r=int(float(rand())/RAND_MAX+0.5)*255;
		int g=int(float(rand())/RAND_MAX+0.5)*255;
		int b=int(float(rand())/RAND_MAX+0.5)*255;
		//printf("%d %d\n",kpL[i].row,kpL[i].col);
		circle(img,cvPoint(col1/sf,row1/sf),3,CV_RGB(r,g,b),0.5,8,0);
		circle(img,cvPoint(col2/sf+imgL.cols,row2/sf),3,CV_RGB(r,g,b),0.5,8,0);
		line(img,cvPoint(col1/sf,row1/sf),cvPoint(col2/sf+imgL.cols,row2/sf),CV_RGB(r,g,b),0.5,8,0);
		//line(img,cvPoint(0,0),cvPoint(200,200),CV_RGB(0,0,255),0.5,8,0);

	}

	//imshow("Match",img);
	imwrite("../out/match.jpg",img);
	//waitKey(0);
}

void drawMatch2(char* imagepath1,char* imagepath2, std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int sf){
	Mat imgL0 = imread(imagepath1, 0);
	Mat imgR0 = imread(imagepath2, 0);
	if((!imgL0.data)||(!imgR0.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return; 
	}

	Mat imgL,imgR;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	imgR.create(imgR0.rows/sf,imgR0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			imgR.data[i*imgR.cols+j]=imgR0.data[sf*i*imgR0.cols+sf*j];
		}
	}
	int rows= imgL.rows>imgR.rows ? imgL.rows : imgR.rows;
	int cols= imgL.cols+imgR.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*imgL.cols+j)];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			img.data[3*(i*cols+imgL.cols+j)+0]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+1]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+2]=imgR.data[(i*imgR.cols+j)];
		}
	}
	int P_numL = KeyPoint_x1.size();
	for(int i=0;i<P_numL;i++){
		//if(Match[i]!=-1){
			int r=int(float(rand())/RAND_MAX+0.5)*255;
			int g=int(float(rand())/RAND_MAX+0.5)*255;
			int b=int(float(rand())/RAND_MAX+0.5)*255;
			//printf("%d %d\n",kpL[i].row,kpL[i].col);
			circle(img,cvPoint(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),6,CV_RGB(r,g,b),0.5,8,0);
			circle(img,cvPoint(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),6,CV_RGB(r,g,b),0.5,8,0);
			line(img,cvPoint(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),cvPoint(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),CV_RGB(r,g,b),0.5,8,0);
			//line(img,cvPoint(0,0),cvPoint(200,200),CV_RGB(0,0,255),0.5,8,0);
		//}
	}

	//imshow("Match",img);
	imwrite("../out/match_.jpg",img);
	//waitKey(0);
}

void drawMatch3(char* imagepath1,char* imagepath2, std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int sf){
	Mat imgL0 = imread(imagepath1, 0);
	Mat imgR0 = imread(imagepath2, 0);
	if((!imgL0.data)||(!imgR0.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return; 
	}

	Mat imgL,imgR;
	imgL.create(imgL0.rows/sf,imgL0.cols/sf,CV_8UC1);
	imgR.create(imgR0.rows/sf,imgR0.cols/sf,CV_8UC1);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			imgL.data[i*imgL.cols+j]=imgL0.data[sf*i*imgL0.cols+sf*j];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			imgR.data[i*imgR.cols+j]=imgR0.data[sf*i*imgR0.cols+sf*j];
		}
	}
	int rows= imgL.rows>imgR.rows ? imgL.rows : imgR.rows;
	int cols= imgL.cols+imgR.cols;
	Mat img;
	img.create(rows,cols,CV_8UC3);
	for(int i=0;i<imgL.rows;i++){
		for(int j=0;j<imgL.cols;j++){
			img.data[3*(i*cols+j)+0]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+1]=imgL.data[(i*imgL.cols+j)];
			img.data[3*(i*cols+j)+2]=imgL.data[(i*imgL.cols+j)];
		}
	}
	for(int i=0;i<imgR.rows;i++){
		for(int j=0;j<imgR.cols;j++){
			img.data[3*(i*cols+imgL.cols+j)+0]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+1]=imgR.data[(i*imgR.cols+j)];
			img.data[3*(i*cols+imgL.cols+j)+2]=imgR.data[(i*imgR.cols+j)];
		}
	}
	int P_numL = KeyPoint_x1.size();
	for(int i=0;i<P_numL;i++){
		if(Match[i]!=-1){
			int r=int(float(rand())/RAND_MAX+0.5)*255;
			int g=int(float(rand())/RAND_MAX+0.5)*255;
			int b=int(float(rand())/RAND_MAX+0.5)*255;
			//printf("%d %d\n",kpL[i].row,kpL[i].col);
			circle(img,cvPoint(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),6,CV_RGB(r,g,b),0.5,8,0);
			circle(img,cvPoint(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),6,CV_RGB(r,g,b),0.5,8,0);
			//line(img,cvPoint(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),cvPoint(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),CV_RGB(r,g,b),0.5,8,0);
			//line(img,cvPoint(0,0),cvPoint(200,200),CV_RGB(0,0,255),0.5,8,0);
		}
		else{
			circle(img,cvPoint(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),6,CV_RGB(255,0,0),0.5,8,0);
			circle(img,cvPoint(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),6,CV_RGB(255,0,0),0.5,8,0);
			line(img,cvPoint(KeyPoint_y1[i]/sf,KeyPoint_x1[i]/sf),cvPoint(KeyPoint_y2[i]/sf+imgL.cols,KeyPoint_x2[i]/sf),CV_RGB(255,0,0),0.5,8,0);
		}
	}

	//imshow("Match",img);
	imwrite("../out/match.jpg",img);
	//waitKey(0);
}

//特征点匹配
int CC_match(char* imagepath1,char* imagepath2,int w_size,float threshold,int* dr_,int* dc_,int ch){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	Feature_Detection(imagepath1,ch,49,KeyPoint_x1,KeyPoint_y1);
	Feature_Detection(imagepath2,ch,49,KeyPoint_x2,KeyPoint_y2);

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matched = new int[KeyPoint_x1.size()];
	memset(matched,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float maxCC=-1000;
	int count=0;
	for(int i=0;i<KeyPoint_x1.size();i++){
		maxCC=-1000;
		int grey1=img_1.data[KeyPoint_x1[i]*cols1+KeyPoint_y1[i]];
		for(int j=0;j<KeyPoint_x2.size();j++){
			if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size      && KeyPoint_x2[j]>=w_size     && KeyPoint_y2[j]>=w_size && 
				KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size && KeyPoint_x2[j]<=rows2-w_size && KeyPoint_y2[j]<=cols2-w_size)
			{
				s12=0;s11=0;s22=0;s1=0;s2=0;
				int grey2=img_2.data[KeyPoint_x2[j]*cols2+KeyPoint_y2[j]];
				int sum1=0;int sum2=0;
				for(int k=-w_size/2;k<=w_size/2;k++){
					for(int m=-w_size/2;m<=w_size/2;m++){
						r1=KeyPoint_x1[i]+k;r2=KeyPoint_x2[j]+k;
						c1=KeyPoint_y1[i]+m;c2=KeyPoint_y2[j]+m;
						//r1=KeyPoint_x1[i];r2=KeyPoint_x2[j];
						//c1=KeyPoint_y1[i];c2=KeyPoint_y2[j];
						sum1 += img_1.data[r1*cols1+c1]-grey1;
						sum2 += img_2.data[r2*cols2+c2]-grey2;
						s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
						s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
						s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
						s1  += (img_1.data[r1*cols1+c1]-grey1);
						s2  += (img_2.data[r2*cols2+c2]-grey2);
					}
				}
				s12 = s12/sum1/sum2;
				s22 = s22/sum2/sum2;
				s11 = s11/sum1/sum1;
				s1 = s1/sum1;
				s2 = s2/sum2;
				CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
				if(CC>maxCC && CC>=threshold){
					maxCC=CC;
					matched[i]=j;
					C_match[i]=maxCC;
				}
			}
		}
	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	count = KeyPoint_x1.size();
	int dr=0;
	int dc=0;
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[matched[i]]);
			KeyPoint_y22.push_back(KeyPoint_y2[matched[i]]);
			dr += KeyPoint_x2[matched[i]]-KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]]-KeyPoint_y1[i];
		}
		else{
			//matched[i]=-1;
			count--;  
		}
	}

	dr = int(float(dr)/count+0.5);
	dc = int(float(dc)/count+0.5);

	*dr_ = dr;
	*dc_ = dc;

	int * match = new int[KeyPoint_x11.size()];
	memset(match,-1,sizeof(int)*KeyPoint_x11.size());
	//RANSAC_ty(KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,20,match);  //剔除粗差

	int sf = 2;
	drawMatch(img_1,img_2,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,matched,sf);
	//drawMatch(img_1,img_2,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,sf);
	delete [] matched;
	delete [] C_match;
	KeyPoint_x1.swap(vector<int>());
	KeyPoint_y1.swap(vector<int>());
	KeyPoint_x2.swap(vector<int>());
	KeyPoint_y2.swap(vector<int>());
	KeyPoint_x11.swap(vector<int>());
	KeyPoint_y11.swap(vector<int>());
	KeyPoint_x22.swap(vector<int>());
	KeyPoint_y22.swap(vector<int>());

	return 0;
}

int CC_match1(char* imagepath1,char* imagepath2,int w_size,float threshold,int ch,float* fs_c,char* matchpointxt){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	Feature_Detection(imagepath1,ch,49,KeyPoint_x1,KeyPoint_y1);
	Feature_Detection(imagepath2,ch,49,KeyPoint_x2,KeyPoint_y2);

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matched = new int[KeyPoint_x1.size()];
	memset(matched,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float maxCC=-1000;
	int count=0;
	for(int i=0;i<KeyPoint_x1.size();i++){
		maxCC=-1000;
		int grey1=img_1.data[KeyPoint_x1[i]*cols1+KeyPoint_y1[i]];
		for(int j=0;j<KeyPoint_x2.size();j++){
			if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size      && KeyPoint_x2[j]>=w_size     && KeyPoint_y2[j]>=w_size && 
				KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size && KeyPoint_x2[j]<=rows2-w_size && KeyPoint_y2[j]<=cols2-w_size)
			{
				s12=0;s11=0;s22=0;s1=0;s2=0;
				int grey2=img_2.data[KeyPoint_x2[j]*cols2+KeyPoint_y2[j]];
				int sum1=0;int sum2=0;
				for(int k=-w_size/2;k<=w_size/2;k++){
					for(int m=-w_size/2;m<=w_size/2;m++){
						r1=KeyPoint_x1[i]+k;r2=KeyPoint_x2[j]+k;
						c1=KeyPoint_y1[i]+m;c2=KeyPoint_y2[j]+m;
						//r1=KeyPoint_x1[i];r2=KeyPoint_x2[j];
						//c1=KeyPoint_y1[i];c2=KeyPoint_y2[j];
						sum1 += img_1.data[r1*cols1+c1]-grey1;
						sum2 += img_2.data[r2*cols2+c2]-grey2;
						s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
						s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
						s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
						s1  += (img_1.data[r1*cols1+c1]-grey1);
						s2  += (img_2.data[r2*cols2+c2]-grey2);
					}
				}
				s12 = s12/sum1/sum2;
				s22 = s22/sum2/sum2;
				s11 = s11/sum1/sum1;
				s1 = s1/sum1;
				s2 = s2/sum2;
				CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
				if(CC>maxCC && CC>=threshold){
					maxCC=CC;
					matched[i]=j;
					C_match[i]=maxCC;
				}
			}
		}
	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	count = KeyPoint_x1.size();
	int dr=0;
	int dc=0;
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[matched[i]]);
			KeyPoint_y22.push_back(KeyPoint_y2[matched[i]]);
			dr += KeyPoint_x2[matched[i]]-KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]]-KeyPoint_y1[i];
		}
		else{
			//matched[i]=-1;
			count--;  
		}
	}

	dr = int(float(dr)/count+0.5);
	dc = int(float(dc)/count+0.5);

	int * match = new int[KeyPoint_x11.size()];
	memset(match,-1,sizeof(int)*KeyPoint_x11.size());
	//float* fs_c = new float[6];
	RANSAC_fs(KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,5,match,fs_c);  //剔除粗差

	FILE *fp=fopen(matchpointxt,"w");
	int countMP=0;
	for(int i=0;i<KeyPoint_x11.size();i++){
		if(match[i]!=-1){
			fprintf(fp,"%d %d %d %d %d\n",countMP,KeyPoint_x11[i],KeyPoint_y11[i],KeyPoint_x22[i],KeyPoint_y22[i]);
			countMP++;
		}
	}

	int sf = 2;
	//drawMatch(img_1,img_2,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,matched,sf);
	drawMatch(img_1,img_2,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,sf);
	delete [] matched;
	delete [] C_match;
	KeyPoint_x1.swap(vector<int>());
	KeyPoint_y1.swap(vector<int>());
	KeyPoint_x2.swap(vector<int>());
	KeyPoint_y2.swap(vector<int>());
	KeyPoint_x11.swap(vector<int>());
	KeyPoint_y11.swap(vector<int>());
	KeyPoint_x22.swap(vector<int>());
	KeyPoint_y22.swap(vector<int>());

	return 0;
}

int CC_match2(char* imagepath1,char* imagepath2,int w_size,float threshold,int ch,float* fs_c,int RCCD_id, char* featurepointxt_1,char* featurepointxt_2,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,row,col,imgID,mrow,mcol;
	float mscore;

	FILE *fp1=fopen(featurepointxt_1,"r");
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);

	FILE *fp2=fopen(featurepointxt_2,"r"); 
	while(!feof(fp2)){
		fscanf(fp2,"%d ",&bj);
		if(bj==1){
			fscanf(fp2,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
		}
		else{
			fscanf(fp2,"%d %d\n",&row,&col);
		}
		KeyPoint_x2.push_back(row);
		KeyPoint_y2.push_back(col);
	}
	fclose(fp2);

	//Feature_Detection(imagepath1,ch,KeyPoint_x1,KeyPoint_y1);
	//Feature_Detection(imagepath2,ch,KeyPoint_x2,KeyPoint_y2);

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matched = new int[KeyPoint_x1.size()];
	memset(matched,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float maxCC=-1000;
	int count=0;
	for(int i=0;i<KeyPoint_x1.size();i++){
		maxCC=-1000;
		int grey1=img_1.data[KeyPoint_x1[i]*cols1+KeyPoint_y1[i]];
		for(int j=0;j<KeyPoint_x2.size();j++){
			if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size      && KeyPoint_x2[j]>=w_size     && KeyPoint_y2[j]>=w_size && 
				KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size && KeyPoint_x2[j]<=rows2-w_size && KeyPoint_y2[j]<=cols2-w_size)
			{
				s12=0;s11=0;s22=0;s1=0;s2=0;
				int grey2=img_2.data[KeyPoint_x2[j]*cols2+KeyPoint_y2[j]];
				int sum1=0;int sum2=0;
				for(int k=-w_size/2;k<=w_size/2;k++){
					for(int m=-w_size/2;m<=w_size/2;m++){
						r1=KeyPoint_x1[i]+k;r2=KeyPoint_x2[j]+k;
						c1=KeyPoint_y1[i]+m;c2=KeyPoint_y2[j]+m;
						//r1=KeyPoint_x1[i];r2=KeyPoint_x2[j];
						//c1=KeyPoint_y1[i];c2=KeyPoint_y2[j];
						sum1 += img_1.data[r1*cols1+c1]-grey1;
						sum2 += img_2.data[r2*cols2+c2]-grey2;
						s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
						s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
						s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
						s1  += (img_1.data[r1*cols1+c1]-grey1);
						s2  += (img_2.data[r2*cols2+c2]-grey2);
					}
				}
				s12 = s12/sum1/sum2;
				s22 = s22/sum2/sum2;
				s11 = s11/sum1/sum1;
				s1 = s1/sum1;
				s2 = s2/sum2;
				CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
				if(CC>maxCC && CC>=threshold){
					maxCC=CC;
					matched[i]=j;
					C_match[i]=maxCC;
				}
			}
		}
	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	count = KeyPoint_x1.size();
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[matched[i]]);
			KeyPoint_y22.push_back(KeyPoint_y2[matched[i]]);
			dr += KeyPoint_x2[matched[i]]-KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]]-KeyPoint_y1[i];
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
				}
				else{
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				}
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",1,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d\n",bj,row,col);
			}
			//matched[i]=-1;
			count--;  
		}
	}
	fclose(fp3);
	fclose(fp4);

	dr = int(float(dr)/count+0.5);
	dc = int(float(dc)/count+0.5);

	//int * match = new int[KeyPoint_x11.size()];
	//memset(match,-1,sizeof(int)*KeyPoint_x11.size());
	//float* fs_c = new float[6];
	//RANSAC_fs(KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,5,match,fs_c);  //剔除粗差

	/*
	FILE *fp=fopen(matchpointxt,"w");
	int countMP=0;
	for(int i=0;i<KeyPoint_x11.size();i++){
		if(match[i]!=-1){
			fprintf(fp,"%d %d %d %d %d\n",countMP,KeyPoint_x11[i],KeyPoint_y11[i],KeyPoint_x22[i],KeyPoint_y22[i]);
			countMP++;
		}
	}
	*/

	int sf = 2;
	drawMatch(img_1,img_2,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,matched,sf);
	//drawMatch(img_1,img_2,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,sf);
	delete [] matched;
	delete [] C_match;
	KeyPoint_x1.swap(vector<int>());
	KeyPoint_y1.swap(vector<int>());
	KeyPoint_x2.swap(vector<int>());
	KeyPoint_y2.swap(vector<int>());
	KeyPoint_x11.swap(vector<int>());
	KeyPoint_y11.swap(vector<int>());
	KeyPoint_x22.swap(vector<int>());
	KeyPoint_y22.swap(vector<int>());
	return 0;
}

int limit_match(char* imagepath1,char* imagepath2,int w_size,float threshold,int ch,float* fs_c,char* matchpointxt){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	Feature_Detection(imagepath1,ch,49,KeyPoint_x1,KeyPoint_y1);
	Feature_Detection(imagepath2,ch,49,KeyPoint_x2,KeyPoint_y2);

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matched = new int[KeyPoint_x1.size()];
	memset(matched,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float maxCC=-1000;
	int count=0;
	for(int i=0;i<KeyPoint_x1.size();i++){
		maxCC=-1000;
		int grey1=img_1.data[KeyPoint_x1[i]*cols1+KeyPoint_y1[i]];
		for(int j=0;j<KeyPoint_x2.size();j++){
			if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size      && KeyPoint_x2[j]>=w_size     && KeyPoint_y2[j]>=w_size && 
				KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size && KeyPoint_x2[j]<=rows2-w_size && KeyPoint_y2[j]<=cols2-w_size&&
				abs(KeyPoint_x2[j]-(fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2]))<30 && 
				abs(KeyPoint_y2[j]-(fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5]))<30)
			{
				s12=0;s11=0;s22=0;s1=0;s2=0;
				int grey2=img_2.data[KeyPoint_x2[j]*cols2+KeyPoint_y2[j]];
				int sum1=0;int sum2=0;
				for(int k=-w_size/2;k<=w_size/2;k++){
					for(int m=-w_size/2;m<=w_size/2;m++){
						r1=KeyPoint_x1[i]+k;r2=KeyPoint_x2[j]+k;
						c1=KeyPoint_y1[i]+m;c2=KeyPoint_y2[j]+m;
						//r1=KeyPoint_x1[i];r2=KeyPoint_x2[j];
						//c1=KeyPoint_y1[i];c2=KeyPoint_y2[j];
						sum1 += img_1.data[r1*cols1+c1]-grey1;
						sum2 += img_2.data[r2*cols2+c2]-grey2;
						s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
						s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
						s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
						s1  += (img_1.data[r1*cols1+c1]-grey1);
						s2  += (img_2.data[r2*cols2+c2]-grey2);
					}
				}
				s12 = s12/sum1/sum2;
				s22 = s22/sum2/sum2;
				s11 = s11/sum1/sum1;
				s1 = s1/sum1;
				s2 = s2/sum2;
				CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
				if(CC>maxCC && CC>=threshold){
					maxCC=CC;
					matched[i]=j;
					C_match[i]=maxCC;
				}
			}
		}
	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	count = KeyPoint_x1.size();
	int dr=0;
	int dc=0;
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[matched[i]]);
			KeyPoint_y22.push_back(KeyPoint_y2[matched[i]]);
			dr += KeyPoint_x2[matched[i]]-KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]]-KeyPoint_y1[i];
		}
		else{
			//matched[i]=-1;
			count--;  
		}
	}

	dr = int(float(dr)/count+0.5);
	dc = int(float(dc)/count+0.5);

	int * match = new int[KeyPoint_x11.size()];
	memset(match,-1,sizeof(int)*KeyPoint_x11.size());
	//float* fs_c = new float[6];
	RANSAC_fs(KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,5,match,fs_c);  //剔除粗差

	//输出匹配点坐标
	//char* matchpointxt = new char[80];
	//sprintf( matchpointxt, "%s", ".\\matchpointxt.txt");
	FILE *fp=fopen(matchpointxt,"w");
	int countMP=0;
	for(int i=0;i<KeyPoint_x11.size();i++){
		if(match[i]!=-1){
			fprintf(fp,"%d %d %d %d %d\n",countMP,KeyPoint_x11[i],KeyPoint_y11[i],KeyPoint_x22[i],KeyPoint_y22[i]);
			countMP++;
		}
	}

	int sf = 2;
	//drawMatch(img_1,img_2,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,matched,sf);
	drawMatch(img_1,img_2,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,sf);
	delete [] matched;
	delete [] C_match;
	KeyPoint_x1.swap(vector<int>());
	KeyPoint_y1.swap(vector<int>());
	KeyPoint_x2.swap(vector<int>());
	KeyPoint_y2.swap(vector<int>());
	KeyPoint_x11.swap(vector<int>());
	KeyPoint_y11.swap(vector<int>());
	KeyPoint_x22.swap(vector<int>());
	KeyPoint_y22.swap(vector<int>());

	return 0;
}

int limit_match1(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int ch,float* fs_c,int RCCD_id, char* featurepointxt_1,char* featurepointxt_2,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,row,col,imgID,mrow,mcol;
	float mscore;

	FILE *fp1=fopen(featurepointxt_1,"r");
	FILE *fp11=fopen(outpointxt_1,"w"); 
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		fprintf(fp11,"%d ",bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp11,"%d %d %d %d %d %f\n",row,col,imgID,mrow,mcol,mscore);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
			fprintf(fp11,"%d %d\n",row,col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);
	fclose(fp11);

	FILE *fp2=fopen(featurepointxt_2,"r"); 
	while(!feof(fp2)){
		fscanf(fp2,"%d ",&bj);
		if(bj==1){
			fscanf(fp2,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
		}
		else{
			fscanf(fp2,"%d %d\n",&row,&col);
		}
		KeyPoint_x2.push_back(row);
		KeyPoint_y2.push_back(col);
	}
	fclose(fp2);

	//Feature_Detection(imagepath1,ch,KeyPoint_x1,KeyPoint_y1);
	//Feature_Detection(imagepath2,ch,KeyPoint_x2,KeyPoint_y2);

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matched = new int[KeyPoint_x1.size()];
	memset(matched,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float maxCC=-1000;
	int count=0;
	if((fs_c[3]*0+fs_c[4]*0+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]<0) || (fs_c[3]*0+fs_c[4]*0+fs_c[5]>cols2 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]>cols2)){
		return 0;
	}
	for(int i=0;i<KeyPoint_x1.size();i++){
		maxCC=-1000;
		int grey1=img_1.data[KeyPoint_x1[i]*cols1+KeyPoint_y1[i]];
		for(int j=0;j<KeyPoint_x2.size();j++){
			if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size      && KeyPoint_x2[j]>=w_size     && KeyPoint_y2[j]>=w_size && 
				KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size && KeyPoint_x2[j]<=rows2-w_size && KeyPoint_y2[j]<=cols2-w_size&&
				abs(KeyPoint_x2[j]-(fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2]))<ser_range && 
				abs(KeyPoint_y2[j]-(fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5]))<ser_range)
			{
				s12=0;s11=0;s22=0;s1=0;s2=0;
				int grey2=img_2.data[KeyPoint_x2[j]*cols2+KeyPoint_y2[j]];
				int sum1=0;int sum2=0;
				for(int k=-w_size/2;k<=w_size/2;k++){
					for(int m=-w_size/2;m<=w_size/2;m++){
						r1=KeyPoint_x1[i]+k;r2=KeyPoint_x2[j]+k;
						c1=KeyPoint_y1[i]+m;c2=KeyPoint_y2[j]+m;
						//r1=KeyPoint_x1[i];r2=KeyPoint_x2[j];
						//c1=KeyPoint_y1[i];c2=KeyPoint_y2[j];
						sum1 += img_1.data[r1*cols1+c1]-grey1;
						sum2 += img_2.data[r2*cols2+c2]-grey2;
						s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
						s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
						s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
						s1  += (img_1.data[r1*cols1+c1]-grey1);
						s2  += (img_2.data[r2*cols2+c2]-grey2);
					}
				}
				s12 = s12/sum1/sum2;
				s22 = s22/sum2/sum2;
				s11 = s11/sum1/sum1;
				s1 = s1/sum1;
				s2 = s2/sum2;
				CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
				if(CC>maxCC && CC>=threshold){
					maxCC=CC;
					matched[i]=j;
					C_match[i]=maxCC;
				}
			}
		}
	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	count = KeyPoint_x1.size();
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[matched[i]]);
			KeyPoint_y22.push_back(KeyPoint_y2[matched[i]]);
			dr += KeyPoint_x2[matched[i]]-KeyPoint_x1[i];
			dc += KeyPoint_y2[matched[i]]-KeyPoint_y1[i];
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
				}
				else{
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				}
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",1,row,col,RCCD_id,KeyPoint_x2[matched[i]],KeyPoint_y2[matched[i]],C_match[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d\n",bj,row,col);
			}
			//matched[i]=-1;
			count--;  
		}
	}
	fclose(fp3);
	fclose(fp4);

	dr = int(float(dr)/count+0.5);
	dc = int(float(dc)/count+0.5);

	//int * match = new int[KeyPoint_x11.size()];
	//memset(match,-1,sizeof(int)*KeyPoint_x11.size());
	//float* fs_c = new float[6];
	//RANSAC_fs(KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,5,match,fs_c);  //剔除粗差

	/*
	FILE *fp=fopen(matchpointxt,"w");
	int countMP=0;
	for(int i=0;i<KeyPoint_x11.size();i++){
		if(match[i]!=-1){
			fprintf(fp,"%d %d %d %d %d\n",countMP,KeyPoint_x11[i],KeyPoint_y11[i],KeyPoint_x22[i],KeyPoint_y22[i]);
			countMP++;
		}
	}
	*/

	int sf = 2;
	drawMatch(img_1,img_2,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,matched,sf);
	//drawMatch(img_1,img_2,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match,sf);
	delete [] matched;
	delete [] C_match;
	KeyPoint_x1.swap(vector<int>());
	KeyPoint_y1.swap(vector<int>());
	KeyPoint_x2.swap(vector<int>());
	KeyPoint_y2.swap(vector<int>());
	KeyPoint_x11.swap(vector<int>());
	KeyPoint_y11.swap(vector<int>());
	KeyPoint_x22.swap(vector<int>());
	KeyPoint_y22.swap(vector<int>());

	return 0;
}

int limit_grid(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	std::vector<int> Biaoji;

	int bj,row,col,imgID,mrow,mcol;
	float mscore;

	FILE *fp1=fopen(featurepointxt_1,"r");
	FILE *fp11=fopen(outpointxt_1,"w"); 
	while(!feof(fp1)){
		fscanf(fp1,"%d ",&bj);
		fprintf(fp11,"%d ",bj);
		if(bj==1){
			fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
			fprintf(fp11,"%d %d %d %d %d %f\n",row,col,imgID,mrow,mcol,mscore);
		}
		else{
			fscanf(fp1,"%d %d\n",&row,&col);
			fprintf(fp11,"%d %d\n",row,col);
		}
		Biaoji.push_back(bj);
		KeyPoint_x1.push_back(row);
		KeyPoint_y1.push_back(col);
	}
	fclose(fp1);
	fclose(fp11);

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	int * matchedx = new int[KeyPoint_x1.size()];
	memset(matchedx,-1,sizeof(int)*KeyPoint_x1.size());
	int * matchedy = new int[KeyPoint_x1.size()];
	memset(matchedy,-1,sizeof(int)*KeyPoint_x1.size());
	float * C_match = new float[KeyPoint_x1.size()];
	memset(C_match,0,sizeof(float)*KeyPoint_x1.size());
	float maxCC=-1000;

	if((fs_c[3]*0+fs_c[4]*0+fs_c[5]<0 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]<0) || (fs_c[3]*0+fs_c[4]*0+fs_c[5]>cols2 && fs_c[3]*0+fs_c[4]*cols1+fs_c[5]>cols2)){
		return 0;
	}
	for(int i=0;i<KeyPoint_x1.size();i++){
		maxCC=-1000;

		int xx=fs_c[0]*KeyPoint_x1[i]+fs_c[1]*KeyPoint_y1[i]+fs_c[2];
		int yy=fs_c[3]*KeyPoint_x1[i]+fs_c[4]*KeyPoint_y1[i]+fs_c[5];
		if(KeyPoint_x1[i]>=w_size       && KeyPoint_y1[i]>=w_size       && //xx-ser_range>=w_size       && yy-ser_range>=w_size && 
			KeyPoint_x1[i]<=rows1-w_size && KeyPoint_y1[i]<=cols1-w_size)   //&& xx+ser_range<=rows2-w_size && yy+ser_range<=cols2-w_size)
		{
			int grey1=img_1.data[KeyPoint_x1[i]*cols1+KeyPoint_y1[i]];
			//最小二乘匹配
			for(int ii=-ser_range;ii<ser_range;ii++){
				for(int jj=-ser_range;jj<ser_range;jj++){

					int rr=xx+ii;
					int cc=yy+jj;
					if(rr>=w_size && rr<=rows2-w_size && cc>=w_size && cc<=cols2-w_size){
						s12=0;s11=0;s22=0;s1=0;s2=0;
						int grey2=img_2.data[rr*cols2+cc];
						int sum1=0;int sum2=0;
						for(int k=-w_size/2;k<=w_size/2;k++){
							for(int m=-w_size/2;m<=w_size/2;m++){
								r1=KeyPoint_x1[i]+k;r2=rr+k;
								c1=KeyPoint_y1[i]+m;c2=cc+m;
								sum1 += img_1.data[r1*cols1+c1]-grey1;
								sum2 += img_2.data[r2*cols2+c2]-grey2;
								s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
								s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
								s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
								s1  += (img_1.data[r1*cols1+c1]-grey1);
								s2  += (img_2.data[r2*cols2+c2]-grey2);
							}
						}
						s12 = s12/float(sum1)/float(sum2);
						s22 = s22/float(sum2)/float(sum2);
						s11 = s11/float(sum1)/float(sum1);
						s1 = s1/sum1;
						s2 = s2/sum2;
						CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
						if(CC>maxCC){
							maxCC=CC;
							matchedx[i]=rr;
							matchedy[i]=cc;
							C_match[i]=maxCC;
						}
					}
				}
			}
		}

	}

	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;  //x为行，y为列
	int dr=0;
	int dc=0;
	FILE *fp3=fopen(featurepointxt_1,"r");
	FILE *fp4=fopen(outpointxt_1,"w"); 
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(matchedx[i]);
			KeyPoint_y22.push_back(matchedy[i]);
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				if(mscore<C_match[i]){
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i]);
				}
				else{
					fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
				}
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",1,row,col,RCCD_id,matchedx[i],matchedy[i],C_match[i]);
			}
		}
		else{
			if(Biaoji[i]==1){
				fscanf(fp3,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&mscore);
				fprintf(fp4,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,mscore);
			}
			else{
				fscanf(fp3,"%d %d %d\n",&bj,&row,&col);
				fprintf(fp4,"%d %d %d\n",bj,row,col);
			}
		}
	}
	fclose(fp3);
	fclose(fp4);


	delete [] matchedx;
	delete [] matchedy;
	delete [] C_match;
	KeyPoint_x1.swap(vector<int>());
	KeyPoint_y1.swap(vector<int>());
	KeyPoint_x2.swap(vector<int>());
	KeyPoint_y2.swap(vector<int>());
	KeyPoint_x11.swap(vector<int>());
	KeyPoint_y11.swap(vector<int>());
	KeyPoint_x22.swap(vector<int>());
	KeyPoint_y22.swap(vector<int>());

	return 0;
}

int intra_CCD_match(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int batch_size,int CCD_id,char* outpointxt_1){

	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列

	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;

	std::vector<int> matchedx,matchedy;
	std::vector<float> C_match;
	
	int count=-1;
	//int batch_size = 48;
	for(int i=0;(i+1)*batch_size<=rows1;i++){
		for(int j=0;(j+1)*batch_size<=48;j++){
			int r=i*batch_size+batch_size/2;
			int c=j*batch_size+batch_size/2+cols1-48;

			if(r>w_size/2 && c>w_size/2 && r<rows1-w_size && c<cols1-w_size){
				KeyPoint_x1.push_back(r);KeyPoint_y1.push_back(c);
				int grey1=img_1.data[r*cols1+c];
				count++;
				float maxCC=-1000;

				matchedx.push_back(r);matchedy.push_back(c);
				C_match.push_back(-1);

				//最小二乘匹配
				for(int ii=-ser_range;ii<ser_range;ii++){
					for(int jj=-ser_range;jj<ser_range;jj++){
						int rr=r+ii;
						int cc=c-cols1+48+jj;
						if(rr>w_size && rr<rows2-w_size && cc>w_size && cc<cols2-w_size){
							s12=0;s11=0;s22=0;s1=0;s2=0;
							int grey2=img_2.data[rr*cols2+cc];
							int sum1=0;int sum2=0;
							for(int k=-w_size/2;k<=w_size/2;k++){
								for(int m=-w_size/2;m<=w_size/2;m++){
									r1=r+k;r2=rr+k;
									c1=c+m;c2=cc+m;
									sum1 += img_1.data[r1*cols1+c1]-grey1;
									sum2 += img_2.data[r2*cols2+c2]-grey2;
									s12 += (img_1.data[r1*cols1+c1]-grey1)*(img_2.data[r2*cols2+c2]-grey2);
									s22 += (img_2.data[r2*cols2+c2]-grey2)*(img_2.data[r2*cols2+c2]-grey2);
									s11 += (img_1.data[r1*cols1+c1]-grey1)*(img_1.data[r1*cols1+c1]-grey1);
									s1  += (img_1.data[r1*cols1+c1]-grey1);
									s2  += (img_2.data[r2*cols2+c2]-grey2);
								}
							}
							s12 = s12/float(sum1)/float(sum2);
							s22 = s22/float(sum2)/float(sum2);
							s11 = s11/float(sum1)/float(sum1);
							s1 = s1/sum1;
							s2 = s2/sum2;
							CC = (s12-s1*s2/N)/sqrt((s11-s1*s1/N)*(s22-s2*s2/N));
							if(CC>maxCC){
								maxCC=CC;
								matchedx[count]=rr;
								matchedy[count]=cc;
								C_match[count]=maxCC;
							}
						}

					}
				}

			}

		}
	}

	FILE *fp4;
	if(CCD_id==0){
		fp4=fopen(outpointxt_1,"w");
	}
	else{
		fp4=fopen(outpointxt_1,"w");
	}
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(C_match[i]>=threshold){
			fprintf(fp4,"%d %d %d %d %d %d %f\n",CCD_id,KeyPoint_x1[i],KeyPoint_y1[i],CCD_id+1,matchedx[i],matchedy[i],C_match[i]);
		}
	}
	fclose(fp4);


	matchedx.swap(vector<int>());
	matchedy.swap(vector<int>());
	C_match.swap(vector<float>());

	KeyPoint_x1.swap(vector<int>());
	KeyPoint_y1.swap(vector<int>());

	return 0;
}


//格网点匹配
int grid_match(char* imagepathL, char* imagepathR,int w_size,float threshold,int* dr_,int* dc_,int ch){
	//读入影像
	Mat img_1 = imread(imagepathL, 0);
	Mat img_2 = imread(imagepathR, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	uchar* imgData1=img_1.data;
	uchar* imgData2=img_2.data;

	//生成格网点
	int SampleRateLine = 100;
	int SampleRateSample = 4;

	//格网点灰度匹配
	//int ch=1;
	if(ch==1){
		int dl,ds;
		int w_size=5;
		int N=w_size*w_size;
		float s12,s11,s22,s1,s2;
		int r1,r2,c1,c2;
		int dl_max=0;int ds_max=0;
		float maxCC=-1000;
		float CC;
		//for(dl=-rows1;dl<rows1;dl++){
		for(dl=-100;dl<100;dl++){
			for(ds=-cols1;ds<cols1;ds++){
				CC=0;
				int count=0;
				for(int i=0+w_size/2;i<rows1-w_size/2;i++){
					for(int j=0+w_size/2;j<cols1-w_size/2;j++){
						int ii=i-dl;int jj=j-ds;
						if(i%SampleRateLine==0 && j%SampleRateSample==0 && ii>=w_size/2 && ii<=rows2-w_size/2 && jj>=w_size/2 && jj<=cols2-w_size/2){
							s12=0;s11=0;s22=0;s1=0;s2=0;
							int sum1=0;int sum2=0;
							int grey1=imgData1[i*cols1+j];
							int grey2=imgData2[(ii)*cols2+jj];
							for(int k=-w_size/2;k<w_size/2;k++){
								for(int m=-w_size/2;m<w_size/2;m++){
									r1=i+k;r2=ii+k;
									c1=j+m;c2=jj+m;
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

		//拼接
		Mat img;
		int beginR = 0<=dl_max ? 0:dl_max;
		int endR = rows1>rows2+dl_max ? rows1:rows2+dl_max;
		int beginC = 0<=ds_max ? 0:ds_max;
		int endC = cols1>cols2+ds_max ? cols1:cols2+ds_max;
		img.create(endR-beginR,endC-beginC,CV_8UC1);
		int rows=img.rows;
		int cols=img.cols;
		uchar* pImg=img.data;

		if(beginR==0 && beginC==0){
			for(int i=0;i<rows1;i++){
				for(int j=0;j<cols1;j++){
					pImg[i*cols+j]=imgData1[i*cols1+j];
				}
			}
			for(int i=0;i<rows2;i++){
				for(int j=0;j<cols2;j++){
					pImg[(i+dl_max)*cols+j+ds_max]=imgData2[i*cols2+j];
				}
			}
		}
		else if(beginR<0 && beginC==0){
			for(int i=0;i<rows1;i++){
				for(int j=0;j<cols1;j++){
					pImg[(i-dl_max)*cols+j]=imgData1[i*cols1+j];
				}
			}
			for(int i=0;i<rows2;i++){
				for(int j=0;j<cols2;j++){
					pImg[i*cols+j+ds_max]=imgData2[i*cols2+j];
				}
			}
		}
		else if(beginR==0 && beginC<0){
			for(int i=0;i<rows1;i++){
				for(int j=0;j<cols1;j++){
					pImg[i*cols+j-ds_max]=imgData1[i*cols1+j];
				}
			}
			for(int i=0;i<rows2;i++){
				for(int j=0;j<cols2;j++){
					pImg[(i+dl_max)*cols+j]=imgData2[i*cols2+j];
				}
			}
		}
		else if(beginR<0 && beginC<0){
			for(int i=0;i<rows1;i++){
				for(int j=0;j<cols1;j++){
					pImg[(i-dl_max)*cols+j-ds_max]=imgData1[i*cols1+j];
				}
			}
			for(int i=0;i<rows2;i++){
				for(int j=0;j<cols2;j++){
					pImg[(i)*cols+j]=imgData2[i*cols2+j];
				}
			}
		}
		//imshow("mosaic",img);
		imwrite("E:\\Mars_VS\\Mars\\out\\mosaic.tif",img);
		//waitKey(0);
	}
	else if(ch==2){
		int dl,ds;
		int w_size=5;
		int N=w_size*w_size;
		float s12,s11,s22,s1,s2;
		int r1,r2,c1,c2;
		int dl_max=0;int ds_max=0;
		float maxCC=-1000;
		float CC;
		int mark;
		std::vector<int> matched;
		int count=0;
		for(int i=0+w_size/2;i<rows1-w_size/2;i++){
			for(int j=0+w_size/2;j<cols1-w_size/2;j++){
				maxCC=-1000;
				mark=0;

				int beginR = i-rows1/100>w_size/2 ? i-rows1/100 : w_size/2;
				int endR = i+rows1/100<rows2-w_size/2 ? i+rows1/100 : rows2-w_size/2;
				for(int ii=beginR;ii<endR;ii++){
					//for(int ii=0+w_size/2;ii<rows2-w_size/2;ii++){
					for(int jj=0+w_size/2;jj<cols2-w_size/2;jj++){
						if(i%SampleRateLine==0 && j%SampleRateSample==0 && ii%SampleRateLine==0 && jj%SampleRateSample==0){
							s12=0;s11=0;s22=0;s1=0;s2=0;
							int sum1=0;int sum2=0;
							int grey1=imgData1[i*cols1+j];
							int grey2=imgData2[(ii)*cols2+jj];
							for(int k=-w_size/2;k<w_size/2;k++){
								for(int m=-w_size/2;m<w_size/2;m++){
									r1=i+k;r2=ii+k;
									c1=j+m;c2=jj+m;
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
							CC = (s12-s1*s2/N)/sqrt(temp);

							if(CC>maxCC && CC>=threshold){
								if(mark==0){
									count=count+1;
									matched.push_back(i);matched.push_back(j);
									matched.push_back(ii);matched.push_back(jj);
									mark=1;
								}
								else if(mark==1){
									matched[4*(count-1)+2]=ii;
									matched[4*(count-1)+3]=jj;
								}
								maxCC=CC;
							}
						}
					}
				}
			}
		}
		int sf = 2;
		drawMatch1(img_1,img_2,matched,sf);
	}
}

//相邻CCD拼接
int Hijitreg(char* imagepathL, char* imagepathR, int OverlapSamples){
	//读入影像
	Mat img_1 = imread(imagepathL, 0);
	Mat img_2 = imread(imagepathR, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	uchar* imgData1=img_1.data;
	uchar* imgData2=img_2.data;

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

	//精确的offsetX+dx和offsetY+dy

	//拼接
	Mat img;
	int beginR = 0<=dl_max ? 0:dl_max;
	int endR = rows1>rows2+dl_max ? rows1:rows2+dl_max;
	img.create(endR-beginR,cols1+cols2-OverlapSamples+ds_max,CV_8UC1);
	int rows=img.rows;
	int cols=img.cols;
	uchar* pImg=img.data;

	if(beginR==0){
		for(int i=0;i<rows1;i++){
			for(int j=0;j<cols1;j++){
				pImg[i*cols+j]=imgData1[i*cols1+j];
			}
		}
		for(int i=0;i<rows2;i++){
			for(int j=0;j<cols2;j++){
				pImg[(i+dl_max)*cols+j+cols1-OverlapSamples+ds_max]=imgData2[i*cols2+j];
			}
		}
	}
	else{
		for(int i=0;i<rows1;i++){
			for(int j=0;j<cols1;j++){
				pImg[(i-dl_max)*cols+j]=imgData1[i*cols1+j];
			}
		}
		for(int i=0;i<rows2;i++){
			for(int j=0;j<cols2;j++){
				pImg[i*cols+j+cols1-OverlapSamples+ds_max]=imgData2[i*cols2+j];
			}
		}
	}
	imshow("mosaic",img);
	imwrite("E:\\Mars_VS\\Mars\\out\\mosaic.tif",img);
	waitKey(0);
}

int Hijitreg1(char* imagepathL, char* imagepathR, int OverlapSamples, char* outpath){
	//读入影像
	Mat img_1 = imread(imagepathL, 0);
	Mat img_2 = imread(imagepathR, 0);
	if((!img_1.data)||(!img_2.data))
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return -1; 
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;

	uchar* imgData1=img_1.data;
	uchar* imgData2=img_2.data;

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

	//精确的offsetX+dx和offsetY+dy

	//拼接
	Mat img;
	int beginR = 0<=dl_max ? 0:dl_max;
	int endR = rows1>rows2+dl_max ? rows1:rows2+dl_max;
	img.create(endR-beginR,cols1+cols2-OverlapSamples+ds_max,CV_8UC1);
	int rows=img.rows;
	int cols=img.cols;
	uchar* pImg=img.data;

	if(beginR==0){
		for(int i=0;i<rows1;i++){
			for(int j=0;j<cols1;j++){
				pImg[i*cols+j]=imgData1[i*cols1+j];
			}
		}
		for(int i=0;i<rows2;i++){
			for(int j=0;j<cols2;j++){
				pImg[(i+dl_max)*cols+j+cols1-OverlapSamples+ds_max]=imgData2[i*cols2+j];
			}
		}
	}
	else{
		for(int i=0;i<rows1;i++){
			for(int j=0;j<cols1;j++){
				pImg[(i-dl_max)*cols+j]=imgData1[i*cols1+j];
			}
		}
		for(int i=0;i<rows2;i++){
			for(int j=0;j<cols2;j++){
				pImg[i*cols+j+cols1-OverlapSamples+ds_max]=imgData2[i*cols2+j];
			}
		}
	}
	//imshow("mosaic",img);
	imwrite(outpath,img);
	//waitKey(0);
}

//根据拼接影像算出的dr和dc计算某左CCD影像对应右CCD序列影像的匹配关系及匹配范围，并输出
int getMatchRange(int CCD_ID,int CCD_rows,int CCD_cols,int OverlapCols,int sum_CCD,int dr,int dc,std::vector<int>& MatchRange){
	//匹配影像的行列数
	int mrows=CCD_rows;
	int mcols=CCD_cols*sum_CCD-(sum_CCD-1)*OverlapCols;
	int* mCCD_beginC = new int[sum_CCD];
	int* mCCD_endC = new int[sum_CCD];
	for(int i=0;i<sum_CCD;i++){
		mCCD_beginC[i] = CCD_cols*i-OverlapCols*i;
		mCCD_endC[i] = mCCD_beginC[i]+CCD_cols;
	}

	//影像拼接偏移量
	int* Offset_rows=new int[sum_CCD];
	memset(Offset_rows,0,sum_CCD*sizeof(int));
	int* Offset_cols=new int[sum_CCD];
	//int temp=-1*OverlapCols;
	memset(Offset_cols,-1,sum_CCD*sizeof(int));
	Offset_cols[0]=0;
	for(int i=0;i<sum_CCD;i++){
		Offset_cols[i]=OverlapCols*Offset_cols[i];
	}

	int beginR=0;
	int beginC=CCD_ID*CCD_cols;
	for(int i=0;i<=CCD_ID;i++){
		beginR += Offset_rows[i];
		beginC += Offset_cols[i];
	}

	int endR=beginR+CCD_rows;
	int endC=beginC+CCD_cols;

	int mbeginR,mbeginC,mendR,mendC;
	mbeginR= beginR+dr>mrows ? mrows : beginR+dr;
	mbeginR= mbeginR<0 ? 0 : mbeginR;
	mbeginC= beginC+dc>mcols ? mcols : beginC+dc;
	mbeginC= mbeginC<0 ? 0 : mbeginC;
	mendR= endR+dr>mrows ? mrows : endR+dr;
	mendR= mendR<0 ? 0 : mendR;
	mendC= endC+dc>mcols ? mcols : endC+dc;
	mendC= mendC<0 ? 0 : mendC;

	int begin_ID=-1;
	int end_ID=-1;
	for(int i=0;i<sum_CCD;i++){
		if(mbeginC>=mCCD_beginC[i] && mbeginC<mCCD_endC[i]){
			begin_ID=i;
			break;
		}
	}
	for(int i=0;i<sum_CCD;i++){
		if(mendC>=mCCD_beginC[i] && mendC<mCCD_endC[i]){
			end_ID=i;
			break;
		}
	}

	int out[4]={0,0,0,0}; //起始行，终止行，起始列，终止列
	for(int i=begin_ID;i<=end_ID;i++){
		if(i==begin_ID){
			out[0]=mbeginR;out[1]=mendR;
			out[2]=mbeginC-mCCD_beginC[begin_ID];out[3]=CCD_cols-1;
			if(out[2]-(out[3]-out[2])>0 && (out[3]-out[2])<10){
				out[2]=out[2]-(out[3]-out[2]);
			}
			//out[2]=0;out[3]=CCD_cols-1;
		}
		else if(i==end_ID){
			out[0]=mbeginR;out[1]=mendR;
			out[2]=0;out[3]=mendC-mCCD_beginC[end_ID];
			if(out[3]+(out[3]-out[2])<CCD_cols && (out[3]-out[2])<10){
				out[3]=out[3]+(out[3]-out[2]);
			}
			//out[2]=0;out[3]=CCD_cols-1;
		}
		else{
			out[0]=mbeginR;out[1]=mendR;
			out[2]=0;out[3]=CCD_cols-1;
		}
		MatchRange.push_back(i);
		MatchRange.push_back(out[0]);
		MatchRange.push_back(out[1]);
		MatchRange.push_back(out[2]);
		MatchRange.push_back(out[3]);
	}

	return 0;

}

void HijTest(){
	/*Mat img = imread("E:\\Mars_VS\\Mars\\out\\3.jpg", 0);
	if((!img.data))
	{ 
	std::cout<< " --(!) Error reading images " << std::endl; 
	return; 
	}
	int rows=img.rows;
	int cols=img.cols;

	int rows1=int(float(rows)/4*3);
	int cols1=int(float(cols)/4*3);
	Mat img1;
	img1.create(rows1,cols1,CV_8UC1);
	for(int i=0;i<img1.rows;i++){
	for(int j=0;j<img1.cols;j++){
	img1.data[i*img1.cols+j]=img.data[i*img.cols+j];
	}
	}

	Mat img2;
	img2.create(rows1,cols1,CV_8UC1);
	for(int i=0;i<img2.rows;i++){
	for(int j=0;j<img2.cols;j++){
	img2.data[i*img2.cols+j]=img.data[(i+4)*img.cols+j+cols/4];
	}
	}

	imwrite("E:\\Mars_VS\\Mars\\out\\4.jpg",img1);
	imwrite("E:\\Mars_VS\\Mars\\out\\5.jpg",img2);*/

	//int OverlapSample = float(cols)*1/2;
	int OverlapSample = 48;
	//Hijitreg("E:\\Mars_VS\\Mars\\out\\3.tif", "E:\\Mars_VS\\Mars\\out\\4.tif", OverlapSample);
	Hijitreg("G:\\Mars\\data\\1650\\mosaic\\PSP_001777_1650_RED2.tif", "G:\\Mars\\data\\1650\\mosaic\\PSP_001777_1650_RED3.tif", OverlapSample);

}

void LocalImg(char* imgpath,int beginR,int endR,int beginC,int endC,char* outpath){
	Mat image = imread(imgpath, 0);
	if( !image.data)
	{ 
		std::cout<< " --(!) Error reading images " << std::endl; 
		return; 
	}

	Mat out;
	out.create(endR-beginR,endC-beginC,CV_8UC1);

	for(int i=beginR;i<endR;i++){
		for(int j=beginC;j<endC;j++){
			out.data[(i-beginR)*out.cols+j-beginC]=image.data[i*image.cols+j];
		}
	}

	imwrite(outpath,out);
}

void MR_test(){
	int dr,dc;
	int CCD_ID=3;
	CC_match("G:\\Mars\\data\\1650\\downsample\\PSP_001777_1650_RED.tif","G:\\Mars\\data\\1655\\downsample\\PSP_001513_1655_RED.tif",15,0.95,&dr,&dc,2);
	//CC_match("G:\\Mars\\data\\1650\\downsample\\PSP_001777_1650_RED.tif","G:\\Mars\\data\\1650\\downsample\\PSP_001777_1650_RED.tif",15,0.95,&dr,&dc);
	std::vector<int> MatchRange;
	getMatchRange(CCD_ID,5000,256,6,10,dr,dc,MatchRange);
	std::vector<int> MatchRange1;
	getMatchRange(MatchRange[0],5000,256,6,10,-dr,-dc,MatchRange1);

	char* imgpath1 = new char[80];
	char* imgpath11 = new char[80];
	sprintf( imgpath1, "%s%d%s", "G:\\Mars\\data\\1650\\mosaic\\PSP_001777_1650_RED", CCD_ID, ".tif" );
	sprintf( imgpath11, "%s%d%s", "G:\\Mars\\data\\1650\\temp\\PSP_001777_1650_RED", CCD_ID, "_local.tif" );
	int Mark=-1;
	for(int i=0;i<MatchRange1.size()/5;i++){
		if(MatchRange1[5*i]==CCD_ID){
			Mark=5*i;
		}
	}

	char* imgpath2 = new char[80];
	char* imgpath22 = new char[80];
	sprintf( imgpath2, "%s%d%s", "G:\\Mars\\data\\1655\\mosaic\\PSP_001513_1655_RED", MatchRange[0], ".tif" );
	//sprintf( imgpath2, "%s%d%s", "G:\\Mars\\data\\1650\\mosaic\\PSP_001777_1650_RED", MatchRange[0], ".tif" );
	sprintf( imgpath22, "%s%d%s", "G:\\Mars\\data\\1655\\temp\\PSP_001513_1655_RED", MatchRange[0], "_local.tif" );

	int sf=8;
	LocalImg(imgpath2,sf*MatchRange[1],sf*MatchRange[2],sf*MatchRange[3],sf*MatchRange[4],imgpath22);
	LocalImg(imgpath1,sf*MatchRange1[Mark+1],sf*MatchRange1[Mark+2],sf*MatchRange1[Mark+3],sf*MatchRange1[Mark+4],imgpath11);
	//grid_match(imgpath11,imgpath22,11,0.9,&dr,&dc,2);
	CC_match(imgpath11,imgpath22,15,0.95,&dr,&dc,5);
}

void eigentest(){
	Matrix3d a;
	a << 1, 2, 3, 
		4,5,6,
		7,8,9;
	MatrixXd b(3, 3);
	b << 1,2, 3,
		1, 4,1,
		2,4,4;
	cout << "a + b =\n" << a + b << endl;
	cout << "a - b =\n" << a - b << endl;
	cout << "Doing a += b;" << endl;
	a += b;
	cout << "Now a =\n" << a << endl;
	cout << "a^T=  " << a.transpose() << endl;
	cout << "a*b= " << a*b << endl;
	Vector3d v(1, 2, 3);
	Vector3d w(1, 0, 0);
	cout << "-v + w - v =\n" << -v + w - v << endl;
	cout << v << endl;
	cout << v.transpose() << endl;
	system("pause");
}

//外方位元素的三次多项式拟合EO(t)=EO(t0)+c1(t-t0)+c2(t-t0)^2+c3(t-t0)^3
//EO(l)=EO(l0)+c1(l-l0)+c2(l-l0)^2+c3(l-l0)^3(后续写成关于行的拟合多项式，或者影像加个配置文件)
void Polynomial3_EO(char* EO_txt,double* Poly_C, char* polyCC_txt){
	FILE *fp=fopen(EO_txt,"r");
	double beginT,LR;
	fscanf(fp,"%lf %lf\n",&beginT,&LR);
	std::vector<double> EO;
	double et,Xs,Ys,Zs,phi,w,ka;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",&et,&Xs,&Ys,&Zs,&phi,&w,&ka);
		EO.push_back(et);
		EO.push_back(Xs);EO.push_back(Ys);EO.push_back(Zs);
		EO.push_back(phi);EO.push_back(w);EO.push_back(ka);
	}
	fclose(fp);

	double t,t0;
	t0=EO[0];
	int number=EO.size()/7;
	MatrixXf x;
	FILE *fp1=fopen(polyCC_txt,"w");
	//fprintf(fp1,"%lf %lf\n",beginT,LR);
	for(int j=1;j<7;j++){
		MatrixXf A=MatrixXf::Zero(number,4);
		MatrixXf L=MatrixXf::Zero(number,1);
		for(int i=0;i<number;i++){
			VectorXf temp1(4);
			t=EO[i*7];
			temp1 << 1.0,(t-t0),pow((t-t0),2),pow((t-t0),3);
			A.row(i)=temp1;
			L(i)=EO[i*7+j];
		}
		x = (A.transpose()*A).inverse()*A.transpose()*L;
		Poly_C[5*(j-1)+0]=t0;
		Poly_C[5*(j-1)+1]=x(0);
		Poly_C[5*(j-1)+2]=x(1);
		Poly_C[5*(j-1)+3]=x(2);
		Poly_C[5*(j-1)+4]=x(3);
		fprintf(fp1,"%.12lf %.12lf %.12lf %.12lf %.12lf\n",t0,x(0),x(1),x(2),x(3));
	}
	fclose(fp1);
}

//根据拟合公式求出对应行的外方位元素
void Get_PolyEO(double et,double* Poly_C,float* EO){
	for(int i=0;i<6;i++){
		double t=et;
		double t0=Poly_C[5*i];
		EO[i]=Poly_C[5*i+1]+Poly_C[5*i+2]*(t-t0)+Poly_C[5*i+3]*pow((t-t0),2)+Poly_C[5*i+4]*pow((t-t0),3);
	}
}

void Get_PolyEO1(double et,int Line,double* Poly_C,float* EO){
	for(int i=0;i<3;i++){
		double t=et;
		double t0=Poly_C[5*i];
		EO[i]=Poly_C[5*i+1]+Poly_C[5*i+2]*(t-t0)+Poly_C[5*i+3]*pow((t-t0),2)+Poly_C[5*i+4]*pow((t-t0),3);
	}
	for(int i=3;i<6;i++){
		double t=Line;
		double t0=0;
		EO[i]=Poly_C[5*i+1]+Poly_C[5*i+2]*(t-t0)+Poly_C[5*i+3]*pow((t-t0),2)+Poly_C[5*i+4]*pow((t-t0),3);
	}
	double R[3][3];
	eul2m_c(EO[3],EO[4],EO[5],1,2,3,R);
	double a1,a2,a3;
	m2eul_c( R, 2, 1, 3, &a3, &a2, &a1);
	EO[3]=a3;
	EO[4]=a2;
	EO[5]=a1;
}

//内方位纠正;IO:x0,dx/ds,dx/dl,y0,dy/ds,dy/dl,k0,k1,k2,f
void IO_correct(int sample,int BIN,int TDI,float* IO,float* xp,float* yp){
	float s = (float(sample)-0.5)*float(BIN)-1024;
	float l = float(TDI)/2-64-(float(BIN)/2-0.5);

	float x = IO[0]+IO[1]*s+IO[2]*l;
	float y = IO[3]+IO[4]*s+IO[5]*l;

	float r = sqrt(x*x+y*y);
	float dr_r = IO[6]+IO[7]*r*r+IO[8]*r*r*r*r;

	*xp = x-dr_r*x;
	*yp = y-dr_r*y;
}

//欧拉角计算旋转矩阵
void Eul2R(float phi,float w,float k,float* R){
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

//根据DEM和单张影像获取地面点坐标
//EO：Xs,Ys,Zs,phi,w,k
void Get_groundtruth(char* DEMdoc, int sample, int line,int BIN, int TDI, float meanZ, float* EO, float* IO, float* GC){
	double rMars=3396190;
	//读取坐标系信息
	GDALAllRegister();
	GDALDataset *poDataset = (GDALDataset*) GDALOpen( DEMdoc,GA_ReadOnly );

	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",DEMdoc);
		return;
	}
	const char *projRef =poDataset->GetProjectionRef();

	//内定向
	float xp,yp;
	IO_correct(sample,BIN,TDI,IO,&xp,&yp);
	float * R = new float[9];
	//Eul2R(EO[3]*3.14159/180,EO[4]*3.14159/180,EO[5]*3.14159/180,R);
	Eul2R(EO[3],EO[4],EO[5],R);
	//Eul2R(0,0,0,R);

	float f = IO[9];
	float * xi = new float[3];
	xi[0] = xp;
	xi[1] = yp;
	xi[2] = f;

	//R转换到eigen格式，通过eigen矩阵运算计算X，Y
	MatrixXf R_ = (Map<MatrixXf>(R,3,3)).transpose();
	MatrixXf xi_ = Map<MatrixXf>(xi,3,1);
	VectorXf Rx = R_*xi_;

	float Xs = EO[0];
	float Ys = EO[1];
	float Zs = EO[2];

	double rate=rMars/sqrt(Xs*Xs+Ys*Ys+Zs*Zs);
	double meanX=Xs*rate;
	double meanY=Ys*rate;
	meanZ=Zs*rate;

	float X,Y,Z;
	float dX,dY,dZ;
	dX=1000000;dY=1000000;dZ=1000000;
	Z = meanZ;

    
	//投影到球面
	double A=Rx(0)*Rx(0)+Rx(1)*Rx(1)+Rx(2)*Rx(2);
	double B=2*(Rx(0)*Xs+Rx(1)*Ys+Rx(2)*Zs);
	double C=Xs*Xs+Ys*Ys+Zs*Zs-rMars*rMars;
	double nlamda1=(-1*B+sqrt(B*B-4*A*C))/2/A;
	double nlamda2=(-1*B-sqrt(B*B-4*A*C))/2/A;
	double nlamda=1;
	
	if(nlamda1*nlamda1>nlamda2*nlamda2){
		nlamda=nlamda2;
	}
	else{
		nlamda=nlamda1;
	}

	X=nlamda*Rx(0)+Xs;
	Y=nlamda*Rx(1)+Ys;
	Z=nlamda*Rx(2)+Zs;
	
	/*
	float lamda = (Z-Zs)/Rx(2);
	X = lamda*Rx(0)+Xs;
	Y = lamda*Rx(1)+Ys;

	float dX0,dY0,dZ0;
	double N,E,H;
	float temp;
	int count = 0;
	while(count<100){
		dZ0=dZ;
		temp=Z;
		
		
		My_rec2NEH(projRef,double(X),double(Y),double(Z),&N,&E,&H);
		LoadDEM(DEMdoc, float(E), float(N), &Z);
		H=double(Z);
		double XX,YY,ZZ;
		My_NEH2rec(projRef,N,E,H,&XX,&YY,&ZZ);
		X=float(XX);Y=float(YY);Z=float(ZZ);

		dZ = abs(temp-Z);
		lamda = (Z-Zs)/Rx(2);

		dX0 = dX;dY0 = dY;
		temp = X;
		X = lamda*Rx(0)+Xs;
		dX = abs(temp-X);
		temp = Y;
		Y = lamda*Rx(1)+Ys;
		dY = abs(temp-Y);
		if(dX>=dX0 && dY>=dY0 && dZ>=dZ0){
			break;
		}
		count++;
		if(count==100){
			printf("%s\n","000000!");
		}
	}*/
	GC[0]=X;
	GC[1]=Y;
	GC[2]=Z;
}

void Get_groundtruth1(float xp ,float yp, float* EO, float f, float rMars, float* GC){
	//double rMars=3396190;
	//读取坐标系信息
	
	float * R = new float[9];
	//Eul2R(EO[3]*3.14159/180,EO[4]*3.14159/180,EO[5]*3.14159/180,R);
	Eul2R(EO[3],EO[4],EO[5],R);
	//Eul2R(0,0,0,R);

	float * xi = new float[3];
	xi[0] = xp;
	xi[1] = yp;
	xi[2] = f;

	//R转换到eigen格式，通过eigen矩阵运算计算X，Y
	MatrixXf R_ = (Map<MatrixXf>(R,3,3)).transpose();
	MatrixXf xi_ = Map<MatrixXf>(xi,3,1);
	VectorXf Rx = R_*xi_;

	float Xs = EO[0];
	float Ys = EO[1];
	float Zs = EO[2];

	float X,Y,Z;

	//投影到球面
	double A=Rx(0)*Rx(0)+Rx(1)*Rx(1)+Rx(2)*Rx(2);
	double B=2*(Rx(0)*Xs+Rx(1)*Ys+Rx(2)*Zs);
	double C=Xs*Xs+Ys*Ys+Zs*Zs-rMars*rMars;
	double nlamda1=(-1*B+sqrt(B*B-4*A*C))/2/A;
	double nlamda2=(-1*B-sqrt(B*B-4*A*C))/2/A;
	double nlamda=1;
	
	if(nlamda1*nlamda1>nlamda2*nlamda2){
		nlamda=nlamda2;
	}
	else{
		nlamda=nlamda1;
	}

	X=nlamda*Rx(0)+Xs;
	Y=nlamda*Rx(1)+Ys;
	Z=nlamda*Rx(2)+Zs;
	
	GC[0]=X;
	GC[1]=Y;
	GC[2]=Z;
}

//根据地面点及外方位元素获取影像坐标
void Get_imageCoor(float* EO, float* IO, float* GC, float* IC){
	float * R = new float[9];
	Eul2R(EO[3],EO[4],EO[5],R);
	MatrixXf R_inv = (Map<MatrixXf>(R,3,3)).inverse();

	float* XYZ = new float[3];
	XYZ[0] = GC[0]-EO[0];
	XYZ[1] = GC[1]-EO[1];
	XYZ[2] = GC[2]-EO[2];
	MatrixXf XYZ_ = Map<MatrixXf>(XYZ,3,1);

	VectorXf Rx = R_inv*XYZ_;

	float x0 = IO[0];
	float y0 = IO[3];
	float f = IO[9];
	float lamda = -f/Rx(2);

	IC[0] = lamda*Rx(0)+x0;
	IC[1] = lamda*Rx(1)+y0;
}

//合起来，根据影像坐标获取地面坐标，再回去右影像找到大概区域进行约束匹配。
void G_M(char* DEMdoc,int sample, int line, int BIN, int TDI, float meanZ, float* EO, float* IO, int RLines, double DLINE, double RBIN, double RTDI, double et0, double* Poly_C, float* RIO, float* IC){
	float* GC = new float[3];
	Get_groundtruth(DEMdoc, sample, line, BIN, TDI, meanZ, EO, IO, GC);

	double LR = ( 74.0 + DLINE/16.0 )/1000000;
	double et1 = et0 + LR*(BIN-TDI)/2;
	double et;
	//float* IC = new float[2];
	float* REO = new float[6];
	for(int i=0;i<RLines;i++){
		et = et1 + i*LR*BIN;
		Get_PolyEO(et,Poly_C,REO);
		Get_imageCoor(REO, RIO, GC, IC);
		if(abs(IC[2]-0)<2){
			break;
		}
	}
}

void SetObserveTxT(char* matchfile,char* EOfile,char* IOtxt,char* xulie_ID1,char* xulie_ID2,char* observetxt,int* base){
	int rows1=40000;
	int rows2=80000;
	//读取内方位元素
	int CCD_num=10;
	int TDI=128;
	int BIN=1;
	int count=0;
	FILE* fp_io = fopen(IOtxt, "r");

	float** IO = new float*[CCD_num];
	for(int i=0;i<CCD_num;i++){
		IO[i]=new float[10];
	}
	for(int j=0;j<CCD_num;j++){
		int temp;
		float temp1;
		for(int k=0;k<11;k++){
			if(k==0){
				fscanf(fp_io,"%d ",&temp);
			}
			else if(k==8||k==9){
				fscanf(fp_io,"%e ",&IO[j][k-1]);
			}
			else{
				fscanf(fp_io,"%f ",&IO[j][k-1]);
			}
		}
		fscanf(fp_io,"\n");
	}
	fclose(fp_io);

	char* matchtxt = new char[80];
	char* matchtxt1 = new char[80];
	char* EOtxt = new char[80];
	//char* EOtxt1 = new char[80];
	FILE *fp1=fopen(observetxt,"w");
	float xp,yp;
	double et;
	float* EO = new float[6];
	float* GC = new float[3];
	double* Poly_C = new double[30];

	//航带间
	for(int i=0;i<CCD_num;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s", matchfile, "\\", xulie_ID1, "\\downsample\\0\\",xulie_ID1, "_RED", i, "_match.txt" );

		double beginET,LR;

		FILE *fp=fopen(matchtxt,"r");
		int bj,row,col,imgID,mrow,mcol;
		float mscore;

		while(!feof(fp)){
			fscanf(fp,"%d ",&bj);
			if(bj==1){
				fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
				IO_correct(col, BIN, TDI, IO[i], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID1, "\\",xulie_ID1, "_RED", i, "_0.txt" );
				//sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID1, "\\",xulie_ID1, "_RED", 5, "_0.txt" );
				FILE* fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				//row=rows1-row;
				//col=2048-col;
				et=beginET+row*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID1, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				double a1,a2,a3,a4,a5;
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				//fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp/IO[i][9], yp/IO[i][9], et, 0, GC[0], GC[1], GC[2]);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

				//mrow=rows2-mrow;
				//mcol=2048-mcol;
				IO_correct(mcol, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", imgID, "_0.txt" );
				//sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", 5, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID2, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				//fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp/IO[imgID][9], yp/IO[imgID][9], et, 1, GC[0], GC[1], GC[2]);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
				count++;
			}
			else{
				fscanf(fp,"%d %d\n",&row,&col);
			}
		}
		fclose(fp);
	}
	

	//CCD间:左影像
	for(int i=0;i<CCD_num-1;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s%d%s", matchfile, "\\", xulie_ID1, "\\downsample\\0\\",xulie_ID1, "_RED", i,"_", i+1, "_intra.txt" );
		sprintf( matchtxt1, "%s%s%s%s%s%s%d%s%d%s", matchfile, "\\", xulie_ID1, "\\downsample\\0\\",xulie_ID1, "_RED", i,"_", i+1, "_intra__.txt" );

		double beginET,LR;
		int imgID,row,col,mimgID,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt1,"r");
		while(!feof(fp) && !feof(fp_)){
			fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
			double a1,a2,a3,a4,a5;
			FILE* fp_eo;
			IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
			sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID1, "\\",xulie_ID1, "_RED", imgID, "_0.txt" );
			fp_eo=fopen(EOtxt,"r");
			fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
			fclose(fp_eo);
			et=beginET+row*LR;

			sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID1, "\\polyCC.txt" );
			fp_eo=fopen(EOtxt,"r");
			for(int ii=0;ii<6;ii++){
				fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
				Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
				Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
			}
			fclose(fp_eo);
			Get_PolyEO(et,Poly_C,EO);
			Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
			//fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);


			/*IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
			sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID1, "\\",xulie_ID1, "_RED", mimgID, "_0.txt" );
			fp_eo=fopen(EOtxt,"r");
			fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
			fclose(fp_eo);
			et=beginET+mrow*LR;

			sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID1, "\\polyCC.txt" );
			fp_eo=fopen(EOtxt,"r");
			for(int ii=0;ii<6;ii++){
				fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
				Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
				Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
			}
			fclose(fp_eo);
			Get_PolyEO(et,Poly_C,EO);
			Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
			fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);*/

			//带间
			int bj;
			fscanf(fp_,"%d ",&bj);
			if(bj==1){
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);
				fscanf(fp_,"%d %d %d %d %d %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(mcol, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID2, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
				count++;
			}
			else{
				fscanf(fp_,"%d %d\n",&row,&col);
			}
			//count++;
		}
		fclose(fp);
		fclose(fp_);
	}
	/**/

	//CCD间：右影像
	/*for(int i=0;i<CCD_num-1;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s%d%s", matchfile, "\\", xulie_ID2, "\\downsample\\0\\",xulie_ID2, "_RED", i,"_", i+1, "_intra.txt" );

		double beginET,LR;
		int imgID,row,col,mimgID,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		while(!feof(fp)){
			fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
			if(row>20000 && row<60000){
				IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", imgID, "_0.txt" );
				FILE* fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+row*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID2, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				double a1,a2,a3,a4,a5;
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);


				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID2, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
				count++;
			}
		}
		fclose(fp);
	}*/

	fclose(fp1);
	*base = count;
}

void SetObserveTxT1(char* matchfile,char* EOfile,char* IOtxt,char* xulie_ID1,char* xulie_ID2,char* observetxt,int* base){
	int rows1=40000;
	int rows2=80000;
	//读取内方位元素
	int CCD_num=10;
	int TDI=128;
	int BIN=1;
	int count=0;
	FILE* fp_io = fopen(IOtxt, "r");

	float** IO = new float*[CCD_num];
	for(int i=0;i<CCD_num;i++){
		IO[i]=new float[10];
	}
	for(int j=0;j<CCD_num;j++){
		int temp;
		float temp1;
		for(int k=0;k<11;k++){
			if(k==0){
				fscanf(fp_io,"%d ",&temp);
			}
			else if(k==8||k==9){
				fscanf(fp_io,"%e ",&IO[j][k-1]);
			}
			else{
				fscanf(fp_io,"%f ",&IO[j][k-1]);
			}
		}
		fscanf(fp_io,"\n");
	}
	fclose(fp_io);

	char* matchtxt = new char[80];
	char* matchtxt1 = new char[80];
	char* EOtxt = new char[80];
	//char* EOtxt1 = new char[80];
	FILE *fp1=fopen(observetxt,"w");
	float xp,yp;
	double et;
	float* EO = new float[6];
	float* GC = new float[3];
	double* Poly_C = new double[30];

	//航带间
	for(int i=0;i<CCD_num;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s", matchfile, "\\", xulie_ID1, "\\downsample\\0\\",xulie_ID1, "_RED", i, "_match.txt" );

		double beginET,LR;

		FILE *fp=fopen(matchtxt,"r");
		int bj,row,col,imgID,mrow,mcol;
		float mscore;

		while(!feof(fp)){
			fscanf(fp,"%d ",&bj);
			if(bj==1){
				fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore);
				IO_correct(col, BIN, TDI, IO[i], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID1, "\\",xulie_ID1, "_RED", i, "_0.txt" );
				//sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID1, "\\",xulie_ID1, "_RED", 5, "_0.txt" );
				FILE* fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				//row=rows1-row;
				//col=2048-col;
				et=beginET+row*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID1, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				double a1,a2,a3,a4,a5;
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				//fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp/IO[i][9], yp/IO[i][9], et, 0, GC[0], GC[1], GC[2]);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

				//mrow=rows2-mrow;
				//mcol=2048-mcol;
				IO_correct(mcol, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", imgID, "_0.txt" );
				//sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", 5, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID2, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				//fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp/IO[imgID][9], yp/IO[imgID][9], et, 1, GC[0], GC[1], GC[2]);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
				count++;
			}
			else{
				fscanf(fp,"%d %d\n",&row,&col);
			}
		}
		fclose(fp);
	}
	

	//CCD间:左影像
	for(int i=0;i<CCD_num-1;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s%d%s", matchfile, "\\", xulie_ID1, "\\downsample\\0\\",xulie_ID1, "_RED", i,"_", i+1, "_intra.txt" );
		sprintf( matchtxt1, "%s%s%s%s%s%s%d%s%d%s", matchfile, "\\", xulie_ID1, "\\downsample\\0\\",xulie_ID1, "_RED", i,"_", i+1, "_intra__.txt" );

		double beginET,LR;
		int imgID,row,col,mimgID,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt1,"r");
		while(!feof(fp) && !feof(fp_)){
			fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
			//if(abs(mrow-row)==6){
			IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
			sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID1, "\\",xulie_ID1, "_RED", imgID, "_0.txt" );
			FILE* fp_eo=fopen(EOtxt,"r");
			fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
			fclose(fp_eo);
			et=beginET+row*LR;

			sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID1, "\\polyCC.txt" );
			fp_eo=fopen(EOtxt,"r");
			double a1,a2,a3,a4,a5;
			for(int ii=0;ii<6;ii++){
				fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
				Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
				Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
			}
			fclose(fp_eo);
			Get_PolyEO(et,Poly_C,EO);
			Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
			fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);


			IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
			sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID1, "\\",xulie_ID1, "_RED", mimgID, "_0.txt" );
			fp_eo=fopen(EOtxt,"r");
			fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
			fclose(fp_eo);
			et=beginET+mrow*LR;

			sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID1, "\\polyCC.txt" );
			fp_eo=fopen(EOtxt,"r");
			for(int ii=0;ii<6;ii++){
				fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
				Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
				Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
			}
			fclose(fp_eo);
			Get_PolyEO(et,Poly_C,EO);
			Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
			fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

			//带间
			int bj;
			fscanf(fp_,"%d ",&bj);
			if(bj==1){
				fscanf(fp_,"%d %d %d %d %d %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(mcol, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID2, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
			}
			else{
				fscanf(fp,"%d %d\n",&row,&col);
			}
			count++;
		//}
		}
		fclose(fp);
		fclose(fp_);
	}

	//CCD间：右影像
	for(int i=0;i<CCD_num-1;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s%d%s", matchfile, "\\", xulie_ID2, "\\downsample\\0\\",xulie_ID2, "_RED", i,"_", i+1, "_intra.txt" );

		double beginET,LR;
		int imgID,row,col,mimgID,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		while(!feof(fp)){
			fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
			if(row>20000 && row<60000){
				IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", imgID, "_0.txt" );
				FILE* fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+row*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID2, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				double a1,a2,a3,a4,a5;
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);


				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "\\", xulie_ID2, "\\",xulie_ID2, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID2, "\\polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
				count++;
			}
		}
		fclose(fp);
	}

	fclose(fp1);
	*base = count;
}



void prepare_for_BA(char* filepath,char* xulie_ID1,char* xulie_ID2){
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

	char* matchfile = filepath;
	char* EOfile = "..\\data\\EO";
	char* IOtxt = "..\\data\\IO\\IO.txt";
	char* observetxt = new char[80];
	sprintf( observetxt, "%s%s%s%s%s", "..\\data\\observedata\\", xulie_ID1,"_", xulie_ID2, ".txt");
	int observe_count=0;
	SetObserveTxT(matchfile,EOfile,IOtxt,xulie_ID1,xulie_ID2,observetxt,&observe_count);
}


//XYZ转为NEH
void GetLocalNEH(char* ProjFile,char* XYZfile,char* NEHfile){
	//读取坐标系信息
	GDALAllRegister();
	GDALDataset *poDataset = (GDALDataset*) GDALOpen( ProjFile,GA_ReadOnly );

	if( poDataset == NULL )
	{
		printf( "File1: %s不能打开！\n",ProjFile);
		return;
	}
	const char *projRef =poDataset->GetProjectionRef();

	int ground_num;
	double X,Y,Z,res;
	double N,E,H;
	FILE *fp_gt=fopen(XYZfile,"r");
	FILE *fp_neh=fopen(NEHfile,"w");
	fscanf(fp_gt,"%d\n",&ground_num);
	fprintf(fp_neh,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		fscanf(fp_gt,"%lf %lf %lf %lf\n",&X,&Y,&Z,&res);
		My_rec2NEH(projRef,double(X),double(Y),double(Z),&N,&E,&H);
		fprintf(fp_neh,"%lf %lf %lf %lf\n",N,E,H,res);
	}
	fclose(fp_gt);
	fclose(fp_neh);
}