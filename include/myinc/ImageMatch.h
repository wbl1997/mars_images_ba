#ifndef _IMAGEMATCH_H_
#define _IMAGEMATCH_H_

#include "EO.h"

#include <math.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "mygdal.h"


using namespace cv;
using namespace std;
using namespace Eigen;

// 分块局部仿射场（mosaic 坐标）：缓解大起伏下全局仿射失效
struct LocalAffineField {
	int mosaic_rows = 0;
	int mosaic_cols = 0;
	int tile_h = 1;
	int tile_w = 1;
	int n_tr = 0;
	int n_tc = 0;
	float global_fs[6];
	std::vector<float> tile_fs;   // n_tr * n_tc * 6
	std::vector<char> tile_valid;

	LocalAffineField();
	void set_global(const float* fs);
	void clear();
	void upsample(float scale);  // 金字塔升采样：线性项不变，平移 * scale
	bool predict(float r, float c, float* pred_r, float* pred_c) const;
	bool copy_affine_at(float r, float c, float* fs_out) const;
};

class ImageMatch{
public:
	static void SetScoreBOptions(bool use_robust_scoreb, bool use_affine_patch_score);
	//��������ȡ
	int Forstner(uchar* pImg,int rows,int cols,int base, std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y);
	void GaussianKernel(float sigma,int dim, float* kernel);
	void Gaussian_Juanji(float* InArray, float* OutArray, int rows, int cols, int dim, float sigma);
	int HarrisCorner(uchar* pImg,int rows,int cols,int Juanji_dim,float Juanji_sigma,int Localmax_dim,float threshold,std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y);
	int Feature_Detection(char* imagepath,int ch,int thresh,std::vector<int>& KeyPoint_x,std::vector<int>& KeyPoint_y);

	//剔除粗差（轻量实现见 RANSAC_fs2_new）
	void RANSAC_fs2(std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,float sigma,int iter,int* match, float* fs_c);
	void RANSAC_fs2_new(const std::vector<int>& KeyPoint_x1,const std::vector<int>& KeyPoint_y1,
		const std::vector<int>& KeyPoint_x2,const std::vector<int>& KeyPoint_y2,
		float sigma,int iter,int* match, float* fs_c);
	void RANSAC_plane(std::vector<int> matchpoint,float sigma,int iter, float* plane_c);
	//����ƥ����
	void drawMatch(Mat imgL0,Mat imgR0,std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int sf);
	void drawMatch1(Mat imgL0,Mat imgR0,std::vector<int> matched,int sf);
	void drawMatch2(char* imagepath1,char* imagepath2, std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int sf);
	void drawMatch3(char* imagepath1,char* imagepath2,char* outpath,std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int sf);
	// 与 script/draw_fenfu_match0_inliers.py --no-draw-lines 一致：黄=未匹配、红=错误匹配、绿=正确匹配
	void drawMatch3(char* imagepath1,char* imagepath2,char* outpath,
		std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,
		std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int sf,
		const std::vector<int>& unmatched_x1,const std::vector<int>& unmatched_y1,
		const std::vector<int>& unmatched_x2,const std::vector<int>& unmatched_y2,
		bool draw_lines);
	// 从 *.txt / *_match.txt 绘制（层末 keep 后，与 Python collect_points 一致）
	void drawFenfuMatchFromFiles(char* imagepath1,char* imagepath2,char* outpath,
		char* filepath,char* xulie_ID1,char* xulie_ID2,int layer,
		int ccd_begin,int ccd_end,int CCD_num,
		const int* mosaic_c1,const int* mosaic_c2,bool draw_lines);
	void drawMatch4(char* imagepath1,char* imagepath2,char* outpath,std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<int> KeyPoint_x2,std::vector<int> KeyPoint_y2,int *Match,int count1,int sf);
	void draw_fea(char* imagepath1, std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,int sf,const char* outpath="../out/fea_.tif");
	void draw_res(char* imagepath1, std::vector<int> KeyPoint_x1,std::vector<int> KeyPoint_y1,std::vector<double> resx,std::vector<double> resy,int sf,const char* outpath="../out/fea_.tif");

	//������ƥ��
	int CC_match2(char* imagepath1,char* imagepath2,int w_size,float threshold,int ch,float* fs_c,int RCCD_id, char* featurepointxt_1,char* featurepointxt_2,char* outpointxt_1);
	int limit_match1(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int ch,float* fs_c,int RCCD_id, char* featurepointxt_1,char* featurepointxt_2,char* outpointxt_1);
	// 用 LocalAffineField（mosaic 坐标）引导搜索；field==nullptr 时退化为 limit_match1
	int limit_match1_field(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int ch,float* fs_c,int RCCD_id,
		char* featurepointxt_1,char* featurepointxt_2,char* outpointxt_1,
		const LocalAffineField* field,int off_r1,int off_c1,int off_r2,int off_c2);
	void build_local_affine_field(const std::vector<int>& x1,const std::vector<int>& y1,
		const std::vector<int>& x2,const std::vector<int>& y2,
		int mosaic_rows,int mosaic_cols,int n_tiles_r,int n_tiles_c,
		float sigma,int ransac_iters,int min_pts,float* global_fs,LocalAffineField& field);
	void mark_local_inliers(const std::vector<int>& x1,const std::vector<int>& y1,
		const std::vector<int>& x2,const std::vector<int>& y2,
		const LocalAffineField& field,float sigma,int* match);
	// 以种子点局部视差引导，Gray+Grad 密格网成网（参考 intra_CCD_match_B）
	int densify_match_from_seeds(char* imagepath1,char* imagepath2,
		int w_size,int ser_range,float threshold,int batch_r,int batch_c,int knn,
		char* seed_match_txt,int target_right_ccd,char* out_append_txt);
	int limit_fea(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* matchpointfile, char* featurepointxt_1, char* featurepointxt_2, char* outpointxt_1);
	//��������Լ���ĸ�����ƥ��
	int limit_grid(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1,char* outpointxt_1);
	int limit_grid1(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1,char* outpointxt_1);
	int limit_grid_global(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1,char* outpointxt_1,bool use_grid_controls=true);
	int limit_grid4(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* matchpointfile, char* featurepointxt_1, char* outpointxt_1, float lambda2=0.1f, int affine_max_dev=512, bool affine_pred_as_match=false);
	int limit_grid5(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* featurepointxt_1, char* outpointxt_1, float lambda2=0.1f, int affine_max_dev=512);
	//���ڸ�����Լ�����ܼ�ƥ��
	int limit_dense2(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,float* fs_c,int RCCD_id, char* matchpointfile, char* featurepointxt_1, char* outpointxt_1,int offY,int bYsize,float* MI_table);
	//CCD�����ӵ�ƥ��
	int intra_CCD_match(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int batch_size,int CCD_id,char* outpointxt_1);
	int intra_CCD_match(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int batch_size_r,int batch_size_c,int CCD_id,char* outpointxt_1);
	int intra_CCD_match_A(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int batch_size_r,int batch_size_c,int CCD_id,char* outpointxt_1);
	int intra_CCD_match_B(char* imagepath1,char* imagepath2,int w_size,int ser_range,float threshold,int batch_size_r,int batch_size_c,int CCD_id,char* outpointxt_1);

	//����CCDӰ��ƴ��
	int Hijitreg1(char* imagepathL, char* imagepathR, int OverlapSamples, char* outpath);

	//����ʽ����ⷽλԪ��
	void Polynomial3_EO(char* EO_txt,double* Poly_C, char* polyCC_txt);
	void Get_PolyEO(double et,double* Poly_C,float* EO);
	//�ڷ�λ����
	void IO_correct(int sample,int BIN,int TDI,float* IO,float* xp,float* yp);
	void IO_correct_arc(float xp,float yp,float* IO,int BIN,int TDI,int* RC);
	//ŷ����ת��ת����
	void Eul2R(float phi,float w,float k,float* R);


	//����+DEM/����ģ��--ȷ�������
	void Get_groundtruth(char* DEMdoc, int sample, int line,int BIN, int TDI, float meanZ, float* EO, float* IO, float* GC);
	void Get_groundtruth(float*IC, float* EO, float f, float Z, float* GC);
	void Get_groundtruth1(float xp ,float yp, float* EO, float f, float rMars, float* GC);
	//���ݵ�������ȷ��Ӱ������
	void Get_imageCoor(float* EO, float* IO, float* GC, float* IC);
	void Get_imageCoor(float* GC, float* EO, float f, float* IC);
	void G2I(float* GC, float* RIO, double* RPoly_C, int* Rrc, double RbeginT, double LR, int Rrows=40000, int BIN=1, int TDI=128);
	//����DEM��ƥ�䣨������������Ӱ�������ȡ�������꣬�ٻ�ȥ��Ӱ���ҵ�����������Լ��ƥ�䣩
	void L2G2R(int* Lrc, float* LIO, float* LEO, float Z, float* RIO, double* RPoly_C, int* Rrc, double RbeginT, double LR, int Rrows=80000, int BIN=1, int TDI=128);
	void L2G2R(int* Lrc, float* LIO, double* LPoly_C, float Z, int* Rrc, float* RIO, double* RPoly_C, double LbeginT, double LLR, double RbeginT, double RLR, int Rrows=80000, int BIN=1, int TDI=128);
	void G_M(char* DEMdoc, int sample, int line, int BIN, int TDI, float meanZ, float* EO, float* IO, int RLines, double DLINE, double RBIN, double RTDI, double et0, double* Poly_C, float* RIO, float* IC);

	void g2itest(char* filepath,char* xulie_ID1,char* xulie_ID2);
	void gmtest(char* filepath,char* xulie_ID1,char* xulie_ID2);
};



#endif
