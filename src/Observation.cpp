#include "Observation.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>

namespace {

char* mutable_cstr(std::string& value) {
	return const_cast<char*>(value.c_str());
}

struct MosaicControlMatch {
	float src_row;
	float src_col;
	float dst_row;
	float dst_col;
};

struct IntraMatchRecord {
	int bj;
	float row;
	float col;
	int imgID;
	float mrow;
	float mcol;
	float score;
};

bool load_mosaic_coefficients(const char* filepath, const char* seq_id, int CCD_num, int* mosaic_c) {
	char mosaictxt[2048];
	snprintf(mosaictxt, sizeof(mosaictxt), "%s/%s/downsample/0/mosaic.txt", filepath, seq_id);
	FILE* fp = fopen(mosaictxt, "r");
	if(fp == NULL){
		printf("未找到拼接系数文件：%s\n", mosaictxt);
		return false;
	}

	int CCD_id;
	for(int ii=0; ii<CCD_num; ++ii){
		if(fscanf(fp, "%d %d %d %d %d\n", &CCD_id, &mosaic_c[ii*4+0], &mosaic_c[ii*4+1], &mosaic_c[ii*4+2], &mosaic_c[ii*4+3]) != 5){
			printf("拼接系数文件读取失败：%s\n", mosaictxt);
			fclose(fp);
			return false;
		}
	}
	fclose(fp);
	return true;
}

bool load_io_row(FILE* fp_io, float* io) {
	int temp_id;
	for(int k=0; k<11; ++k){
		if(k == 0){
			if(fscanf(fp_io, "%d", &temp_id) != 1) return false;
		}
		else if(k == 8 || k == 9){
			if(fscanf(fp_io, "%e", &io[k-1]) != 1) return false;
		}
		else{
			if(fscanf(fp_io, "%f", &io[k-1]) != 1) return false;
		}
	}
	return true;
}

bool load_io_table(const char* IOtxt, int row_count, float io[10][10], const char* context) {
	FILE* fp_io = fopen(IOtxt, "r");
	if(fp_io == NULL){
		printf("[ERROR][%s] 无法打开 IO 文件: %s\n", context, IOtxt);
		return false;
	}
	for(int j=0; j<row_count; ++j){
		if(!load_io_row(fp_io, io[j])){
			printf("[ERROR][%s] IO 文件读取失败: %s row=%d\n", context, IOtxt, j);
			fclose(fp_io);
			return false;
		}
	}
	fclose(fp_io);
	return true;
}

bool compute_grid_neighbor_affine(const char* filepath, const char* xulie_ID1, int layer,
	int ccd, int begin, int end, int CCD_num, const int* mosaic_c1, const int* mosaic_c2,
	const float* global_fs, float* local_fs)
{
	for(int i=0; i<6; ++i) local_fs[i] = global_fs[i];
	if(end <= begin || ccd < begin || ccd >= end){
		return false;
	}

	const int nb = std::max(begin, ccd - 1);
	const int ne = std::min(end - 1, ccd + 1);
	std::vector<int> x1, y1, x2, y2;
	for(int j=nb; j<=ne; ++j){
		char match_path[2048];
		snprintf(match_path, sizeof(match_path), "%s/%s/downsample/%d/%s_RED%d_match.txt",
			filepath, xulie_ID1, layer, xulie_ID1, j);
		FILE* fp = fopen(match_path, "r");
		if(fp == NULL){
			printf("[grid_neighbor_affine] skip missing %s\n", match_path);
			continue;
		}

		while(true){
			int bj;
			if(fscanf(fp, "%d", &bj) != 1) break;
			if(bj == 1){
				float row, col, mrow, mcol, score;
				int imgID;
				if(fscanf(fp, "%f %f %d %f %f %f", &row, &col, &imgID, &mrow, &mcol, &score) != 6) break;
				if(imgID < 0 || imgID >= CCD_num) continue;
				x1.push_back(int(row + mosaic_c1[j*4+0]));
				y1.push_back(int(col + mosaic_c1[j*4+2]));
				x2.push_back(int(mrow + mosaic_c2[imgID*4+0]));
				y2.push_back(int(mcol + mosaic_c2[imgID*4+2]));
			}
			else{
				float row, col;
				if(fscanf(fp, "%f %f", &row, &col) != 2) break;
			}
		}
		fclose(fp);
	}

	if(x1.size() < 3){
		printf("[grid_neighbor_affine] CCD=%d neighbors=[%d,%d] too few points=%zu, use global fs\n",
			ccd, nb, ne, x1.size());
		return false;
	}

	ImageMatch IM;
	std::vector<int> match(x1.size(), -1);
	IM.RANSAC_fs2(x1, y1, x2, y2, 150, 100000, match.data(), local_fs);
	int nin = 0;
	for(size_t i=0; i<match.size(); ++i){
		if(match[i] != -1) ++nin;
	}
	if(nin < 3){
		for(int i=0; i<6; ++i) local_fs[i] = global_fs[i];
		printf("[grid_neighbor_affine] CCD=%d neighbors=[%d,%d] RANSAC inliers=%d/%zu, use global fs\n",
			ccd, nb, ne, nin, x1.size());
		return false;
	}

	printf("[grid_neighbor_affine] CCD=%d neighbors=[%d,%d] points=%zu inliers=%d fs=[%.6g %.6g %.6g %.6g %.6g %.6g]\n",
		ccd, nb, ne, x1.size(), nin,
		local_fs[0], local_fs[1], local_fs[2], local_fs[3], local_fs[4], local_fs[5]);
	return true;
}

int append_inter_ccd_match_points(const char* featurepoint_path, int ccd_index, const int* mosaic_c,
	std::vector<int>& keypoint_rows_ds4, std::vector<int>& keypoint_cols_ds4) {
	FILE* fp = fopen(featurepoint_path, "r");
	if(fp == NULL){
		printf("[WARN][CCD间绘制] 未找到匹配结果文件: %s\n", featurepoint_path);
		return -1;
	}

	const double ds4_scale = 1.0 / 16.0;
	int matched_count = 0;
	while(true){
		int bj;
		if(fscanf(fp, "%d", &bj) != 1){
			break;
		}

		float row, col;
		if(bj == 1){
			int mimgID;
			float mrow, mcol, score;
			if(fscanf(fp, "%f %f %d %f %f %f", &row, &col, &mimgID, &mrow, &mcol, &score) != 6){
				break;
			}
			keypoint_rows_ds4.push_back(int(double(row + mosaic_c[ccd_index*4+0]) * ds4_scale + 0.5));
			keypoint_cols_ds4.push_back(int(double(col + mosaic_c[ccd_index*4+2]) * ds4_scale + 0.5));
			matched_count++;
		}
		else{
			if(fscanf(fp, "%f %f", &row, &col) != 2){
				break;
			}
		}
	}

	fclose(fp);
	return matched_count;
}

int append_inter_seq_match_pairs(const char* featurepoint_path, int src_ccd_index,
	const int* mosaic_src, const int* mosaic_dst, int dst_ccd_num,
	std::vector<int>& src_rows_ds4, std::vector<int>& src_cols_ds4,
	std::vector<int>& dst_rows_ds4, std::vector<int>& dst_cols_ds4) {
	FILE* fp = fopen(featurepoint_path, "r");
	if(fp == NULL){
		printf("[WARN][CCD间绘制] 未找到 Intra__ 文件: %s\n", featurepoint_path);
		return -1;
	}

	const double ds4_scale = 1.0 / 16.0;
	int matched_count = 0;
	while(true){
		int bj;
		if(fscanf(fp, "%d", &bj) != 1){
			break;
		}

		float row, col;
		if(bj == 1){
			int imgID;
			float mrow, mcol, score;
			if(fscanf(fp, "%f %f %d %f %f %f", &row, &col, &imgID, &mrow, &mcol, &score) != 6){
				break;
			}
			if(imgID < 0 || imgID >= dst_ccd_num){
				continue;
			}
			src_rows_ds4.push_back(int(double(row  + mosaic_src[src_ccd_index*4+0]) * ds4_scale + 0.5));
			src_cols_ds4.push_back(int(double(col  + mosaic_src[src_ccd_index*4+2]) * ds4_scale + 0.5));
			dst_rows_ds4.push_back(int(double(mrow + mosaic_dst[imgID*4+0]) * ds4_scale + 0.5));
			dst_cols_ds4.push_back(int(double(mcol + mosaic_dst[imgID*4+2]) * ds4_scale + 0.5));
			matched_count++;
		}
		else{
			if(fscanf(fp, "%f %f", &row, &col) != 2){
				break;
			}
		}
	}

	fclose(fp);
	return matched_count;
}

bool pass_affine_gate(float src_row, float src_col, float dst_row, float dst_col, const float* fs_c, bool reverse) {
	float vx, vy;
	if(!reverse){
		vx = src_row*fs_c[0] + src_col*fs_c[1] + fs_c[2] - dst_row;
		vy = src_row*fs_c[3] + src_col*fs_c[4] + fs_c[5] - dst_col;
	}
	else{
		vx = dst_row*fs_c[0] + dst_col*fs_c[1] + fs_c[2] - src_row;
		vy = dst_row*fs_c[3] + dst_col*fs_c[4] + fs_c[5] - src_col;
	}
	return sqrt(vx*vx + vy*vy) < 500.0f && fabs(vx) < 150.0f && fabs(vy) < 250.0f;
}

bool evaluate_with_global_controls(float src_row, float src_col, float dst_row, float dst_col,
	const std::vector<MosaicControlMatch>& controls, const float* fs_c, bool reverse) {
	(void)fs_c;
	if(controls.empty()){
		return false;
	}

	const double neighbor_radius = 150.0;
	const int min_neighbor_count = 5;
	const double sigma_threshold = 1.0;

	std::vector<int> local_control_ids;
	local_control_ids.reserve(controls.size());
	for(int ii=0; ii<(int)controls.size(); ++ii){
		const float ctrl_src_row = reverse ? controls[ii].dst_row : controls[ii].src_row;
		const float ctrl_src_col = reverse ? controls[ii].dst_col : controls[ii].src_col;
		const double dr = double(ctrl_src_row) - double(src_row);
		const double dc = double(ctrl_src_col) - double(src_col);
		if(sqrt(dr*dr + dc*dc) < neighbor_radius){
			local_control_ids.push_back(ii);
		}
	}
	if((int)local_control_ids.size() < min_neighbor_count){
		return false;
	}

	double dr_mean = 0.0;
	double dc_mean = 0.0;
	for(int ii=0; ii<(int)local_control_ids.size(); ++ii){
		const MosaicControlMatch& ctrl = controls[local_control_ids[ii]];
		const float ctrl_src_row = reverse ? ctrl.dst_row : ctrl.src_row;
		const float ctrl_src_col = reverse ? ctrl.dst_col : ctrl.src_col;
		const float ctrl_dst_row = reverse ? ctrl.src_row : ctrl.dst_row;
		const float ctrl_dst_col = reverse ? ctrl.src_col : ctrl.dst_col;
		dr_mean += double(ctrl_dst_row) - double(ctrl_src_row);
		dc_mean += double(ctrl_dst_col) - double(ctrl_src_col);
	}
	dr_mean /= local_control_ids.size();
	dc_mean /= local_control_ids.size();

	double dr_std = 0.0;
	double dc_std = 0.0;
	double cov_dr_dc = 0.0;
	for(int ii=0; ii<(int)local_control_ids.size(); ++ii){
		const MosaicControlMatch& ctrl = controls[local_control_ids[ii]];
		const float ctrl_src_row = reverse ? ctrl.dst_row : ctrl.src_row;
		const float ctrl_src_col = reverse ? ctrl.dst_col : ctrl.src_col;
		const float ctrl_dst_row = reverse ? ctrl.src_row : ctrl.dst_row;
		const float ctrl_dst_col = reverse ? ctrl.src_col : ctrl.dst_col;
		const double ctrl_dr = double(ctrl_dst_row) - double(ctrl_src_row);
		const double ctrl_dc = double(ctrl_dst_col) - double(ctrl_src_col);
		dr_std += (ctrl_dr - dr_mean) * (ctrl_dr - dr_mean);
		dc_std += (ctrl_dc - dc_mean) * (ctrl_dc - dc_mean);
		cov_dr_dc += (ctrl_dr - dr_mean) * (ctrl_dc - dc_mean);
	}
	dr_std /= local_control_ids.size();
	dc_std /= local_control_ids.size();
	cov_dr_dc /= local_control_ids.size();

	const double cand_dr = double(dst_row) - double(src_row);
	const double cand_dc = double(dst_col) - double(src_col);
	const double delta_dr = cand_dr - dr_mean;
	const double delta_dc = cand_dc - dc_mean;

	dr_std = std::max(dr_std, 1.0);
	dc_std = std::max(dc_std, 1.0);

	double det = dr_std * dc_std - cov_dr_dc * cov_dr_dc;
	if(det < 1e-6){
		const double z_dr = fabs(delta_dr) / sqrt(dr_std);
		const double z_dc = fabs(delta_dc) / sqrt(dc_std);
		return z_dr <= sigma_threshold && z_dc <= sigma_threshold;
	}

	const double maha2 =
		( dc_std * delta_dr * delta_dr
		- 2.0 * cov_dr_dc * delta_dr * delta_dc
		+ dr_std * delta_dc * delta_dc ) / det;
	return maha2 <= sigma_threshold * sigma_threshold;
}

bool load_polyCC_cache(const char* path, double* Poly_C) {
	FILE* fp = fopen(path, "r");
	if(fp == NULL) return false;
	for(int ii=0; ii<6; ++ii){
		double a1,a2,a3,a4,a5;
		if(fscanf(fp, "%lf %lf %lf %lf %lf\n", &a1,&a2,&a3,&a4,&a5) != 5){
			fclose(fp);
			return false;
		}
		Poly_C[ii*5+0]=a1; Poly_C[ii*5+1]=a2; Poly_C[ii*5+2]=a3;
		Poly_C[ii*5+3]=a4; Poly_C[ii*5+4]=a5;
	}
	fclose(fp);
	return true;
}

bool load_eo_begin_lr(const char* path, double* beginET, double* LR) {
	FILE* fp = fopen(path, "r");
	if(fp == NULL) return false;
	const int n = fscanf(fp, "%lf %lf\n", beginET, LR);
	fclose(fp);
	return n == 2;
}

}

// some func
// rec:XYZ; re:长轴 f:扁率 lon,lat:弧度单位制
template <typename T>
void Observation::rec2geo(T rec[3],T re,T f,T* lon,T* lat,T* H){
	T X = rec[0];T Y = rec[1];T Z = rec[2];
	T a = re; T b = a - a*f;
	T ec = T(1) - b*b/a/a;
	T ecc = ec/(T(1)-ec);
	T Lon0 = atan(Y/X);
	if(X<T(0)){
		Lon0 += 3.1415926;
	}
	T R = sqrt(X*X+Y*Y);
	T B1 = atan(Z/R);
	T B2;
	while(true){
		T w1 = sqrt(T(1.0)-ec*(sin(B1)*sin(B1)));
		T n1 = a/w1;
		B2 = atan((Z+n1*ec*sin(B1))/R);
		if(abs(B1-B2)<=T(0.0000000001)){
			break;
		}
		B1 = B2;
	}
	*lon = Lon0;
	*lat = B2;
	T W = sqrt(T(1.0)-ec*(sin(B2)*sin(B2)));
	T N = a/W;
	*H = R/cos(B2) - N;
}

//外方位元素的三次多项式拟合EO(t)=EO(t0)+c1(t-t0)+c2(t-t0)^2+c3(t-t0)^3
void Observation::Polynomial3_EO1(char* EO_txt,double* Poly_C, char* polyCC_txt){
	FILE *fp=fopen(EO_txt,"r");
	if(fp==NULL){
		std::cout<<"Error opening EO file: "<<EO_txt<<std::endl;
		return;
	}
	double beginT,LR;
	if(fscanf(fp,"%lf %lf\n",&beginT,&LR) != 2){
		std::cout<<"[ERROR][Polynomial3_EO1] EO 文件头读取失败: "<<EO_txt<<std::endl;
		fclose(fp);
		return;
	}
	std::vector<double> EO;
	double et,Xs,Ys,Zs,phi,w,ka;
	while(fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",&et,&Xs,&Ys,&Zs,&phi,&w,&ka) == 7){
		EO.push_back(et);
		EO.push_back(Xs);EO.push_back(Ys);EO.push_back(Zs);
		EO.push_back(phi);EO.push_back(w);EO.push_back(ka);
	}
	fclose(fp);

	double t,t0;
	int number=EO.size()/7;
	if(number < 4){
		std::cout<<"[ERROR][Polynomial3_EO1] EO 采样不足 ("<<number
			<<")，无法拟合: "<<EO_txt<<std::endl;
		return;
	}
	t0=EO[0];
	MatrixXd x;
	FILE *fp1=fopen(polyCC_txt,"w");
	//fprintf(fp1,"%lf %lf\n",beginT,LR);
	for(int j=1;j<7;j++){
		MatrixXd A=MatrixXd::Zero(number,4);
		MatrixXd L=MatrixXd::Zero(number,1);
		for(int i=0;i<number;i++){
			VectorXd temp1(4);
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
void Observation::Polynomial3_EO(char* EO_txt,double* Poly_C, char* polyCC_txt){
	FILE *fp=fopen(EO_txt,"r");
	if(fp==NULL){
		std::cout<<"无法打开EO文件："<<EO_txt<<std::endl;
		return;
	}
	double beginT,LR;
	if(fscanf(fp,"%lf %lf\n",&beginT,&LR) != 2){
		std::cout<<"[ERROR][Polynomial3_EO] EO 文件头读取失败: "<<EO_txt<<std::endl;
		fclose(fp);
		return;
	}
	std::vector<double> EO;
	double et,Xs,Ys,Zs,phi,w,ka;
	while(fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",&et,&Xs,&Ys,&Zs,&phi,&w,&ka) == 7){
		EO.push_back(et);
		EO.push_back(Xs);EO.push_back(Ys);EO.push_back(Zs);
		EO.push_back(phi);EO.push_back(w);EO.push_back(ka);
	}
	fclose(fp);

	double t,t0;
	int number=EO.size()/7;
	if(number < 4){
		std::cout<<"[ERROR][Polynomial3_EO] EO 采样不足 ("<<number
			<<")，无法拟合: "<<EO_txt<<std::endl;
		return;
	}
	t0=EO[0];
	MatrixXd x;
	FILE *fp1=fopen(polyCC_txt,"w");
	//fprintf(fp1,"%lf %lf\n",beginT,LR);
	for(int j=1;j<7;j++){
		MatrixXd A=MatrixXd::Zero(number,4);
		MatrixXd L=MatrixXd::Zero(number,1);
		for(int i=0;i<number;i++){
			VectorXd temp1(4);
			t=EO[i*7];
			temp1 << 1.0,(t-t0),pow((t-t0),2),pow((t-t0),3);   //三次多项式
			//temp1 << 1.0,(t-t0),sin(t-t0),sin(2*(t-t0));
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
void Observation::Get_PolyEO1(double et,double* Poly_C,float* EO){
	for(int i=0;i<6;i++){
		double t=et;
		double t0=Poly_C[5*i];
		EO[i]=Poly_C[5*i+1]+Poly_C[5*i+2]*(t-t0)+Poly_C[5*i+3]*pow((t-t0),2)+Poly_C[5*i+4]*pow((t-t0),3);
	}
}
void Observation::Get_PolyEO(double et,double* Poly_C,float* EO){
	for(int i=0;i<6;i++){
		double t=et;
		double t0=Poly_C[5*i];
		EO[i]=Poly_C[5*i+1]+Poly_C[5*i+2]*(t-t0)+Poly_C[5*i+3]*pow((t-t0),2)+Poly_C[5*i+4]*pow((t-t0),3);
		//EO[i]=Poly_C[5*i+1]+Poly_C[5*i+2]*(t-t0)+Poly_C[5*i+3]*sin(t-t0)+Poly_C[5*i+4]*sin(2*(t-t0));
	}
}

//内方位纠正;IO:x0,dx/ds,dx/dl,y0,dy/ds,dy/dl,k0,k1,k2,f
void Observation::Eul2R(float phi,float w,float k,float* R){
	phi=-phi;
	R[0]=cos(phi)*cos(k)-sin(phi)*sin(w)*sin(k);
	R[1]=-cos(phi)*sin(k)-sin(phi)*sin(w)*cos(k);
	//R[1]=-R[1];
	R[2]=-sin(phi)*cos(w);
	R[3]=cos(w)*sin(k);
	//R[3]=-R[3];
	R[4]=cos(w)*cos(k);
	R[5]=-sin(w);
	//R[5]=-R[5];
	R[6]=sin(phi)*cos(k)+cos(phi)*sin(w)*sin(k);
	R[7]=-sin(phi)*sin(k)+cos(phi)*sin(w)*cos(k);
	//R[7]=-R[7];
	R[8]=cos(phi)*cos(w);
}
void Observation::IO_correct(int sample,int BIN,int TDI,float* IO,float* xp,float* yp){
	float s = (float(sample)-0.5)*float(BIN)-1024;
	float l = float(TDI)/2-64-(float(BIN)/2-0.5);

	float x = IO[0]+IO[1]*s+IO[2]*l;
	float y = IO[3]+IO[4]*s+IO[5]*l;

	float r = sqrt(x*x+y*y);
	float dr_r = IO[6]+IO[7]*r*r+IO[8]*r*r*r*r;

	*xp = x-dr_r*x;
	*yp = y-dr_r*y;
}
void Observation::IO_correct(float sample,int BIN,int TDI,float* IO,float* xp,float* yp){
	float s = (float(sample)-0.5)*float(BIN)-1024;
	float l = float(TDI)/2-64-(float(BIN)/2-0.5);

	float x = IO[0]+IO[1]*s+IO[2]*l;
	float y = IO[3]+IO[4]*s+IO[5]*l;

	float r = sqrt(x*x+y*y);
	float dr_r = IO[6]+IO[7]*r*r+IO[8]*r*r*r*r;

	*xp = x-dr_r*x;
	*yp = y-dr_r*y;
}
void Observation::Get_groundtruth(float xp ,float yp, float* EO, float f, float rMars, float* GC){
	//double rMars=3396190;
	//读取坐标系信息

	float R[9];
	//Eul2R(EO[3]*3.14159/180,EO[4]*3.14159/180,EO[5]*3.14159/180,R);
	Eul2R(EO[3],EO[4],EO[5],R);
	//Eul2R(0,0,0,R);

	float xi[3];
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
void Observation::Get_groundtruth(float xp ,float yp, float* EO, MatrixXf dR, float* dS, float f, float rMars, float* GC){
	//double rMars=3396190;
	//读取坐标系信息

	float R[9];
	//Eul2R(EO[3]*3.14159/180,EO[4]*3.14159/180,EO[5]*3.14159/180,R);
	Eul2R(EO[3],EO[4],EO[5],R);
	//Eul2R(0,0,0,R);

	float xi[3];
	xi[0] = xp;
	xi[1] = yp;
	xi[2] = f;

	//R转换到eigen格式，通过eigen矩阵运算计算X，Y
	MatrixXf R_ = dR*(Map<MatrixXf>(R,3,3)).transpose();
	MatrixXf xi_ = Map<MatrixXf>(xi,3,1);
	VectorXf Rx = R_*xi_;

	Vector3f XYZs;
	XYZs<<EO[0]-dS[0],EO[1]-dS[1],EO[2]-dS[2];
	XYZs=dR*XYZs;


	float Xs = EO[0]-dS[0];
	float Ys = EO[1]-dS[1];
	float Zs = EO[2]-dS[2];

	float X,Y,Z;
	Z=0;

	float nlamda=Z-XYZs(2)/Rx(2);

	X=nlamda*Rx(0)+XYZs(0);
	Y=nlamda*Rx(1)+XYZs(1);
	Z=nlamda*Rx(2)+XYZs(2);

	GC[0]=X;
	GC[1]=Y;
	GC[2]=Z;
}
void Observation::Get_groundtruth(char* DEMdoc, int sample, int line,int BIN, int TDI, float meanZ, float* EO, float* IO, float* GC){
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
	float R[9];
	//Eul2R(EO[3]*3.14159/180,EO[4]*3.14159/180,EO[5]*3.14159/180,R);
	Eul2R(EO[3],EO[4],EO[5],R);
	//Eul2R(0,0,0,R);

	float f = IO[9];
	float xi[3];
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
void Observation::Get_imageCoor(float* EO, MatrixXf dR, float* dS, float* IO, float* GC, float* IC){
	float R[9];
	Eul2R(EO[3],EO[4],EO[5],R);
	MatrixXf R_inv = (dR*(Map<MatrixXf>(R,3,3)).transpose()).transpose();

	Vector3f XYZs;
	XYZs<<EO[0]-dS[0],EO[1]-dS[1],EO[2]-dS[2];
	XYZs=dR*XYZs;

	Vector3f XYZ_;
	XYZ_<<GC[0]-XYZs(0),GC[1]-XYZs(1),GC[2]-XYZs(2);

	VectorXf Rx = R_inv*XYZ_;

	float x0 = IO[0];
	float y0 = IO[3];
	float f = IO[9];
	float lamda = f/Rx(2);

	IC[0] = lamda*Rx(0)+x0;
	IC[1] = lamda*Rx(1)+y0;
}
void Observation::G2I(float* GC, float* RIO, double* RPoly_C, int* Rrc, double RbeginT, double LR, MatrixXf dR, float* dS, int Rrows, int BIN, int TDI){
	ImageMatch IM;
	float IC[2];
	//地面点反投至右影像（需要拟合方程）
	int diedai_count = 0;
	float REO[6];
	double et = RbeginT + int(Rrows/2)*LR*BIN;
	Get_PolyEO(et,RPoly_C,REO);
	Get_imageCoor(REO, dR, dS, RIO,GC, IC);
	IM.IO_correct_arc(IC[0],IC[1],RIO,BIN,TDI,Rrc);
	while(abs(Rrc[0]-0)>=1 && diedai_count<100){
		//printf("%d\n",BIN);
		et = et+Rrc[0]*LR*BIN;
		Get_PolyEO(et,RPoly_C,REO);
		Get_imageCoor(REO, dR, dS, RIO, GC, IC);
		//printf("%f %f\n",IC[0],IC[1]);
		IM.IO_correct_arc(IC[0],IC[1],RIO,BIN,TDI,Rrc);
		//printf("%d %d\n",Rrc[0],Rrc[1]);
		diedai_count++;
	}

	if(Rrc[0]==0){
		Rrc[0]=int((et-RbeginT)/LR+0.5);
	}
	if(diedai_count==100){
		printf("diedai:%d\n",diedai_count);
	}
}



//原始分辨率相邻CCD影像匹配、拼接，生成框幅影像，并记录dr、dc
// 仅拼接 [ccd_begin, ccd_end)；mosaic.txt 仍写满 CCD_num 行（范围外填 0），保证按 CCD 下标读取
void Observation::xulie_mosaic1(char* src_path, char* prefix, int OverlapSamples, int seq_index){
	int CCD_num = cfg_.CCD_num;
	const int ccd_begin = cfg_.ccd_begin();
	const int ccd_end = cfg_.ccd_end();
	if(ccd_end - ccd_begin < 1){
		printf("xulie_mosaic1: CCD range [%d,%d) empty\n", ccd_begin, ccd_end);
		return;
	}
	int rows = seq_index==0 ? cfg_.rows1 : cfg_.rows2;
	int cols = cfg_.cols;
	std::vector<int> dr(CCD_num, 0);
	std::vector<int> dc(CCD_num, 0);
	ImageProcess IPtemp;

		printf("xulie_mosaic1: CCD range [%d,%d)\n", ccd_begin, ccd_end);
		for(int i=ccd_begin;i<ccd_end-1;i++){
			char img_path1[2048];
			snprintf(img_path1, sizeof(img_path1), "%s/%s%d%s", src_path, prefix, i, ".tif");
			std::cout << "img_path1: " << img_path1 << std::endl;
			char img_path2[2048];
			snprintf(img_path2, sizeof(img_path2), "%s/%s%d%s", src_path, prefix, i+1, ".tif");
			std::cout << "img_path2: " << img_path2 << std::endl;
			IPtemp.Hijitreg_gdal(img_path1, img_path2, OverlapSamples, &(dr[i]), &(dc[i]));
		}

	std::vector<int> beginR(CCD_num, 0);
	std::vector<int> endR(CCD_num, 0);
	std::vector<int> beginC(CCD_num, 0);
	std::vector<int> endC(CCD_num, 0);

	int minR=65535;
	int maxR=-65535;
	int minC=65535;
	int maxC=-65535;

	// 以 ccd_begin 为拼接原点，仅在范围内链式累计
	beginR[ccd_begin] = 0;
	endR[ccd_begin] = rows;
	beginC[ccd_begin]=0;
	endC[ccd_begin]=cols;
	minR=beginR[ccd_begin]; maxR=endR[ccd_begin];
	minC=beginC[ccd_begin]; maxC=endC[ccd_begin];
	for(int i=ccd_begin+1;i<ccd_end;i++){
		beginR[i]=beginR[i-1]+dr[i-1];
		endR[i]=beginR[i]+rows;
		beginC[i]=endC[i-1]-OverlapSamples+dc[i-1];
		endC[i]=beginC[i]+cols;
		if(beginR[i]<minR) minR=beginR[i];
		if(endR[i]>maxR) maxR=endR[i];
		if(beginC[i]<minC) minC=beginC[i];
		if(endC[i]>maxC) maxC=endC[i];
	}

		char mosaictxt[2048];
		snprintf(mosaictxt, sizeof(mosaictxt), "%s%s", src_path, "/mosaic.txt");
		FILE *fp=fopen(mosaictxt,"w");
	for(int i=0;i<CCD_num;i++){
		if(i>=ccd_begin && i<ccd_end){
			beginR[i]-= minR;
			endR[i]  -= minR;
			beginC[i]-= minC;
			endC[i]  -= minC;
		} else {
			beginR[i]=endR[i]=beginC[i]=endC[i]=0;
		}
		fprintf(fp,"%d %d %d %d %d\n",i,beginR[i],endR[i],beginC[i],endC[i]);
	}
	fclose(fp);

	rows=maxR-minR;
	cols=maxC-minC;
	if(rows<=0 || cols<=0){
		printf("xulie_mosaic1: invalid mosaic size %d x %d\n", rows, cols);
		return;
	}

	//生成拼接框幅
		GDALAllRegister();
		GDALDataType Type0 = GDT_Byte;
		char mosaic_path[2048];
		snprintf(mosaic_path, sizeof(mosaic_path), "%s%s", src_path, "/mosaic.tif");
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF"); //图像驱动
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED"); //配置图像信息
	GDALDataset* dst = pDriver->Create(mosaic_path, cols, rows, 1, GDT_Byte, ppszOptions);
	if (dst == nullptr)
	{
		printf("Can't Create Image!");
		CSLDestroy(ppszOptions);
		return;
	}

	//读取原始影像灰度并写入（仅范围内 CCD）
	int rows1,cols1;
	int rows0,cols0;
	for(int i=ccd_begin;i<ccd_end;i++){
			GDALDataset *poDataset;
			GDALRasterBand *poBand;
			char img_path1[2048];
			snprintf(img_path1, sizeof(img_path1), "%s%s%s%d%s", src_path, "/", prefix, i, ".tif");
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

		int npartc=4;
		int npartr=1;
		int rows1=rows0/npartr;
		int cols1=cols0/npartc;
		for(int j=0;j<npartc;j++){
			if(j==npartc-1){
				cols1=cols0-j*cols1;
			}
			for(int k=0;k<npartr;k++){
				uchar* imgData1 = new uchar[rows1*cols1];
				if(poBand->RasterIO(GF_Read, (cols0/npartc)*j, rows1*k, cols1, rows1,
					imgData1, cols1, rows1, Type0, 0, 0 ) != CE_None){
					delete []imgData1;
					continue;
				}

				if(dst->GetRasterBand(1)->RasterIO(GF_Write,
					beginC[i]+(cols0/npartc)*j,
					beginR[i]+rows1*k,
					cols1,
					rows1,
					imgData1,
					cols1,
					rows1,
					Type0,
					0,
					0) != CE_None){
					printf("Can't Write Image!");
					delete []imgData1;
					GDALClose(poDataset);
					GDALClose(dst);
					CSLDestroy(ppszOptions);
					return;
				}
				delete []imgData1;
			}
		}
		GDALClose(poDataset);
	}

	GDALClose(dst);
		CSLDestroy(ppszOptions);
		char ds_path1[2048];
		snprintf(ds_path1, sizeof(ds_path1), "%s%s", src_path, "/mosaic_ds4.tif");
		IPtemp.Down_sample(mosaic_path, 16, ds_path1);
	}

//影像降采样并存储（分四级：2，4，8，16）
void Observation::xulie_downsample(){
	const char* filepath = cfg_.filepath;
	const char* xulie_ID1 = cfg_.xulie_ID1;
	const char* xulie_ID2 = cfg_.xulie_ID2;
	int CCD_num=cfg_.CCD_num;

	for(int ii=0;ii<2;ii++){
		const char* xulie_ID;
		if(ii==0){
			xulie_ID=xulie_ID1;
		}
		else if(ii==1){
			xulie_ID=xulie_ID2;
		}

		//cout<<xulie_ID<<endl;

		std::string prefix = std::string(xulie_ID) + "_RED";

		std::string downsample_path = std::string(filepath) + "/" + xulie_ID + "/downsample";
		int status = mkdir(downsample_path.c_str(), 0755);
		(void)status;

		std::string src_path = downsample_path + "/0";

		for(int j=1;j<5;j++){
			//新建文件夹
			std::string ds_path = downsample_path + "/" + std::to_string(j);
			status = mkdir(ds_path.c_str(), 0755);

			//降采样
			int batchsize=pow(double(2),double(j));
			ImageProcess IPtemp;
			for(int i=cfg_.ccd_begin();i<cfg_.ccd_end();i++){
				std::string img_path = src_path + "/" + prefix + std::to_string(i) + ".tif";
				std::string dst_path = ds_path + "/" + prefix + std::to_string(i) + ".tif";
				IPtemp.Down_sample(mutable_cstr(img_path), batchsize, mutable_cstr(dst_path));
			}
		}
	}
}
void Observation::prepare_feature_match_mosaic(int have_mosaic0){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	std::string filepath1 = filepath;

	std::cout<<"开始拼接"<<std::endl;

	//原始分辨率拼接&输出每级拼接参数及影像到downsample文件夹
	for(int i=0;i<2;i++){
		const char* xulie_ID = nullptr;
		int rows;
		if(i==0){
			xulie_ID = xulie_ID1;
			rows=rows1;
		}
		else if(i==1){
			xulie_ID = xulie_ID2;
			rows=rows2;
		}

			std::string src_path = std::string(filepath) + "/" + xulie_ID + "/downsample/0";
			std::string prefix = std::string(xulie_ID) + "_RED";
		int OverlapSamples=cfg_.mosaic.overlap_samples;

		if(!have_mosaic0){
			std::cout<<"开始拼接原始分辨率影像"<<std::endl;
			xulie_mosaic1(mutable_cstr(src_path), mutable_cstr(prefix), OverlapSamples, i);
			std::cout<<"原始分辨率影像拼接完成"<<std::endl;
		}

			std::string mosaictxt = filepath1 + "/" + xulie_ID + "/downsample/0/mosaic.txt";
		FILE *fp0=fopen(mosaictxt.c_str(),"r");
		if(fp0==NULL){
			return;
		}
		int CCD_id,beginR,endR,beginC,endC;
		CCD_id=beginR=endR=beginC=endC=0;

		std::cout<<"开始生成降采样拼接参数"<<std::endl;

			for(int j=1;j<5;j++){
				std::string ds_path_mosaictxt = filepath1 + "/" + xulie_ID + "/downsample/" + std::to_string(j) + "/mosaic.txt";
			std::string dir_path= filepath1 + "/" + std::string(xulie_ID) + "/downsample/" + std::to_string(j);
			IP_.createDirectoryRecursive(dir_path);
			FILE *fp=fopen(ds_path_mosaictxt.c_str(),"w");
			int ds_r=pow(double(2),double(j));
				while(fscanf(fp0,"%d %d %d %d %d\n",&CCD_id,&beginR,&endR,&beginC,&endC) == 5){
					fprintf(fp,"%d %d %d %d %d\n",CCD_id,beginR/ds_r,endR/ds_r,beginC/ds_r,endC/ds_r);
				}
			fclose(fp);
			rewind(fp0);
		}
		fclose(fp0);
	}
}

void Observation::fenfu_extract(int mark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int CCD_num=cfg_.CCD_num;
	char filepath1[2048];
	strncpy(filepath1, filepath, sizeof(filepath1)-1);

	std::cout<<"开始提取特征点"<<std::endl;

	int layers=0;
	if(mark==0){
		layers=4;
	}
	int ch=cfg_.feature_extract.channel;
	int Localmax_win[5];
	for(int idx=0; idx<5; idx++){
		Localmax_win[idx]=cfg_.feature_extract.localmax_win[idx];
	}
	const int ccd_begin = cfg_.ccd_begin();
	const int ccd_end = cfg_.ccd_end();
	std::cout<<"CCD range ["<<ccd_begin<<","<<ccd_end<<")"<<std::endl;

	for(int i=layers;i>=0;i--){
		#pragma omp parallel for schedule(dynamic)
		for(int j=ccd_begin;j<ccd_end;j++){
			printf("提取特征点：L-%d-%d……\n",i,j);
			char imgL_path[2048];
			snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID1, i, xulie_ID1, j);
			char featurepoint1[2048];
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			std::vector<int> KeyPoint_x1,KeyPoint_y1;
			ImageMatch IM;
			IM.Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			FILE *fp_f=fopen(featurepoint1,"w");
			for(size_t idx=0;idx<KeyPoint_x1.size();idx++){
				fprintf(fp_f,"%d %d %d\n",0,KeyPoint_x1[idx],KeyPoint_y1[idx]);
			}
			vector<int>().swap(KeyPoint_x1);
			vector<int>().swap(KeyPoint_y1);
			fclose(fp_f);
		}

		#pragma omp parallel for schedule(dynamic)
		for(int j=ccd_begin;j<ccd_end;j++){
			printf("提取特征点：R-%d-%d……\n",i,j);
			char imgL_path[2048];
			snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID2, i, xulie_ID2, j);
			char featurepoint1[2048];
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID2, i, xulie_ID2, j);

			std::vector<int> KeyPoint_x1,KeyPoint_y1;
			ImageMatch IM;
			IM.Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			FILE *fp_f=fopen(featurepoint1,"w");
			for(size_t idx=0;idx<KeyPoint_x1.size();idx++){
				fprintf(fp_f,"%d %d %d\n",0,KeyPoint_x1[idx],KeyPoint_y1[idx]);
			}
			vector<int>().swap(KeyPoint_x1);
			vector<int>().swap(KeyPoint_y1);
			fclose(fp_f);
		}
	}
}

void Observation::fenfu_match1(float* fs_c,int mark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	char filepath1[2048];
	strncpy(filepath1, filepath, sizeof(filepath1)-1);
	filepath1[sizeof(filepath1)-1] = '\0';

	int layers=0;
	int kc=1;
	if(mark==0){
		kc=2;
		layers=4;
	}
	else if(mark==1){
		kc=1;
		layers=0;
	}
	const PipelineConfig::FeatureMatchParams& match_cfg = cfg_.feature_match;
	const bool use_local = match_cfg.use_local_affine;
	const bool do_densify = match_cfg.densify_grid;
	const int ccd_begin = cfg_.ccd_begin();
	const int ccd_end = cfg_.ccd_end();
	std::cout<<"CCD range ["<<ccd_begin<<","<<ccd_end<<")"<<std::endl;

	ImageMatch IMtemp;
	LocalAffineField guide_field;
	guide_field.set_global(fs_c);
	bool guide_ready = false;

	//逐层匹配
	int CCD_id;
	int BJ=0;
	for(int i=layers;i>=0;i--){
		std::cout<<"开始匹配第"<<i<<"层……"<<std::endl;
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		std::string ds_path_mosaictxt1 = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/" + std::to_string(i) + "/mosaic.txt";
		FILE *fpm1=fopen(ds_path_mosaictxt1.c_str(),"r");
		if(fpm1==NULL){
			std::cout<<"无法打开拼接系数文件："<<ds_path_mosaictxt1<<std::endl;
			return;
		}
		std::vector<int> mosaic_c1(CCD_num*4);
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]) != 5){
				std::cout<<"拼接系数文件读取失败："<<ds_path_mosaictxt1<<std::endl;
				fclose(fpm1);
				return;
			}
		}
		fclose(fpm1);
		std::string ds_path_mosaictxt2 = std::string(filepath1) + "/" + xulie_ID2 + "/downsample/" + std::to_string(i) + "/mosaic.txt";
		FILE *fpm2=fopen(ds_path_mosaictxt2.c_str(),"r");
		if(fpm2==NULL){
			std::cout<<"无法打开拼接系数文件："<<ds_path_mosaictxt2<<std::endl;
			return;
		}
		std::vector<int> mosaic_c2(CCD_num*4);
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]) != 5){
				std::cout<<"拼接系数文件读取失败："<<ds_path_mosaictxt2<<std::endl;
				fclose(fpm2);
				return;
			}
		}
		fclose(fpm2);

		// 匹配：点级 OpenMP 在 limit_match1 内；CCD 外层串行，避免嵌套过订阅
		const int n_ccd = ccd_end - ccd_begin;
		const int n_pairs = std::max(1, n_ccd * n_ccd);
		int pair_done = 0;
		const double t0_layer = omp_get_wtime();
		const int rowsL = std::max(1, cfg_.rows1 >> i);
		const int colsLR = cfg_.cols;
		printf("[fenfu_match] layer=%d pairs=%d guide=%d threads=%d\n",
			i, n_pairs, (int)(use_local && guide_ready), omp_get_max_threads());
		fflush(stdout);

		for(int j=ccd_begin;j<ccd_end;j++){
			ImageMatch IM_local;
			std::string imgL_path = std::string(filepath) + "/" + xulie_ID1 + "/downsample/" + std::to_string(i) + "/" + xulie_ID1 + "_RED" + std::to_string(j) + ".tif";
			std::string featurepoint1 = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/" + std::to_string(i) + "/" + xulie_ID1 + "_RED" + std::to_string(j) + ".txt";
			std::string outpoint1 = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/" + std::to_string(i) + "/" + xulie_ID1 + "_RED" + std::to_string(j) + "_.txt";
			for(int k=ccd_begin;k<ccd_end;k++){
				++pair_done;
				const double t0_pair = omp_get_wtime();
				printf("[fenfu_match] L%d R%d  pair %d/%d (%.0f%%) layer_elapsed=%.1fs\n",
					j, k, pair_done, n_pairs, 100.0 * pair_done / n_pairs,
					omp_get_wtime() - t0_layer);
				fflush(stdout);

				std::string imgR_path = std::string(filepath) + "/" + xulie_ID2 + "/downsample/" + std::to_string(i) + "/" + xulie_ID2 + "_RED" + std::to_string(k) + ".tif";
				std::string featurepoint2 = std::string(filepath1) + "/" + xulie_ID2 + "/downsample/" + std::to_string(i) + "/" + xulie_ID2 + "_RED" + std::to_string(k) + ".txt";
				if(i==4){
					printf("[fenfu_match] layer4 CC_match L%d-R%d\n", j, k);
					IM_local.CC_match2(mutable_cstr(imgL_path), mutable_cstr(imgR_path), match_cfg.coarse_window_size, match_cfg.coarse_cc_threshold, 2, fs_c, k, mutable_cstr(featurepoint1), mutable_cstr(featurepoint2), mutable_cstr(outpoint1));
				}
				else{
					float fs_c1[6];
					fs_c1[0]=fs_c[0];
					fs_c1[1]=fs_c[1];
					fs_c1[2]=fs_c[2]*kc+fs_c[0]*mosaic_c1[j*4+0]+fs_c[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
					fs_c1[3]=fs_c[3];
					fs_c1[4]=fs_c[4];
					fs_c1[5]=fs_c[5]*kc+fs_c[3]*mosaic_c1[j*4+0]+fs_c[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];

					// 粗筛：仿射列坐标完全落在右影像外则跳过（省读图+梯度）
					const float c00 = fs_c1[3]*0 + fs_c1[4]*0 + fs_c1[5];
					const float c01 = fs_c1[3]*0 + fs_c1[4]*colsLR + fs_c1[5];
					const float c10 = fs_c1[3]*rowsL + fs_c1[4]*0 + fs_c1[5];
					const float c11 = fs_c1[3]*rowsL + fs_c1[4]*colsLR + fs_c1[5];
					const bool no_overlap =
						(c00 < 0 && c01 < 0 && c10 < 0 && c11 < 0) ||
						(c00 > colsLR && c01 > colsLR && c10 > colsLR && c11 > colsLR);
					if (no_overlap) {
						printf("[fenfu_match] skip L%d-R%d (no overlap) %.2fs\n",
							j, k, omp_get_wtime() - t0_pair);
						fflush(stdout);
						continue;
					}

					const int w_size = (i==0) ? match_cfg.final_layer_window_size : match_cfg.guided_window_factor*(5-i);
					const int ser = match_cfg.search_range_factor*(5-i);
					const float th = (i==0) ? match_cfg.final_layer_cc_threshold : match_cfg.guided_cc_threshold;

					if(use_local && guide_ready){
						IM_local.limit_match1_field(mutable_cstr(imgL_path), mutable_cstr(imgR_path), w_size, ser, th, 2, fs_c1, k,
							mutable_cstr(featurepoint1), mutable_cstr(featurepoint2), mutable_cstr(outpoint1),
							&guide_field,
							mosaic_c1[j*4+0], mosaic_c1[j*4+2],
							mosaic_c2[k*4+0], mosaic_c2[k*4+2]);
					}
					else{
						IM_local.limit_match1(mutable_cstr(imgL_path), mutable_cstr(imgR_path), w_size, ser, th, 2, fs_c1, k, mutable_cstr(featurepoint1), mutable_cstr(featurepoint2), mutable_cstr(outpoint1));
					}
				}
				remove(featurepoint1.c_str());
				rename(outpoint1.c_str(),featurepoint1.c_str());
				printf("[fenfu_match] done L%d-R%d %.1fs\n", j, k, omp_get_wtime() - t0_pair);
				fflush(stdout);
			}
		}
		printf("[fenfu_match] layer=%d finished elapsed=%.1fs\n", i, omp_get_wtime() - t0_layer);
		fflush(stdout);

		//i结束后，进行匹配点筛选和重排，加载拼接系数并计算fs系数。
		std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
		std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
		for(int j=ccd_begin;j<ccd_end;j++){
			std::string featurepoint1 = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/" + std::to_string(i) + "/" + xulie_ID1 + "_RED" + std::to_string(j) + ".txt";
			std::cout<<"读取特征点："<<featurepoint1<<std::endl;
			FILE *fp=fopen(featurepoint1.c_str(),"r");
			if(fp==NULL){
				std::cout<<"无法打开特征点文件："<<featurepoint1<<std::endl;
				continue;
			}
			int bj,row,col,imgID,mrow,mcol;
			float m_score;
			double sfr=pow(double(2),double(i))/pow(double(2),double(4));
			while(fscanf(fp,"%d ",&bj) == 1){
				if(bj==1){
					if(fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score) != 6){
						break;
					}
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
					if(fscanf(fp,"%d %d\n",&row,&col) != 2){
						break;
					}
				}
			}
			fclose(fp);
		}
		std::vector<int> match(KeyPoint_x1.size(), -1);
		float ransac_sigma = match_cfg.ransac_sigma_factor * static_cast<float>(5 - i);
		if (i <= 3) {
			ransac_sigma *= 8.f;
		}
		std::cout<<"开始RANSAC匹配（全局，仅更新 fs_c） layer="<<i
			<<" points="<<KeyPoint_x1.size()
			<<" iter="<<match_cfg.ransac_iterations
			<<" sigma="<<ransac_sigma<<std::endl;
		fflush(stdout);
		IMtemp.RANSAC_fs2(KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,ransac_sigma,match_cfg.ransac_iterations,match.data(),fs_c);
		std::cout<<"RANSAC匹配完成 layer="<<i
			<<" fs=["<<fs_c[0]<<","<<fs_c[1]<<","<<fs_c[2]<<","
			<<fs_c[3]<<","<<fs_c[4]<<","<<fs_c[5]<<"]"<<std::endl;
		fflush(stdout);

		// 分块局部仿射：用于可视化剔点 + 引导下一层
		LocalAffineField layer_field;
		int mosaic_rows = 1, mosaic_cols = 1;
		for(size_t pi=0; pi<KeyPoint_x1.size(); ++pi){
			mosaic_rows = std::max(mosaic_rows, KeyPoint_x1[pi]+1);
			mosaic_cols = std::max(mosaic_cols, KeyPoint_y1[pi]+1);
			mosaic_rows = std::max(mosaic_rows, KeyPoint_x2[pi]+1);
			mosaic_cols = std::max(mosaic_cols, KeyPoint_y2[pi]+1);
		}
		if(use_local && !KeyPoint_x1.empty()){
			const float local_sigma = match_cfg.local_affine_sigma;
			int tiles_r = match_cfg.local_tiles_r;
			int tiles_c = match_cfg.local_tiles_c;
			IMtemp.build_local_affine_field(
				KeyPoint_x1, KeyPoint_y1, KeyPoint_x2, KeyPoint_y2,
				mosaic_rows, mosaic_cols,
				tiles_r, tiles_c,
				local_sigma, match_cfg.ransac_iterations, match_cfg.local_min_points,
				fs_c, layer_field);

			std::cout<<"Local mark/build sigma: local_ransac="<<local_sigma
				<<" inlier="<<local_sigma<<" tiles="<<tiles_r<<"x"<<tiles_c
				<<" layer="<<i<<std::endl;
			IMtemp.mark_local_inliers(KeyPoint_x1, KeyPoint_y1, KeyPoint_x2, KeyPoint_y2,
				layer_field, local_sigma, match.data());
		}

		char imgL_path1[2048];
		snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
		char imgR_path1[2048];
		snprintf(imgR_path1, sizeof(imgR_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID2);
		char outpath[64];
		snprintf(outpath, sizeof(outpath), "../out/fenfu_match%d.tif", i);

		// 中间层：用当前 Match + 未匹配黄点绘制（与 Python --no-draw-lines 同风格）
		// 第 0 层等 *_match.txt 写出（及 densify）后再按文件绘制，与 Python 完全一致
		if(BJ != 1){
			std::vector<int> un_x1, un_y1, un_x2, un_y2;
			const double sfr_draw = pow(double(2), double(i)) / pow(double(2), double(4));
			for(int j=ccd_begin;j<ccd_end;j++){
				std::string featurepoint1 = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/" + std::to_string(i) + "/" + xulie_ID1 + "_RED" + std::to_string(j) + ".txt";
				FILE *fp=fopen(featurepoint1.c_str(),"r");
				if(fp){
					int bj,row,col,imgID,mrow,mcol; float m_score;
					while(fscanf(fp,"%d ",&bj) == 1){
						if(bj==1){
							if(fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score)!=6) break;
						}else{
							if(fscanf(fp,"%d %d\n",&row,&col)!=2) break;
							un_x1.push_back(int(double(row+mosaic_c1[j*4+0])*sfr_draw));
							un_y1.push_back(int(double(col+mosaic_c1[j*4+2])*sfr_draw));
						}
					}
					fclose(fp);
				}
				std::string featurepoint2 = std::string(filepath1) + "/" + xulie_ID2 + "/downsample/" + std::to_string(i) + "/" + xulie_ID2 + "_RED" + std::to_string(j) + ".txt";
				fp=fopen(featurepoint2.c_str(),"r");
				if(fp){
					int bj,row,col,imgID,mrow,mcol; float m_score;
					while(fscanf(fp,"%d ",&bj) == 1){
						if(bj==1){
							if(fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score)!=6) break;
						}else{
							if(fscanf(fp,"%d %d\n",&row,&col)!=2) break;
							un_x2.push_back(int(double(row+mosaic_c2[j*4+0])*sfr_draw));
							un_y2.push_back(int(double(col+mosaic_c2[j*4+2])*sfr_draw));
						}
					}
					fclose(fp);
				}
			}
			std::cout<<"开始绘制匹配结果在mosaic img（中间层，无连线）……"<<std::endl;
			IMtemp.drawMatch3(imgL_path1,imgR_path1,outpath,
				KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match.data(),1,
				un_x1,un_y1,un_x2,un_y2,false);
		}

		// 准备下一层引导场（升采样）
		if(use_local && i > 0 && layer_field.n_tr > 0){
			guide_field = layer_field;
			guide_field.upsample(static_cast<float>(kc));
			guide_ready = true;
			std::cout<<"LocalAffineField upsampled x"<<kc<<" for next layer"<<std::endl;
		} else if(use_local && i == 0){
			guide_field = layer_field;
			guide_ready = (guide_field.n_tr > 0);
		}

		vector<int>().swap(KeyPoint_x1);
		vector<int>().swap(KeyPoint_y1);
		vector<int>().swap(KeyPoint_x2);
		vector<int>().swap(KeyPoint_y2);
		vector<int>().swap(KeyPoint_x11);
		vector<int>().swap(KeyPoint_y11);
		vector<int>().swap(KeyPoint_x22);
		vector<int>().swap(KeyPoint_y22);

		//存储结果并跳出
		if(BJ==1){
			for(int j=ccd_begin;j<ccd_end;j++){
				std::string featurepoint1 = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/" + std::to_string(i) + "/" + xulie_ID1 + "_RED" + std::to_string(j) + ".txt";
				std::string featurepoint2 = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/" + std::to_string(i) + "/" + xulie_ID1 + "_RED" + std::to_string(j) + "_match.txt";
				FILE *fp=fopen(featurepoint1.c_str(),"r");
				if(fp==NULL){
					std::cout<<"无法打开特征点文件："<<featurepoint1<<std::endl;
					continue;
				}
				FILE *fp_re=fopen(featurepoint2.c_str(),"w");
				int bj,row,col,imgID,mrow,mcol;
				float m_score;
				while(fscanf(fp,"%d ",&bj) == 1){
					if(bj==1){
						if(fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score) != 6){
							break;
						}
						const float mr1 = static_cast<float>(row + mosaic_c1[j*4+0]);
						const float mc1 = static_cast<float>(col + mosaic_c1[j*4+2]);
						const float mr2 = static_cast<float>(mrow + mosaic_c2[imgID*4+0]);
						const float mc2 = static_cast<float>(mcol + mosaic_c2[imgID*4+2]);
						bool keep = false;
						if(use_local && guide_ready){
							float pr=0.f, pc=0.f;
							guide_field.predict(mr1, mc1, &pr, &pc);
							const float vx = pr - mr2;
							const float vy = pc - mc2;
							const float resid = std::sqrt(vx*vx+vy*vy);
							const float local_th = match_cfg.local_affine_sigma;
							// 高相关救援：坡面正确匹配常偏离粗块仿射，但 NCC 仍高
							keep = (resid < local_th) ||
								(m_score >= match_cfg.local_keep_score_min && resid < local_th * 2.f);
						} else {
							float vx=mr1*fs_c[0]+mc1*fs_c[1]+fs_c[2]-mr2;
							float vy=mr1*fs_c[3]+mc1*fs_c[4]+fs_c[5]-mc2;
							const float g_rad = match_cfg.global_keep_radius;
							const float g_abs = match_cfg.global_keep_abs;
							keep = (sqrt(vx*vx+vy*vy)<g_rad && abs(vx)<g_abs && abs(vy)<g_abs);
						}
						if(keep){
							fprintf(fp_re,"%d %d %d %d %d %d %f\n",bj,row,col,imgID,mrow,mcol,m_score);
						}
					}
					else{
						if(fscanf(fp,"%d %d\n",&row,&col) != 2){
							break;
						}
					}
				}
				fclose(fp);
				fclose(fp_re);
			}

			// 密格网成网：按左 CCD × 右 CCD 种子局部视差引导
			if(do_densify){
				std::cout<<"开始密格网 densify（参考 intra_CCD_match_B）……"<<std::endl;
				for(int j=ccd_begin;j<ccd_end;j++){
					std::string seed_path = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/0/" + xulie_ID1 + "_RED" + std::to_string(j) + "_match.txt";
					std::string dense_path = std::string(filepath1) + "/" + xulie_ID1 + "/downsample/0/" + xulie_ID1 + "_RED" + std::to_string(j) + "_dense.txt";
					remove(dense_path.c_str());
					char imgL_path[2048];
					snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j);

					std::vector<int> right_counts(CCD_num, 0);
					FILE* fps = fopen(seed_path.c_str(), "r");
					if(fps){
						int bj,imgID,row,col,mrow,mcol; float ms;
						while(fscanf(fps,"%d %d %d %d %d %d %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&ms)==7){
							if(bj==1 && imgID>=ccd_begin && imgID<ccd_end) right_counts[imgID]++;
						}
						fclose(fps);
					}
					int n_dense_total = 0;
					for(int k=ccd_begin;k<ccd_end;k++){
						if(right_counts[k] < match_cfg.local_min_points) continue;
						char imgR_path[2048];
						snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, k);
						n_dense_total += IMtemp.densify_match_from_seeds(
							imgL_path, imgR_path,
							match_cfg.densify_window_size,
							match_cfg.densify_search_range,
							match_cfg.densify_cc_threshold,
							match_cfg.densify_batch_r,
							match_cfg.densify_batch_c,
							match_cfg.densify_knn,
							mutable_cstr(seed_path), k, mutable_cstr(dense_path));
					}
					// 合并 densify 结果到 match 种子文件，供后续 grid_match 使用
					if(n_dense_total > 0){
						FILE* fpd = fopen(dense_path.c_str(), "r");
						FILE* fpm = fopen(seed_path.c_str(), "a");
						if(fpd && fpm){
							char line[2048];
							while(fgets(line, sizeof(line), fpd)){
								fputs(line, fpm);
							}
						}
						if(fpd) fclose(fpd);
						if(fpm) fclose(fpm);
						std::cout<<"CCD "<<j<<" densify merged "<<n_dense_total<<" points into match.txt"<<std::endl;
					}
				}
			}

			// 与 script/draw_fenfu_match0_inliers.py --no-draw-lines 一致（keep + densify 之后）
				{
					char imgL_path1[2048];
					snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
					char imgR_path1[2048];
					snprintf(imgR_path1, sizeof(imgR_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID2);
					char outpath[64];
					snprintf(outpath, sizeof(outpath), "../out/fenfu_match%d.tif", i);
					std::cout<<"开始按文件绘制 fenfu_match（黄=未匹配/红=错误/绿=正确，无连线）……"<<std::endl;
					IMtemp.drawFenfuMatchFromFiles(imgL_path1, imgR_path1, outpath,
						filepath1, xulie_ID1, xulie_ID2, i,
						ccd_begin, ccd_end, CCD_num,
						mosaic_c1.data(), mosaic_c2.data(), false);
				}

			break;
		}
	}
}

void Observation::ImageAllMatch(char* imagepath1,char* imagepath2){
	float threshold=0.5;
	int Localmax_win=32;
	int w_size=11;

	//提取特征点
	//左影像
	std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
	ImageMatch IM;
	IM.Feature_Detection(imagepath1,6,Localmax_win,KeyPoint_x1,KeyPoint_y1);

	//右影像
	std::vector<int> KeyPoint_x2,KeyPoint_y2;  //x为行，y为列
	IM.Feature_Detection(imagepath2,6,Localmax_win,KeyPoint_x2,KeyPoint_y2);


	//匹配
	Mat img_1 = imread(imagepath1, 0);
	Mat img_2 = imread(imagepath2, 0);
	if((!img_1.data)||(!img_2.data))
	{
		std::cout<< " --(!) Error reading images " << std::endl;
		return;
	}
	int rows1=img_1.rows;
	int cols1=img_1.cols;
	int rows2=img_2.rows;
	int cols2=img_2.cols;


	float CC=0;
	float N=w_size*w_size;
	float s12,s11,s22,s1,s2;
	int r1,r2,c1,c2;
	std::vector<int> matched(KeyPoint_x1.size(), -1);
	std::vector<float> C_match(KeyPoint_x1.size(), 0.f);
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
	for(int i=0;i<KeyPoint_x1.size();i++){
		if(matched[i]!=-1){
			KeyPoint_x11.push_back(KeyPoint_x1[i]);
			KeyPoint_y11.push_back(KeyPoint_y1[i]);
			KeyPoint_x22.push_back(KeyPoint_x2[i]);
			KeyPoint_y22.push_back(KeyPoint_y2[i]);
		}
	}

	//仿射系数提取
	float fs_c[6];
	std::fill(fs_c, fs_c + 6, 1.0f);
	ImageMatch IMtemp;
	std::vector<int> match(KeyPoint_x11.size(), -1);
	IMtemp.RANSAC_fs2(KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,200,500,match.data(),fs_c);  //剔除粗差

	char match_outpath[] = "../out/match.tif";
	IMtemp.drawMatch3(imagepath1,imagepath2,match_outpath,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match.data(),1);


	FILE *fp_re=fopen("../out/matches.txt","w");
	for(int i=0;i<KeyPoint_x11.size();i++){
		if(match[i]!=-1){
			fprintf(fp_re,"%d %d %d %d\n",KeyPoint_x11[i],KeyPoint_y11[i],KeyPoint_x22[i],KeyPoint_y22[i]);
		}
	}
	fclose(fp_re);


	std::vector<int>().swap(KeyPoint_x1);
	std::vector<int>().swap(KeyPoint_y1);
	std::vector<int>().swap(KeyPoint_x2);
	std::vector<int>().swap(KeyPoint_y2);
	std::vector<int>().swap(KeyPoint_x11);
	std::vector<int>().swap(KeyPoint_y11);
	std::vector<int>().swap(KeyPoint_x22);
	std::vector<int>().swap(KeyPoint_y22);
}



//计算仿射系数
void Observation::Compute_fsc(float* fs_c,int mark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	printf("Compute_fsc begin!\n");
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	char* filepath1=filepath;

	ImageMatch IM;
	Observation OB;

	//逐层匹配
	int CCD_id;
	int BJ=0;
	int count1=0;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		std::cout<<"读取拼接系数……"<<std::endl;
		char ds_path_mosaictxt1[2048];
		snprintf(ds_path_mosaictxt1, sizeof(ds_path_mosaictxt1), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID1, i);
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		if(fpm1==NULL){
			std::cout<<"未找到拼接系数文件："<<ds_path_mosaictxt1<<std::endl;
			return;
		}
		std::vector<int> mosaic_c1(CCD_num * 4);
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]) != 5){
				std::cout<<"拼接系数文件读取失败："<<ds_path_mosaictxt1<<std::endl;
				fclose(fpm1);
				return;
			}
		}
		fclose(fpm1);
		char ds_path_mosaictxt2[2048];
		snprintf(ds_path_mosaictxt2, sizeof(ds_path_mosaictxt2), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID2, i);
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		if(fpm2==NULL){
			std::cout<<"未找到拼接系数文件："<<ds_path_mosaictxt2<<std::endl;
			return;
		}
		std::vector<int> mosaic_c2(CCD_num * 4);
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]) != 5){
				std::cout<<"拼接系数文件读取失败："<<ds_path_mosaictxt2<<std::endl;
				fclose(fpm2);
				return;
			}
		}
		fclose(fpm2);

		std::cout<<"开始匹配第"<<i<<"层……"<<std::endl;
		std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
		std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
		//航带间（仅 [ccd_begin, ccd_end)）
		int count0=0;
		const int ccd_begin = cfg_.ccd_begin();
		const int ccd_end = cfg_.ccd_end();
		for(int j=ccd_begin;j<ccd_end;j++){
			char featurepoint1[2048];
			if(mark==0){
				snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			}
			else{
				snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_grid.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			}
			std::cout<<"读取特征点："<<featurepoint1<<std::endl;

			FILE *fp=fopen(featurepoint1,"r");
			if(fp==NULL){
				std::cout<<"未找到特征点文件："<<featurepoint1<<std::endl;
				continue;
			}
			int bj,imgID;
			float row,col,mrow,mcol;
			float m_score;
			double sfr=pow(double(2),double(i))/pow(double(2),double(4));
			while(fscanf(fp,"%d ",&bj) == 1){
				if(bj==1 && count0%1==0){
					if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score) != 6){
						break;
					}
					KeyPoint_x1.push_back(int(row+mosaic_c1[j*4+0]));
					KeyPoint_y1.push_back(int(col+mosaic_c1[j*4+2]));
					KeyPoint_x2.push_back(int(mrow+mosaic_c2[imgID*4+0]));
					KeyPoint_y2.push_back(int(mcol+mosaic_c2[imgID*4+2]));
					KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
					KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
					KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[imgID*4+0])*sfr));
					KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[imgID*4+2])*sfr));
					count1++;
				}
				else{
					if(fscanf(fp,"%f %f\n",&row,&col) != 2){
						break;
					}
				}
				count0++;
			}
			fclose(fp);
		}

		std::cout<<"航带间特征点提取完成，数量："<<KeyPoint_x1.size()<<std::endl;

		//CCD间（仅范围内相邻对）
		std::cout<<"开始CCD间特征点提取……"<<std::endl;
		for(int j=ccd_begin;j<ccd_end-1;j++){
			char featurepoint1[2048];
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_%d_intra__.txt", filepath1, xulie_ID1, i, xulie_ID1, j, j+1);

			FILE *fp=fopen(featurepoint1,"r");
			if(fp==NULL){
				std::cout<<"未找到特征点文件："<<featurepoint1<<std::endl;
				continue;
			}

			int bj,imgID;
			float row,col,mrow,mcol;
			float m_score;
			double sfr=pow(double(2),double(i))/pow(double(2),double(4));
			while(fscanf(fp,"%d ",&bj) == 1){
				if(bj==1 && count0%1==0){
					if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score) != 6){
						break;
					}
					KeyPoint_x1.push_back(int(row+mosaic_c1[j*4+0]));
					KeyPoint_y1.push_back(int(col+mosaic_c1[j*4+2]));
					KeyPoint_x2.push_back(int(mrow+mosaic_c2[imgID*4+0]));
					KeyPoint_y2.push_back(int(mcol+mosaic_c2[imgID*4+2]));
					KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
					KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
					KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[imgID*4+0])*sfr));
					KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[imgID*4+2])*sfr));
					count0++;
				}
				else if(bj==1){
					if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score) != 6){
						break;
					}
					count0++;
				}
				else{
					if(fscanf(fp,"%f %f\n",&row,&col) != 2){
						break;
					}
				}
			}
			fclose(fp);
		}
		std::cout<<"CCD间特征点提取完成，数量："<<KeyPoint_x1.size()<<std::endl;//*/

		std::cout<<"开始RANSAC匹配……"<<std::endl;
		std::vector<int> match(KeyPoint_x1.size(), -1);
		// IM.RANSAC_fs2(KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,100,1000,match,fs_c);  //剔除粗差
		IM.RANSAC_fs2(KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,150,100000,match.data(),fs_c);
		std::cout<<"RANSAC匹配完成，数量："<<KeyPoint_x1.size()<<std::endl;

		std::vector<int>().swap(KeyPoint_x1);
		std::vector<int>().swap(KeyPoint_y1);
		std::vector<int>().swap(KeyPoint_x2);
		std::vector<int>().swap(KeyPoint_y2);
		std::vector<int>().swap(KeyPoint_x11);
		std::vector<int>().swap(KeyPoint_y11);
		std::vector<int>().swap(KeyPoint_x22);
		std::vector<int>().swap(KeyPoint_y22);
	}
}
//计算互信息
void Observation::Compute_MI(float* MI_table){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int CCD_num=cfg_.CCD_num;
	int rows1=cfg_.rows1;int cols1=cfg_.cols;
	int rows2=cfg_.rows2;int cols2=cfg_.cols;
	printf("Compute_MI begin!\n");
	//生成互信息查找表
	std::vector<float> P12(256 * 256, 0.0f);
	std::vector<float> P1(256, 0.0f);
	std::vector<float> P2(256, 0.0f);
	int sum_mp=0;
	int bj,row,col,imgID,mrow,mcol;
	float mscore;
	for(int j=0;j<CCD_num;j++){
		std::vector<uchar> data1(rows1 * cols1, 0);
		std::vector<uchar> data2(rows2 * cols2, 0);

		//读取对应影像
		char imgL_path[2048];
		snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j);
		tif_load(imgL_path,data1.data());
		char imgR_path[2048];
		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, j);
		tif_load(imgR_path,data2.data());

		char matchpointfile[2048];
		snprintf(matchpointfile, sizeof(matchpointfile), "%s/%s/downsample/0/%s_RED%d_match.txt", filepath, xulie_ID1, xulie_ID1, j);
		//读取控制点位置
		std::vector<int> matchpoint;
		FILE *fp1=fopen(matchpointfile,"r");
		if(fp1==NULL){
			std::cout<<"未找到控制点文件："<<matchpointfile<<std::endl;
			continue;
		}
		while(fscanf(fp1,"%d ",&bj) == 1){
			if(bj==1){
				if(fscanf(fp1,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore) != 6){
					break;
				}
				if(imgID==j){
					matchpoint.push_back(row);matchpoint.push_back(col);
					matchpoint.push_back(imgID);matchpoint.push_back(mrow);matchpoint.push_back(mcol);
				}
			}
			else{
				if(fscanf(fp1,"%d %d\n",&row,&col) != 2){
					break;
				}
			}
		}
		fclose(fp1);

		for(int i=0;i<matchpoint.size()/5;i++){
			int grey1=data1[matchpoint[5*i]*cols1+matchpoint[5*i+1]];
			int grey2=data2[matchpoint[5*i+3]*cols2+matchpoint[5*i+4]];
			P12[grey1*256+grey2] += 1.0;
			P1[grey1] += 1.0;
			P2[grey2] += 1.0;
			sum_mp++;
		}

		vector<int>().swap(matchpoint);
	}

	//计算互信息
	for(int i=0;i<256;i++){
		for(int j=0;j<256;j++){
			/*float P12_,P1_,P2_;
			P12_=P1_=P2_=0;
			for(int ii=-2;ii<=2;ii++){
				P1_ += P1[i+ii]/5;
				P2_ += P2[j+ii]/5;
				for(int jj=-2;jj<=2;jj++){
					P12_ += P12[(i+ii)*256+(j+jj)]/25;
				}
			}*/
			MI_table[i*256+j]=(log(P12[i*256+j]/sum_mp+0.000001)-log(P1[i]/sum_mp+0.000001)-log(P2[j]/sum_mp+0.000001))/log(2.0);
			//MI_table[i*256+j]=(log(P12_+0.000001)-log(P1_+0.000001)-log(P2_+0.000001))/log(2.0);
		}
	}
}
//剔除粗差
void Observation::Tichu_CX(int mark, float* fs_c,int LRmark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	printf("Tichu_CX begin! mark=%d LRmark=%d\n", mark, LRmark);
	const PipelineConfig::BaParams& ba = cfg_.ba;
	const float tichu_rmax = ba.tichu_affine_max_residual;
	const float tichu_vx = ba.tichu_affine_max_vx;
	const float tichu_vy = ba.tichu_affine_max_vy;
	const float tichu_radius = ba.tichu_local_radius;
	const float tichu_nsigma = ba.tichu_sigma_factor;
	if(mark == 0 || mark == 1 || mark == 4){
		printf("[Tichu_CX] thresholds: residual<%.0f |vx|<%.0f |vy|<%.0f local_r=%.0f sigma×%.1f\n",
			tichu_rmax, tichu_vx, tichu_vy, tichu_radius, tichu_nsigma);
		fflush(stdout);
	}
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	char* filepath1=filepath;

	//逐层匹配
	int CCD_id;
	int BJ=0;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char ds_path_mosaictxt1[2048];
		snprintf(ds_path_mosaictxt1, sizeof(ds_path_mosaictxt1), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID1, i);
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		if(fpm1==NULL){
			std::cout<<"未找到拼接系数文件："<<ds_path_mosaictxt1<<std::endl;
			return;
		}
		std::vector<int> mosaic_c1(CCD_num * 4);
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]) != 5){
				std::cout<<"拼接系数文件读取失败："<<ds_path_mosaictxt1<<std::endl;
				fclose(fpm1);
				return;
			}
		}
		fclose(fpm1);
		char ds_path_mosaictxt2[2048];
		snprintf(ds_path_mosaictxt2, sizeof(ds_path_mosaictxt2), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID2, i);
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		if(fpm2==NULL){
			std::cout<<"未找到拼接系数文件："<<ds_path_mosaictxt2<<std::endl;
			return;
		}
		std::vector<int> mosaic_c2(CCD_num * 4);
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]) != 5){
				std::cout<<"拼接系数文件读取失败："<<ds_path_mosaictxt2<<std::endl;
				fclose(fpm2);
				return;
			}
		}
		fclose(fpm2);

		//存储结果并跳出
		if(BJ==1){
			int CX_count=0;
			int M_count=0;
			const int ccd_begin = cfg_.ccd_begin();
			const int ccd_end = cfg_.ccd_end();
			const int j_begin = ccd_begin;
			const int j_end = (mark==3 || mark==4) ? (ccd_end - 1) : ccd_end;
			for(int j=j_begin;j<j_end;j++){
				printf("%d\n",j);
				std::vector<float> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
				std::vector<float> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
				std::vector<int> matchimage_ID;
				std::vector<float> match_score;
				char featurepoint1[2048];
				if(mark==0){
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				}
				else if(mark==1){
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_grid.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				}
				else if(mark==2){
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_dc.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				}
				else if(mark==3){
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_%d_intra.txt", filepath1, xulie_ID1, i, xulie_ID1, j, j+1);
				}
				else if(mark==4){
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_%d_intra__.txt", filepath1, xulie_ID1, i, xulie_ID1, j, j+1);
				}
				else{
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				}
				char featurepoint2[2048];
				snprintf(featurepoint2, sizeof(featurepoint2), "%s/%s/downsample/%d/%s_RED%d_match.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				if(mark==3){
					snprintf(featurepoint2, sizeof(featurepoint2), "%s/%s/downsample/%d/%s_RED%d_%d_intra_match.txt", filepath1, xulie_ID1, i, xulie_ID1, j, j+1);
				}
				int bj,imgID;
				float row,col,mrow,mcol;
				float m_score;
				int count=0;
				if(mark!=3 && mark!=4){
					FILE *fp=fopen(featurepoint1,"r");
					if(fp==NULL){
						std::cout<<"未找到特征点文件："<<featurepoint1<<std::endl;
						continue;
					}
					FILE *fp_re=fopen(featurepoint2,"w");
					while(fscanf(fp,"%d ",&bj) == 1){
						if(bj==1){
							if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score) != 6){
								break;
							}
							KeyPoint_x1.push_back(row+mosaic_c1[j*4+0]);
							KeyPoint_y1.push_back(col+mosaic_c1[j*4+2]);
							KeyPoint_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
							KeyPoint_y2.push_back(mcol+mosaic_c2[imgID*4+2]);
							KeyPoint_x11.push_back(row);
							KeyPoint_y11.push_back(col);
							KeyPoint_x22.push_back(mrow);
							KeyPoint_y22.push_back(mcol);
							matchimage_ID.push_back(imgID);
							match_score.push_back(m_score);
						}
						else{
							if(fscanf(fp,"%f %f\n",&row,&col) != 2){
								break;
							}
						}
					}
					fclose(fp);

					for(int jj=0;jj<KeyPoint_x1.size();jj++){
						float vx=KeyPoint_x1[jj]*fs_c[0]+KeyPoint_y1[jj]*fs_c[1]+fs_c[2]-KeyPoint_x2[jj];
						float vy=KeyPoint_x1[jj]*fs_c[3]+KeyPoint_y1[jj]*fs_c[4]+fs_c[5]-KeyPoint_y2[jj];
						M_count++;

						if(sqrt(vx*vx+vy*vy)<tichu_rmax && abs(vx)<tichu_vx && abs(vy)<tichu_vy){
							std::vector<double> d_r,d_c;
							std::vector<double> weighttemp;
							double dr_mean,dr_std,dc_mean,dc_std;
							dr_mean=dr_std=dc_mean=dc_std=0;
							for(int ii=0;ii<KeyPoint_x1.size();ii++){
								double dist=sqrt(double(KeyPoint_x1[jj]-KeyPoint_x1[ii])*(KeyPoint_x1[jj]-KeyPoint_x1[ii])+(KeyPoint_y1[jj]-KeyPoint_y1[ii])*(KeyPoint_y1[jj]-KeyPoint_y1[ii]));
								if(dist<tichu_radius && ii!=jj){
									d_r.push_back(KeyPoint_x2[ii]-KeyPoint_x1[ii]);
									d_c.push_back(KeyPoint_y2[ii]-KeyPoint_y1[ii]);
									double wt=1.0;
									//weighttemp.push_back(exp(-1/2*dist));
									weighttemp.push_back(wt);
									dr_mean += (KeyPoint_x2[ii]-KeyPoint_x1[ii])*wt;
									dc_mean += (KeyPoint_y2[ii]-KeyPoint_y1[ii])*wt;
								}
							}

							//行向视差
							//dr_mean =  std::accumulate(std::begin(d_r), std::end(d_r), 0.0) / d_r.size();  //均值
							dr_mean /= std::accumulate(std::begin(weighttemp), std::end(weighttemp), 0.0);
							std::for_each (std::begin(d_r), std::end(d_r), [&](const double d) {
								dr_std  += (d-dr_mean)*(d-dr_mean);
							});
							dr_std = sqrt(dr_std/d_r.size());  //标准差

							//列向视差
							//dc_mean =  std::accumulate(std::begin(d_c), std::end(d_c), 0.0) / d_c.size();  //均值
							dc_mean /= std::accumulate(std::begin(weighttemp), std::end(weighttemp), 0.0);
							std::for_each (std::begin(d_c), std::end(d_c), [&](const double d) {
								dc_std  += (d-dc_mean)*(d-dc_mean);
							});
							dc_std = sqrt(dc_std/d_c.size());  //标准差

							// nσ 局部视差剔除
							if( abs((KeyPoint_x2[jj]-KeyPoint_x1[jj])-dr_mean)<tichu_nsigma*dr_std  &&  abs((KeyPoint_y2[jj]-KeyPoint_y1[jj])-dc_mean)<tichu_nsigma*dc_std )
							{
								fprintf(fp_re,"%d %f %f %d %f %f %f\n",1,KeyPoint_x11[jj],KeyPoint_y11[jj],matchimage_ID[jj],KeyPoint_x22[jj],KeyPoint_y22[jj],match_score[jj]);
							}
							else{
								CX_count++;
							}
						}
						else{
							CX_count++;
						}
					}
					fclose(fp_re);

					std::vector<float>().swap(KeyPoint_x1);
					std::vector<float>().swap(KeyPoint_y1);
					std::vector<float>().swap(KeyPoint_x2);
					std::vector<float>().swap(KeyPoint_y2);
					std::vector<float>().swap(KeyPoint_x11);
					std::vector<float>().swap(KeyPoint_y11);
					std::vector<float>().swap(KeyPoint_x22);
					std::vector<float>().swap(KeyPoint_y22);
					std::vector<int>().swap(matchimage_ID);
					std::vector<float>().swap(match_score);
				}
				else if(mark==3){
					if (LRmark==0){
						FILE *fp=fopen(featurepoint1,"r");
						if(fp==NULL){
							printf("未找到特征点文件：%s\n",featurepoint1);
							continue;
						}
						FILE *fp_re=fopen(featurepoint2,"w");
						while(fscanf(fp,"%d %f %f %d %f %f %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&m_score) == 7){
							KeyPoint_x1.push_back(row);
							KeyPoint_y1.push_back(col);
							KeyPoint_x2.push_back(mrow);
							KeyPoint_y2.push_back(mcol);
							matchimage_ID.push_back(imgID);
							match_score.push_back(m_score);
						}
						fclose(fp);

						for(int jj=0;jj<KeyPoint_x1.size();jj++){
							M_count++;

							std::vector<double> d_r,d_c;
							std::vector<double> weighttemp;
							double dr_mean,dr_std,dc_mean,dc_std;
							dr_mean=dr_std=dc_mean=dc_std=0;
							for(int ii=0;ii<KeyPoint_x1.size();ii++){
								double dist=sqrt(double(KeyPoint_x1[jj]-KeyPoint_x1[ii])*(KeyPoint_x1[jj]-KeyPoint_x1[ii])+(KeyPoint_y1[jj]-KeyPoint_y1[ii])*(KeyPoint_y1[jj]-KeyPoint_y1[ii]));
								if(dist<300 && ii!=jj){
									d_r.push_back(KeyPoint_x2[ii]-KeyPoint_x1[ii]);
									d_c.push_back(KeyPoint_y2[ii]-KeyPoint_y1[ii]);
									const double sigma_dist = 150.0;
									double wt=exp(-(dist*dist)/(2.0*sigma_dist*sigma_dist));
									//weighttemp.push_back(exp(-1/2*dist));
									weighttemp.push_back(wt);
									dr_mean += (KeyPoint_x2[ii]-KeyPoint_x1[ii])*wt;
									dc_mean += (KeyPoint_y2[ii]-KeyPoint_y1[ii])*wt;
								}
							}

							//行向视差
							//dr_mean =  std::accumulate(std::begin(d_r), std::end(d_r), 0.0) / d_r.size();  //均值
							dr_mean /= std::accumulate(std::begin(weighttemp), std::end(weighttemp), 0.0);
							std::for_each (std::begin(d_r), std::end(d_r), [&](const double d) {
								dr_std  += (d-dr_mean)*(d-dr_mean);
							});
							dr_std = sqrt(dr_std/d_r.size());  //标准差

							//列向视差
							//dc_mean =  std::accumulate(std::begin(d_c), std::end(d_c), 0.0) / d_c.size();  //均值
							dc_mean /= std::accumulate(std::begin(weighttemp), std::end(weighttemp), 0.0);
							std::for_each (std::begin(d_c), std::end(d_c), [&](const double d) {
								dc_std  += (d-dc_mean)*(d-dc_mean);
							});
							dc_std = sqrt(dc_std/d_c.size());  //标准差

							//3倍标准差剔除
							if( abs((KeyPoint_x2[jj]-KeyPoint_x1[jj])-dr_mean)<2*dr_std  &&  abs((KeyPoint_y2[jj]-KeyPoint_y1[jj])-dc_mean)<2*dc_std )
							{
								fprintf(fp_re,"%d %f %f %d %f %f %f\n",matchimage_ID[jj]-1,KeyPoint_x1[jj],KeyPoint_y1[jj],matchimage_ID[jj],KeyPoint_x2[jj],KeyPoint_y2[jj],match_score[jj]);
							}
							else{
								CX_count++;
							}
							vector<double>().swap(d_r);
							vector<double>().swap(d_c);
							vector<double>().swap(weighttemp);

						}
						fclose(fp_re);
						vector<float>().swap(KeyPoint_x1);
						vector<float>().swap(KeyPoint_y1);
						vector<float>().swap(KeyPoint_x2);
						vector<float>().swap(KeyPoint_y2);
						vector<float>().swap(KeyPoint_x11);
						vector<float>().swap(KeyPoint_y11);
						vector<float>().swap(KeyPoint_x22);
						vector<float>().swap(KeyPoint_y22);
						vector<int>().swap(matchimage_ID);
						vector<float>().swap(match_score);
					}
					else{
						//右影像
						snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_%d_intra.txt", filepath1, xulie_ID2, i, xulie_ID2, j, j+1);
						snprintf(featurepoint2, sizeof(featurepoint2), "%s/%s/downsample/%d/%s_RED%d_%d_intra_match.txt", filepath1, xulie_ID2, i, xulie_ID2, j, j+1);
						FILE * fp=fopen(featurepoint1,"r");
						if(fp==NULL){
							std::cout<<"未找到特征点文件："<<featurepoint1<<std::endl;
							continue;
						}
						FILE * fp_re=fopen(featurepoint2,"w");
						while(fscanf(fp,"%d %f %f %d %f %f %f\n",&bj,&row,&col,&imgID,&mrow,&mcol,&m_score) == 7){
							KeyPoint_x1.push_back(row);
							KeyPoint_y1.push_back(col);
							KeyPoint_x2.push_back(mrow);
							KeyPoint_y2.push_back(mcol);
							matchimage_ID.push_back(imgID);
							match_score.push_back(m_score);
						}
						fclose(fp);

						for(int jj=0;jj<KeyPoint_x1.size();jj++){
							M_count++;

							std::vector<double> d_r,d_c;
							std::vector<double> weighttemp;
							double dr_mean,dr_std,dc_mean,dc_std;
							dr_mean=dr_std=dc_mean=dc_std=0;
							for(int ii=0;ii<KeyPoint_x1.size();ii++){
								double dist=sqrt(double(KeyPoint_x1[jj]-KeyPoint_x1[ii])*(KeyPoint_x1[jj]-KeyPoint_x1[ii])+(KeyPoint_y1[jj]-KeyPoint_y1[ii])*(KeyPoint_y1[jj]-KeyPoint_y1[ii]));
								if(dist<300 && ii!=jj){
									d_r.push_back(KeyPoint_x2[ii]-KeyPoint_x1[ii]);
									d_c.push_back(KeyPoint_y2[ii]-KeyPoint_y1[ii]);
									double wt=1.0;
									//weighttemp.push_back(exp(-1/2*dist));
									weighttemp.push_back(wt);
									dr_mean += (KeyPoint_x2[ii]-KeyPoint_x1[ii])*wt;
									dc_mean += (KeyPoint_y2[ii]-KeyPoint_y1[ii])*wt;
								}
							}

							//行向视差
							//dr_mean =  std::accumulate(std::begin(d_r), std::end(d_r), 0.0) / d_r.size();  //均值
							dr_mean /= std::accumulate(std::begin(weighttemp), std::end(weighttemp), 0.0);
							std::for_each (std::begin(d_r), std::end(d_r), [&](const double d) {
								dr_std  += (d-dr_mean)*(d-dr_mean);
							});
							dr_std = sqrt(dr_std/d_r.size());  //标准差

							//列向视差
							//dc_mean =  std::accumulate(std::begin(d_c), std::end(d_c), 0.0) / d_c.size();  //均值
							dc_mean /= std::accumulate(std::begin(weighttemp), std::end(weighttemp), 0.0);
							std::for_each (std::begin(d_c), std::end(d_c), [&](const double d) {
								dc_std  += (d-dc_mean)*(d-dc_mean);
							});
							dc_std = sqrt(dc_std/d_c.size());  //标准差

							//3倍标准差剔除
							if( abs((KeyPoint_x2[jj]-KeyPoint_x1[jj])-dr_mean)<1*dr_std  &&  abs((KeyPoint_y2[jj]-KeyPoint_y1[jj])-dc_mean)<1*dc_std )
							{
								fprintf(fp_re,"%d %f %f %d %f %f %f\n",matchimage_ID[jj]-1,KeyPoint_x1[jj],KeyPoint_y1[jj],matchimage_ID[jj],KeyPoint_x2[jj],KeyPoint_y2[jj],match_score[jj]);
							}
							else{
								CX_count++;
							}
							vector<double>().swap(d_r);
							vector<double>().swap(d_c);
							vector<double>().swap(weighttemp);

						}
						fclose(fp_re);
						std::vector<float>().swap(KeyPoint_x1);
						std::vector<float>().swap(KeyPoint_y1);
						std::vector<float>().swap(KeyPoint_x2);
						std::vector<float>().swap(KeyPoint_y2);
						std::vector<float>().swap(KeyPoint_x11);
						std::vector<float>().swap(KeyPoint_y11);
						std::vector<float>().swap(KeyPoint_x22);
						std::vector<float>().swap(KeyPoint_y22);
						std::vector<int>().swap(matchimage_ID);
						std::vector<float>().swap(match_score);
					}
				}
				else{
					// mark==4: 对_intra__.txt进行粗差剔除，保持原始顺序
					char* id_L = (LRmark==0) ? xulie_ID1 : xulie_ID2;
					int* mosaic_cL = (LRmark==0) ? mosaic_c1.data() : mosaic_c2.data();
					int* mosaic_cR = (LRmark==0) ? mosaic_c2.data() : mosaic_c1.data();
					char featurepoint1[2048];
					char featurepoint_tmp[2048];
					char controlpoint_full[2048];
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_%d_intra__.txt", filepath1, id_L, i, id_L, j, j+1);
					snprintf(featurepoint_tmp, sizeof(featurepoint_tmp), "%s/%s/downsample/%d/%s_RED%d_%d_intra___tmp.txt", filepath1, id_L, i, id_L, j, j+1);
					snprintf(controlpoint_full, sizeof(controlpoint_full), "%s/%s/downsample/%d/%s_RED%d_match.txt", filepath1, id_L, i, id_L, j);
					FILE *fp=fopen(featurepoint1,"r");
					if(fp==NULL){
						printf("未找到特征点文件：%s\n",featurepoint1);
						continue;
					}
					// file_order_bj: 记录原文件每行的 bj，file_order_idx: 对应到 KP* 或 UP* 中的索引
					std::vector<int> file_order_bj,file_order_idx;
					std::vector<float> KPx1,KPy1,KPx2,KPy2;
					std::vector<float> KPx11,KPy11,KPx22,KPy22;
					std::vector<int> KP_imgID;
					std::vector<float> KP_score;
					std::vector<float> UP_x,UP_y;
					std::vector<float> CtrlPx1,CtrlPy1,CtrlPx2,CtrlPy2;
					int bj_t; float row_t,col_t,mrow_t,mcol_t,mscore_t; int imgID_t;
					while(fscanf(fp,"%d ",&bj_t)==1){
						if(bj_t==1){
							if(fscanf(fp,"%f %f %d %f %f %f\n",&row_t,&col_t,&imgID_t,&mrow_t,&mcol_t,&mscore_t)!=6) break;
							float gx1=row_t+mosaic_cL[j*4+0];
							float gy1=col_t+mosaic_cL[j*4+2];
							float gx2=mrow_t+mosaic_cR[imgID_t*4+0];
							float gy2=mcol_t+mosaic_cR[imgID_t*4+2];
							file_order_bj.push_back(1);
							file_order_idx.push_back((int)KPx1.size());
							KPx1.push_back(gx1); KPy1.push_back(gy1);
							KPx2.push_back(gx2); KPy2.push_back(gy2);
							KPx11.push_back(row_t); KPy11.push_back(col_t);
							KPx22.push_back(mrow_t); KPy22.push_back(mcol_t);
							KP_imgID.push_back(imgID_t); KP_score.push_back(mscore_t);
						}
						else{
							if(fscanf(fp,"%f %f\n",&row_t,&col_t)!=2) break;
							file_order_bj.push_back(0);
							file_order_idx.push_back((int)UP_x.size());
							UP_x.push_back(row_t); UP_y.push_back(col_t);
						}
					}
					fclose(fp);

					// 读取 i_match_ 控制点，并从 i_match.txt 恢复完整左右像坐标，作为局部视差统计的控制样本
					if(LRmark==0){
						FILE *fp_ctrl_full=fopen(controlpoint_full,"r");
						if(fp_ctrl_full!=NULL){
							int ctrl_bj,ctrl_imgID;
							float ctrl_row,ctrl_col,ctrl_mrow,ctrl_mcol,ctrl_score;
							while(true){
								if(fscanf(fp_ctrl_full,"%d ",&ctrl_bj)!=1) break;
								if(ctrl_bj==1){
									if(fscanf(fp_ctrl_full,"%f %f %d %f %f %f\n",&ctrl_row,&ctrl_col,&ctrl_imgID,&ctrl_mrow,&ctrl_mcol,&ctrl_score)!=6) break;
									CtrlPx1.push_back(ctrl_row+mosaic_cL[j*4+0]);
									CtrlPy1.push_back(ctrl_col+mosaic_cL[j*4+2]);
									CtrlPx2.push_back(ctrl_mrow+mosaic_cR[ctrl_imgID*4+0]);
									CtrlPy2.push_back(ctrl_mcol+mosaic_cR[ctrl_imgID*4+2]);
								}
								else{
									if(fscanf(fp_ctrl_full,"%f %f\n",&ctrl_row,&ctrl_col)!=2) break;
								}
							}
							fclose(fp_ctrl_full);
						}
					}
					else{
						// xulie_ID2 侧利用 xulie_ID1 的跨航带匹配控制点，反向构造成当前方向的控制样本
							char controlpoint_full_rev[2048];
						for(int ctrl_ccd=0;ctrl_ccd<CCD_num;ctrl_ccd++){
							snprintf(controlpoint_full_rev, sizeof(controlpoint_full_rev), "%s/%s/downsample/%d/%s_RED%d_match.txt", filepath1, xulie_ID1, i, xulie_ID1, ctrl_ccd);
							FILE *fp_ctrl_full=fopen(controlpoint_full_rev,"r");
							if(fp_ctrl_full==NULL){
								continue;
							}
							int ctrl_bj,ctrl_imgID;
							float ctrl_row,ctrl_col;
							float ctrl_mrow,ctrl_mcol,ctrl_score;
							while(true){
								if(fscanf(fp_ctrl_full,"%d ",&ctrl_bj)!=1) break;
								if(ctrl_bj==1){
									if(fscanf(fp_ctrl_full,"%f %f %d %f %f %f\n",&ctrl_row,&ctrl_col,&ctrl_imgID,&ctrl_mrow,&ctrl_mcol,&ctrl_score)!=6) break;
									if(ctrl_imgID==j){
										CtrlPx1.push_back(ctrl_mrow+mosaic_cL[j*4+0]);
										CtrlPy1.push_back(ctrl_mcol+mosaic_cL[j*4+2]);
										CtrlPx2.push_back(ctrl_row+mosaic_cR[ctrl_ccd*4+0]);
										CtrlPy2.push_back(ctrl_col+mosaic_cR[ctrl_ccd*4+2]);
									}
								}
								else{
									if(fscanf(fp_ctrl_full,"%f %f\n",&ctrl_row,&ctrl_col)!=2) break;
								}
							}
							fclose(fp_ctrl_full);
						}
					}
					// 第一阶段：计算每个 bj==1 点是否通过过滤
					std::vector<bool> pass_filter(KPx1.size(),true);
					for(int jj=0;jj<(int)KPx1.size();jj++){
						M_count++;
						float vx,vy;
						if(LRmark==0){
							vx=KPx1[jj]*fs_c[0]+KPy1[jj]*fs_c[1]+fs_c[2]-KPx2[jj];
							vy=KPx1[jj]*fs_c[3]+KPy1[jj]*fs_c[4]+fs_c[5]-KPy2[jj];
						}
						else{
							vx=KPx2[jj]*fs_c[0]+KPy2[jj]*fs_c[1]+fs_c[2]-KPx1[jj];
							vy=KPx2[jj]*fs_c[3]+KPy2[jj]*fs_c[4]+fs_c[5]-KPy1[jj];
						}
						if(sqrt(vx*vx+vy*vy)<tichu_rmax && abs(vx)<tichu_vx && abs(vy)<tichu_vy){
							std::vector<double> d_r,d_c;
							std::vector<double> weighttemp;
							double dr_mean,dr_std,dc_mean,dc_std;
							dr_mean=dr_std=dc_mean=dc_std=0;
							for(int ii=0;ii<(int)KPx1.size();ii++){
								double dist=sqrt(double(KPx1[jj]-KPx1[ii])*(KPx1[jj]-KPx1[ii])+(KPy1[jj]-KPy1[ii])*(KPy1[jj]-KPy1[ii]));
								if(dist<tichu_radius && ii!=jj){
									d_r.push_back(KPx2[ii]-KPx1[ii]);
									d_c.push_back(KPy2[ii]-KPy1[ii]);
									weighttemp.push_back(1.0);
									dr_mean+=(KPx2[ii]-KPx1[ii]);
									dc_mean+=(KPy2[ii]-KPy1[ii]);
								}
							}
							for(int ii=0;ii<(int)CtrlPx1.size();ii++){
								double dist=sqrt(double(CtrlPx1[ii]-KPx1[jj])*(CtrlPx1[ii]-KPx1[jj])+(CtrlPy1[ii]-KPy1[jj])*(CtrlPy1[ii]-KPy1[jj]));
								if(dist<tichu_radius){
									d_r.push_back(CtrlPx2[ii]-CtrlPx1[ii]);
									d_c.push_back(CtrlPy2[ii]-CtrlPy1[ii]);
									weighttemp.push_back(1.0);
									dr_mean+=(CtrlPx2[ii]-CtrlPx1[ii]);
									dc_mean+=(CtrlPy2[ii]-CtrlPy1[ii]);
								}
							}
							if(!d_r.empty()){
								dr_mean/=std::accumulate(std::begin(weighttemp),std::end(weighttemp),0.0);
								dc_mean/=std::accumulate(std::begin(weighttemp),std::end(weighttemp),0.0);
								std::for_each(std::begin(d_r),std::end(d_r),[&](const double d){dr_std+=(d-dr_mean)*(d-dr_mean);});
								std::for_each(std::begin(d_c),std::end(d_c),[&](const double d){dc_std+=(d-dc_mean)*(d-dc_mean);});
								dr_std=sqrt(dr_std/d_r.size());
								dc_std=sqrt(dc_std/d_c.size());
								if(!(abs((KPx2[jj]-KPx1[jj])-dr_mean)<tichu_nsigma*dr_std && abs((KPy2[jj]-KPy1[jj])-dc_mean)<tichu_nsigma*dc_std)){
									pass_filter[jj]=false; CX_count++;
								}
							}
							// 无邻近点时保留该点（pass_filter 默认为 true）
							vector<double>().swap(d_r);
							vector<double>().swap(d_c);
							vector<double>().swap(weighttemp);
						}
						else{ pass_filter[jj]=false; CX_count++; }
					}
					// 第二阶段：按原始顺序写入
					FILE *fp_re=fopen(featurepoint_tmp,"w");
					for(int fi=0;fi<(int)file_order_bj.size();fi++){
						int idx=file_order_idx[fi];
						if(file_order_bj[fi]==1){
							if(pass_filter[idx]){
								fprintf(fp_re,"%d %f %f %d %f %f %f\n",1,KPx11[idx],KPy11[idx],KP_imgID[idx],KPx22[idx],KPy22[idx],KP_score[idx]);
							}
						}
						else{
							fprintf(fp_re,"%d %f %f\n",0,UP_x[idx],UP_y[idx]);
						}
					}
					fclose(fp_re);
					remove(featurepoint1);
					rename(featurepoint_tmp,featurepoint1);
					vector<int>().swap(file_order_bj); vector<int>().swap(file_order_idx);
					vector<float>().swap(KPx1); vector<float>().swap(KPy1);
					vector<float>().swap(KPx2); vector<float>().swap(KPy2);
					vector<float>().swap(KPx11); vector<float>().swap(KPy11);
					vector<float>().swap(KPx22); vector<float>().swap(KPy22);
					vector<int>().swap(KP_imgID); vector<float>().swap(KP_score);
					vector<float>().swap(UP_x); vector<float>().swap(UP_y);
					vector<float>().swap(CtrlPx1); vector<float>().swap(CtrlPy1);
					vector<float>().swap(CtrlPx2); vector<float>().swap(CtrlPy2);
				}
			}
			printf("M_count:%d    CX_count:%d\n",M_count,CX_count);
			printf("Tichu_CX end!\n\n");
			break;
		}
	}
}

void Observation::Tichu_CX_ByGlobalControl(float* fs_c,int LRmark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int CCD_num = cfg_.CCD_num;
	printf("Tichu_CX_ByGlobalControl begin! LRmark=%d\n", LRmark);

	std::vector<int> mosaic_c1(CCD_num * 4);
	std::vector<int> mosaic_c2(CCD_num * 4);
	if(!load_mosaic_coefficients(filepath, xulie_ID1, CCD_num, mosaic_c1.data()) ||
	   !load_mosaic_coefficients(filepath, xulie_ID2, CCD_num, mosaic_c2.data())){
		return;
	}

	std::vector<MosaicControlMatch> global_controls;
	const int ccd_begin = cfg_.ccd_begin();
	const int ccd_end = cfg_.ccd_end();
	for(int j=ccd_begin; j<ccd_end; ++j){
		char control_path[2048];
		snprintf(control_path, sizeof(control_path), "%s/%s/downsample/0/%s_RED%d_match.txt", filepath, xulie_ID1, xulie_ID1, j);
		FILE* fp = fopen(control_path, "r");
		if(fp == NULL){
			continue;
		}

		int bj, imgID;
		float row, col, mrow, mcol, mscore;
		while(true){
			if(fscanf(fp, "%d ", &bj) != 1){
				break;
			}
			if(bj == 1){
				if(fscanf(fp, "%f %f %d %f %f %f\n", &row, &col, &imgID, &mrow, &mcol, &mscore) != 6){
					break;
				}
				if(imgID < 0 || imgID >= CCD_num){
					continue;
				}
				MosaicControlMatch ctrl;
				ctrl.src_row = row  + mosaic_c1[j*4+0];
				ctrl.src_col = col  + mosaic_c1[j*4+2];
				ctrl.dst_row = mrow + mosaic_c2[imgID*4+0];
				ctrl.dst_col = mcol + mosaic_c2[imgID*4+2];
				if(pass_affine_gate(ctrl.src_row, ctrl.src_col, ctrl.dst_row, ctrl.dst_col, fs_c, false)){
					global_controls.push_back(ctrl);
				}
			}
			else{
				if(fscanf(fp, "%f %f\n", &row, &col) != 2){
					break;
				}
			}
		}
		fclose(fp);
	}

	printf("Global control count: %d\n", (int)global_controls.size());

	char* seq_id = (LRmark==0) ? xulie_ID1 : xulie_ID2;
	int* mosaic_src = (LRmark==0) ? mosaic_c1.data() : mosaic_c2.data();
	int* mosaic_dst = (LRmark==0) ? mosaic_c2.data() : mosaic_c1.data();

	int total_match_count = 0;
	int total_removed_count = 0;
	for(int j=ccd_begin; j<ccd_end-1; ++j){
		char input_path[2048];
		char temp_path[2048];
		snprintf(input_path, sizeof(input_path), "%s/%s/downsample/0/%s_RED%d_%d_intra__.txt", filepath, seq_id, seq_id, j, j+1);
		snprintf(temp_path, sizeof(temp_path), "%s/%s/downsample/0/%s_RED%d_%d_intra___gc_tmp.txt", filepath, seq_id, seq_id, j, j+1);

		FILE* fp = fopen(input_path, "r");
		if(fp == NULL){
			printf("未找到输入匹配文件：%s\n", input_path);
			continue;
		}

		std::vector<IntraMatchRecord> records;
		while(true){
			IntraMatchRecord rec;
			rec.imgID = -1;
			rec.mrow = 0.0f;
			rec.mcol = 0.0f;
			rec.score = 0.0f;

			if(fscanf(fp, "%d ", &rec.bj) != 1){
				break;
			}
			if(rec.bj == 1){
				if(fscanf(fp, "%f %f %d %f %f %f\n", &rec.row, &rec.col, &rec.imgID, &rec.mrow, &rec.mcol, &rec.score) != 6){
					break;
				}
			}
			else{
				if(fscanf(fp, "%f %f\n", &rec.row, &rec.col) != 2){
					break;
				}
			}
			records.push_back(rec);
		}
		fclose(fp);

		FILE* fp_out = fopen(temp_path, "w");
		if(fp_out == NULL){
			printf("无法创建输出文件：%s\n", temp_path);
			continue;
		}

		int pair_match_count = 0;
		int pair_removed_count = 0;
		for(int ii=0; ii<(int)records.size(); ++ii){
			const IntraMatchRecord& rec = records[ii];
			if(rec.bj != 1){
				fprintf(fp_out, "%d %f %f\n", rec.bj, rec.row, rec.col);
				continue;
			}
			if(rec.imgID < 0 || rec.imgID >= CCD_num){
				pair_removed_count++;
				total_removed_count++;
				continue;
			}

			pair_match_count++;
			total_match_count++;

			const float src_row = rec.row  + mosaic_src[j*4+0];
			const float src_col = rec.col  + mosaic_src[j*4+2];
			const float dst_row = rec.mrow + mosaic_dst[rec.imgID*4+0];
			const float dst_col = rec.mcol + mosaic_dst[rec.imgID*4+2];

			const bool keep = evaluate_with_global_controls(src_row, src_col, dst_row, dst_col, global_controls, fs_c, LRmark==1);
			if(keep){
				fprintf(fp_out, "%d %f %f %d %f %f %f\n", rec.bj, rec.row, rec.col, rec.imgID, rec.mrow, rec.mcol, rec.score);
			}
			else{
				pair_removed_count++;
				total_removed_count++;
			}
		}
		fclose(fp_out);

		remove(input_path);
		rename(temp_path, input_path);
		printf("LRmark=%d pair=%d matched=%d removed=%d\n", LRmark, j, pair_match_count, pair_removed_count);
	}

	printf("Tichu_CX_ByGlobalControl end! LRmark=%d matched=%d removed=%d controls=%d\n\n",
		LRmark, total_match_count, total_removed_count, (int)global_controls.size());
}

void Observation::MatchGet(int mark, float* fs_c){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	printf("Tichu_CX begin!\n");
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	char* filepath1=filepath;

	//逐层匹配
	int CCD_id;
	int BJ=0;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char ds_path_mosaictxt1[2048];
		snprintf(ds_path_mosaictxt1, sizeof(ds_path_mosaictxt1), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID1, i);
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		if(fpm1==NULL){
			printf("Error: %s not exist!\n",ds_path_mosaictxt1);
			continue;
		}
		std::vector<int> mosaic_c1(CCD_num * 4);
		bool mosaic_ok = true;
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]) != 5){
				printf("Error: failed reading %s\n", ds_path_mosaictxt1);
				mosaic_ok = false;
				break;
			}
		}
		fclose(fpm1);
		if(!mosaic_ok){
			continue;
		}
		char ds_path_mosaictxt2[2048];
		snprintf(ds_path_mosaictxt2, sizeof(ds_path_mosaictxt2), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID2, i);
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		if(fpm2==NULL){
			printf("Error: %s not exist!\n",ds_path_mosaictxt2);
			continue;
		}
		std::vector<int> mosaic_c2(CCD_num * 4);
		mosaic_ok = true;
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]) != 5){
				printf("Error: failed reading %s\n", ds_path_mosaictxt2);
				mosaic_ok = false;
				break;
			}
		}
		fclose(fpm2);
		if(!mosaic_ok){
			continue;
		}


		//存储结果并跳出
		if(BJ==1){
			int CX_count=0;
			int M_count=0;
			const int beginCCD = cfg_.ccd_begin();
			const int endCCD = cfg_.ccd_end();
			for(int j=beginCCD;j<endCCD;j++){
	            printf("%d\n",j);
				std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
				std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
				std::vector<int> matchimage_ID;
				std::vector<float> match_score;
				char featurepoint1[2048];
				if(mark==0){
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				}
				else if(mark==1){
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_grid.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				}
				else if(mark==2){
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_dc.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				}
				else{
					snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				}
				char featurepoint2[2048];
				snprintf(featurepoint2, sizeof(featurepoint2), "%s/%s/downsample/%d/%s_RED%d_match.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				FILE *fp=fopen(featurepoint1,"r");
				if(fp==NULL){
					printf("Error: %s not exist!\n",featurepoint1);
					continue;
				}

				FILE *fp_re=fopen(featurepoint2,"w");
				int bj,row,col,imgID,mrow,mcol;
				float m_score;
				double sfr=pow(double(2),double(i))/pow(double(2),double(4));
				int count=0;
				while(fscanf(fp,"%d ",&bj) == 1){
					if(bj==1){
						if(fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score) != 6){
							break;
						}
						KeyPoint_x1.push_back(row+mosaic_c1[j*4+0]);
						KeyPoint_y1.push_back(col+mosaic_c1[j*4+2]);
						KeyPoint_x2.push_back(mrow+mosaic_c2[imgID*4+0]);
						KeyPoint_y2.push_back(mcol+mosaic_c2[imgID*4+2]);
						KeyPoint_x11.push_back(row);
						KeyPoint_y11.push_back(col);
						KeyPoint_x22.push_back(mrow);
						KeyPoint_y22.push_back(mcol);
						matchimage_ID.push_back(imgID);
						match_score.push_back(m_score);
					}
					else{
						if(fscanf(fp,"%d %d\n",&row,&col) != 2){
							break;
						}
					}
				}
				fclose(fp);

				for(int jj=0;jj<KeyPoint_x1.size();jj++){
					float vx=KeyPoint_x1[jj]*fs_c[0]+KeyPoint_y1[jj]*fs_c[1]+fs_c[2]-KeyPoint_x2[jj];
					float vy=KeyPoint_x1[jj]*fs_c[3]+KeyPoint_y1[jj]*fs_c[4]+fs_c[5]-KeyPoint_y2[jj];
					M_count++;
					if(sqrt(vx*vx+vy*vy)<2500 && abs(vx)<1000 && abs(vy)<550){
						std::vector<double> d_r,d_c;
						std::vector<double> weighttemp;
						double dr_mean,dr_std,dc_mean,dc_std;
						dr_mean=dr_std=dc_mean=dc_std=0;
						for(int ii=0;ii<KeyPoint_x1.size();ii++){
							double dist=sqrt(double(KeyPoint_x1[jj]-KeyPoint_x1[ii])*(KeyPoint_x1[jj]-KeyPoint_x1[ii])+(KeyPoint_y1[jj]-KeyPoint_y1[ii])*(KeyPoint_y1[jj]-KeyPoint_y1[ii]));
							if(dist<500 && ii!=jj){
								d_r.push_back(KeyPoint_x2[ii]-KeyPoint_x1[ii]);
								d_c.push_back(KeyPoint_y2[ii]-KeyPoint_y1[ii]);
								double wt=1.0;
								//weighttemp.push_back(exp(-1/2*dist));
								weighttemp.push_back(wt);
								dr_mean += (KeyPoint_x2[ii]-KeyPoint_x1[ii])*wt;
								dc_mean += (KeyPoint_y2[ii]-KeyPoint_y1[ii])*wt;
							}
						}

						//行向视差
						//dr_mean =  std::accumulate(std::begin(d_r), std::end(d_r), 0.0) / d_r.size();  //均值
						dr_mean /= std::accumulate(std::begin(weighttemp), std::end(weighttemp), 0.0);
						std::for_each (std::begin(d_r), std::end(d_r), [&](const double d) {
							dr_std  += (d-dr_mean)*(d-dr_mean);
						});
						dr_std = sqrt(dr_std/d_r.size());  //标准差

						//列向视差
						//dc_mean =  std::accumulate(std::begin(d_c), std::end(d_c), 0.0) / d_c.size();  //均值
						dc_mean /= std::accumulate(std::begin(weighttemp), std::end(weighttemp), 0.0);
						std::for_each (std::begin(d_c), std::end(d_c), [&](const double d) {
							dc_std  += (d-dc_mean)*(d-dc_mean);
						});
						dc_std = sqrt(dc_std/d_c.size());  //标准差

						//3倍标准差剔除
						if( abs((KeyPoint_x2[jj]-KeyPoint_x1[jj])-dr_mean)<3*dr_std  &&  abs((KeyPoint_y2[jj]-KeyPoint_y1[jj])-dc_mean)<3*dc_std )
						{
							fprintf(fp_re,"%d %d %d %d %d %d %f\n",1,KeyPoint_x11[jj],KeyPoint_y11[jj],matchimage_ID[jj],KeyPoint_x22[jj],KeyPoint_y22[jj],match_score[jj]);
						}
						else{
							CX_count++;
						}
					}
					else{
						CX_count++;
					}//*/
				}
				fclose(fp_re);

				std::vector<int>().swap(KeyPoint_x1);
				std::vector<int>().swap(KeyPoint_y1);
				std::vector<int>().swap(KeyPoint_x2);
				std::vector<int>().swap(KeyPoint_y2);
				std::vector<int>().swap(KeyPoint_x11);
				std::vector<int>().swap(KeyPoint_y11);
				std::vector<int>().swap(KeyPoint_x22);
				std::vector<int>().swap(KeyPoint_y22);
				std::vector<int>().swap(matchimage_ID);
				std::vector<float>().swap(match_score);
			}
			printf("M_count:%d    CX_count:%d\n",M_count,CX_count);
			printf("Tichu_CX end!\n\n");
			break;
		}
	}
}


//生成网格点并匹配
void Observation::SemiDenseGrid_match1(float* fs_c){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	ImageMatch IM;
	printf("SemiDenseGrid_match begin!\n");
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	char* filepath1=filepath;

	int CCD_id;
	int BJ=0;
	const PipelineConfig::GridMatchParams& grid_cfg = cfg_.grid_match;
	int batch_size=grid_cfg.batch_size;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char ds_path_mosaictxt1[2048];
		snprintf(ds_path_mosaictxt1, sizeof(ds_path_mosaictxt1), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID1, i);
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		if(fpm1==NULL){
			printf("Error: %s not exist!\n",ds_path_mosaictxt1);
			continue;
		}
		std::vector<int> mosaic_c1(CCD_num * 4);
		bool mosaic_ok = true;
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]) != 5){
				printf("Error: failed reading %s\n", ds_path_mosaictxt1);
				mosaic_ok = false;
				break;
			}
		}
		fclose(fpm1);
		if(!mosaic_ok){
			continue;
		}
		char ds_path_mosaictxt2[2048];
		snprintf(ds_path_mosaictxt2, sizeof(ds_path_mosaictxt2), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID2, i);
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		if(fpm2==NULL){
			printf("Error: %s not exist!\n",ds_path_mosaictxt2);
			continue;
		}
		std::vector<int> mosaic_c2(CCD_num * 4);
		mosaic_ok = true;
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]) != 5){
				printf("Error: failed reading %s\n", ds_path_mosaictxt2);
				mosaic_ok = false;
				break;
			}
		}
		fclose(fpm2);
		if(!mosaic_ok){
			continue;
		}

		int kk=0;
		int begin=cfg_.ccd_begin();
		int end=cfg_.ccd_end();
		const int n_ccd = end - begin;
		const int n_pairs = std::max(1, n_ccd * n_ccd);
		int pair_done = 0;
		const double t0_layer = omp_get_wtime();
		printf("[SemiDenseGrid1] layer=%d CCD[%d,%d) pairs=%d neighbor_affine=%d\n",
			i, begin, end, n_pairs, (int)grid_cfg.use_neighbor_affine);
		fflush(stdout);
		// 点级并行；CCD 外层保持串行，避免 OpenMP 过订阅
		// #pragma omp parallel for schedule(dynamic)
		for(int j=begin;j<end;j++){
			ImageMatch IM_local;
			printf("[SemiDenseGrid1] LCCD=%d (%d/%d)\n", j, j - begin + 1, n_ccd);
			fflush(stdout);
			char imgL_path[2048];
			snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID1, i, xulie_ID1, j);
			char featurepoint1[2048];
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_grid.txt", filepath1, xulie_ID1, i, xulie_ID1, j);

			char featurepoint1_coarse[2048];

			//备份旧 _grid.txt（粗层约束）
			snprintf(featurepoint1_coarse, sizeof(featurepoint1_coarse), "%s/%s/downsample/%d/%s_RED%d_grid_coarse.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			rename(featurepoint1, featurepoint1_coarse);

			//生成格网点
			FILE *fp_grid=fopen(featurepoint1,"w");
			for(int rr=0;(rr+1)*batch_size<=rows1;rr++){
				for(int cc=0;(cc+1)*batch_size<=cols;cc++){
					fprintf(fp_grid,"%d %d %d\n",0,rr*batch_size+batch_size/2,cc*batch_size+batch_size/2);
				}
			}
			fclose(fp_grid);//*/

			char matchpoint1[2048];
			snprintf(matchpoint1, sizeof(matchpoint1), "%s/%s/downsample/%d/%s_RED%d_match.txt", filepath1, xulie_ID1, i, xulie_ID1, j);

			float fs_for_left[6];
			for(int ii=0; ii<6; ++ii) fs_for_left[ii] = fs_c[ii];
			if(grid_cfg.use_neighbor_affine){
				compute_grid_neighbor_affine(filepath1, xulie_ID1, i, j, begin, end, CCD_num,
					mosaic_c1.data(), mosaic_c2.data(), fs_c, fs_for_left);
			}

			char outpoint1[2048];
			snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/%d/%s_RED%d_.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			for(int k=begin;k<end;k++){
				++pair_done;
				printf("[SemiDenseGrid1] pair L=%d R=%d  %d/%d (%.0f%%) elapsed=%.1fs\n",
					j, k, pair_done, n_pairs, 100.0 * pair_done / n_pairs,
					omp_get_wtime() - t0_layer);
				fflush(stdout);
				char imgR_path[2048];
				snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID2, i, xulie_ID2, k);

				float fs_c1[6];
				fs_c1[0]=fs_for_left[0];
				fs_c1[1]=fs_for_left[1];
				fs_c1[2]=fs_for_left[2]+fs_for_left[0]*mosaic_c1[j*4+0]+fs_for_left[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
				fs_c1[3]=fs_for_left[3];
				fs_c1[4]=fs_for_left[4];
				fs_c1[5]=fs_for_left[5]+fs_for_left[3]*mosaic_c1[j*4+0]+fs_for_left[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];

				//limit_grid
				{
					int ret;
					if(j<3){
						ret=IM_local.limit_grid4(imgL_path, imgR_path, grid_cfg.window_size, grid_cfg.search_range, grid_cfg.cc_threshold, fs_c1, k, matchpoint1, featurepoint1, outpoint1, grid_cfg.lambda2, grid_cfg.affine_max_dev, grid_cfg.affine_pred_as_match);
					}
					else{
						ret=IM_local.limit_grid4(imgL_path, imgR_path, grid_cfg.window_size, grid_cfg.search_range, grid_cfg.cc_threshold, fs_c1, k, matchpoint1, featurepoint1, outpoint1, grid_cfg.lambda2, grid_cfg.affine_max_dev, grid_cfg.affine_pred_as_match);
					}
					if(ret != -1){
						int rem=remove(featurepoint1);
						int ren=rename(outpoint1,featurepoint1);
					}
				}
			}
		}
		printf("[SemiDenseGrid1] layer=%d done pairs=%d elapsed=%.1fs\n",
			i, pair_done, omp_get_wtime() - t0_layer);
		fflush(stdout);
		//i结束后，进行匹配点筛选和重排，加载拼接系数并计算fs系数。
		//仿射系数计算(由TichuCX代替)
	}
	printf("SemiDenseGrid_match end!\n");
}
void Observation::SemiDenseGrid_match2(float* fs_c,int mark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	ImageMatch IM;
	printf("SemiDenseGrid_match begin!\n");
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	char* filepath1=filepath;

	int CCD_id;
	int BJ=0;
	const PipelineConfig::GridMatchParams& grid_cfg = cfg_.grid_match;
	int batch_size=grid_cfg.batch_size;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char ds_path_mosaictxt1[2048];
		snprintf(ds_path_mosaictxt1, sizeof(ds_path_mosaictxt1), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID1, i);
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		if(fpm1==NULL){
			printf("Error: %s not exist!\n",ds_path_mosaictxt1);
			continue;
		}
		std::vector<int> mosaic_c1(CCD_num * 4);
		bool mosaic_ok = true;
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]) != 5){
				printf("Error: failed reading %s\n", ds_path_mosaictxt1);
				mosaic_ok = false;
				break;
			}
		}
		fclose(fpm1);
		if(!mosaic_ok){
			continue;
		}
		char ds_path_mosaictxt2[2048];
		snprintf(ds_path_mosaictxt2, sizeof(ds_path_mosaictxt2), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID2, i);
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		if(fpm2==NULL){
			printf("Error: %s not exist!\n",ds_path_mosaictxt2);
			continue;
		}
		std::vector<int> mosaic_c2(CCD_num * 4);
		mosaic_ok = true;
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]) != 5){
				printf("Error: failed reading %s\n", ds_path_mosaictxt2);
				mosaic_ok = false;
				break;
			}
		}
		fclose(fpm2);
		if(!mosaic_ok){
			continue;
		}

		int kk=0;
		int begin=cfg_.ccd_begin();
		int end=cfg_.ccd_end();
		const int n_ccd = end - begin;
		const int n_pairs = std::max(1, n_ccd * n_ccd);
		int pair_done = 0;
		const double t0_layer = omp_get_wtime();
		printf("[SemiDenseGrid2] layer=%d CCD[%d,%d) pairs=%d mark=%d\n", i, begin, end, n_pairs, mark);
		fflush(stdout);
		// if(mark==1){
		// 	begin=0;
		// 	end=9;
		// }
		//生成网格点并匹配
		for(int j=begin;j<end;j++){
			printf("[SemiDenseGrid2] LCCD=%d (%d/%d)\n", j, j - begin + 1, n_ccd);
			fflush(stdout);
			char imgL_path[2048];
			snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID1, i, xulie_ID1, j);
			char featurepoint1[2048];
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_match.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			if(mark==1){
				snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_%d_intra__.txt", filepath1, xulie_ID1, i, xulie_ID1, j, j+1);
			}

			char outpoint1[2048];

			snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/%d/%s_RED%d_.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			for(int k=begin;k<end;k++){
				++pair_done;
				printf("[SemiDenseGrid2] pair L=%d R=%d  %d/%d (%.0f%%) elapsed=%.1fs\n",
					j, k, pair_done, n_pairs, 100.0 * pair_done / n_pairs,
					omp_get_wtime() - t0_layer);
				fflush(stdout);
				char imgR_path[2048];
				snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID2, i, xulie_ID2, k);

				float fs_c1[6];
				fs_c1[0]=fs_c[0];
				fs_c1[1]=fs_c[1];
				fs_c1[2]=fs_c[2]+fs_c[0]*mosaic_c1[j*4+0]+fs_c[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
				fs_c1[3]=fs_c[3];
				fs_c1[4]=fs_c[4];
				fs_c1[5]=fs_c[5]+fs_c[3]*mosaic_c1[j*4+0]+fs_c[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];

				//limit_grid
				{
					int ret;
					if(j<3){
						ret=IM.limit_grid5(imgL_path, imgR_path, grid_cfg.followup_window_size, grid_cfg.followup_search_range, grid_cfg.followup_cc_threshold, fs_c1, k, featurepoint1, outpoint1, grid_cfg.lambda2, grid_cfg.affine_max_dev);
					}
					else{
						ret=IM.limit_grid5(imgL_path, imgR_path, grid_cfg.followup_window_size, grid_cfg.followup_search_range, grid_cfg.followup_cc_threshold, fs_c1, k, featurepoint1, outpoint1, grid_cfg.lambda2, grid_cfg.affine_max_dev);
					}
					if(ret != -1){
						int rem=remove(featurepoint1);
						int ren=rename(outpoint1,featurepoint1);
					}
				}
			}

		}
		printf("[SemiDenseGrid2] layer=%d done pairs=%d elapsed=%.1fs\n",
			i, pair_done, omp_get_wtime() - t0_layer);
		fflush(stdout);
		//i结束后，进行匹配点筛选和重排，加载拼接系数并计算fs系数。
		//仿射系数计算(由TichuCX代替)
	}
	printf("SemiDenseGrid_match end!\n");
}

//匹配结果精化
void Observation::Iterative_refinement(float* fs_c, int mark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	ImageMatch IM;
	printf("Iterative_refinement begin!\n");
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	char* filepath1=filepath;

	int CCD_id;
	int BJ=0;
	const PipelineConfig::GridMatchParams& grid_cfg = cfg_.grid_match;
	const PipelineConfig::FeatureMatchParams& fm = cfg_.feature_match;
	int batch_size=grid_cfg.batch_size;
	for(int i=0;i>=0;i--){
		if(i==0){
			BJ=1;
		}
		//读取拼接系数
		char ds_path_mosaictxt1[2048];
		snprintf(ds_path_mosaictxt1, sizeof(ds_path_mosaictxt1), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID1, i);
		FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
		if(fpm1==NULL){
			printf("Error: %s not exist!\n",ds_path_mosaictxt1);
			continue;
		}
		std::vector<int> mosaic_c1(CCD_num * 4);
		bool mosaic_ok = true;
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]) != 5){
				printf("Error: failed reading %s\n", ds_path_mosaictxt1);
				mosaic_ok = false;
				break;
			}
		}
		fclose(fpm1);
		if(!mosaic_ok){
			continue;
		}
		char ds_path_mosaictxt2[2048];
		snprintf(ds_path_mosaictxt2, sizeof(ds_path_mosaictxt2), "%s/%s/downsample/%d/mosaic.txt", filepath1, xulie_ID2, i);
		FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
		if(fpm2==NULL){
			printf("Error: %s not exist!\n",ds_path_mosaictxt2);
			continue;
		}
		std::vector<int> mosaic_c2(CCD_num * 4);
		mosaic_ok = true;
		for(int ii=0;ii<CCD_num;ii++){
			if(fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]) != 5){
				printf("Error: failed reading %s\n", ds_path_mosaictxt2);
				mosaic_ok = false;
				break;
			}
		}
		fclose(fpm2);
		if(!mosaic_ok){
			continue;
		}

		int begin=cfg_.ccd_begin();
		int end=cfg_.ccd_end();
		const int n_ccd = end - begin;
		const int n_pairs = std::max(1, n_ccd * n_ccd);
		int pair_done = 0;
		const double t0_ir = omp_get_wtime();
		printf("[IR] mark=%d ccd=[%d,%d) pairs=%d mode=%s threads=%d\n",
			mark, begin, end, n_pairs, mark == 0 ? "limit_fea" : "limit_grid4",
			omp_get_max_threads());
		fflush(stdout);

		//生成网格点并匹配
		for(int j=begin;j<end;j++){
			printf("[IR] left CCD=%d\n", j);
			fflush(stdout);
			char imgL_path[2048];
			snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID1, i, xulie_ID1, j);

			char featurepoint_temp[2048];
			snprintf(featurepoint_temp, sizeof(featurepoint_temp), "%s/%s/downsample/%d/%s_RED%d_temp.txt", filepath1, xulie_ID1, i, xulie_ID1, j);

			char featurepoint1[2048];

			if(mark==0){
				snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				//重新匹配64*64特征
				FILE *fp_fea=fopen(featurepoint1,"r");
				if(fp_fea==NULL){
					printf("Error: %s not exist!\n",featurepoint1);
					continue;
				}
				FILE *fp_grid=fopen(featurepoint_temp,"w");
				int bj,row,col,imgID,mrow,mcol;
				float m_score;
				while(fscanf(fp_fea,"%d ",&bj) == 1){
					if(bj==1){
						if(fscanf(fp_fea,"%d %d %d %d %d %f\n",&row,&col,&imgID,&mrow,&mcol,&m_score) != 6){
							break;
						}
						fprintf(fp_grid,"%d %d %d\n",0,row,col);
					}
					else{
						if(fscanf(fp_fea,"%d %d\n",&row,&col) != 2){
							break;
						}
						fprintf(fp_grid,"%d %d %d\n",0,row,col);
					}
				}
				fclose(fp_fea);
				fclose(fp_grid);
			}
			else{
				snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_grid.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
				//生成格网点
				FILE *fp_grid=fopen(featurepoint_temp,"w");
				for(int rr=0;(rr+1)*batch_size<=rows1;rr++){
					for(int cc=0;(cc+1)*batch_size<=cols;cc++){
						fprintf(fp_grid,"%d %d %d\n",0,rr*batch_size+batch_size/2,cc*batch_size+batch_size/2);
					}
				}
				fclose(fp_grid);
			}

			//读取控制数据
			char matchpoint1[2048];
			snprintf(matchpoint1, sizeof(matchpoint1), "%s/%s/downsample/%d/%s_RED%d_match.txt", filepath1, xulie_ID1, i, xulie_ID1, j);

			char outpoint1[2048];
			snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/%d/%s_RED%d_.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			for(int k=begin;k<end;k++){
				++pair_done;
				const double t0_pair = omp_get_wtime();
				printf("[IR] L%d R%d  pair %d/%d (%.0f%%) elapsed=%.1fs\n",
					j, k, pair_done, n_pairs, 100.0 * pair_done / n_pairs,
					omp_get_wtime() - t0_ir);
				fflush(stdout);
				char imgR_path[2048];
				snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID2, i, xulie_ID2, k);
				char featurepoint2[2048];
				snprintf(featurepoint2, sizeof(featurepoint2), "%s/%s/downsample/%d/%s_RED%d.txt", filepath1, xulie_ID2, i, xulie_ID2, k);

				float fs_c1[6];
				fs_c1[0]=fs_c[0];
				fs_c1[1]=fs_c[1];
				fs_c1[2]=fs_c[2]+fs_c[0]*mosaic_c1[j*4+0]+fs_c[1]*mosaic_c1[j*4+2]-mosaic_c2[k*4+0];
				fs_c1[3]=fs_c[3];
				fs_c1[4]=fs_c[4];
				fs_c1[5]=fs_c[5]+fs_c[3]*mosaic_c1[j*4+0]+fs_c[4]*mosaic_c1[j*4+2]-mosaic_c2[k*4+2];

				//limit_grid/fea（算法恢复为原始 limit_fea / limit_grid4）
				if(mark==0){
					IM.limit_fea(imgL_path, imgR_path,
						fm.iterative_window_size, fm.iterative_search_range, fm.iterative_cc_threshold,
						fs_c1, k, matchpoint1, featurepoint_temp, featurepoint2, outpoint1);
					remove(featurepoint_temp);
					rename(outpoint1,featurepoint_temp);
				}
				else{
					IM.limit_grid4(imgL_path, imgR_path,
						grid_cfg.iterative_window_size, grid_cfg.iterative_search_range, grid_cfg.iterative_cc_threshold,
						fs_c1, k, matchpoint1, featurepoint_temp, outpoint1, grid_cfg.lambda2, grid_cfg.affine_max_dev, grid_cfg.affine_pred_as_match);
					remove(featurepoint_temp);
					rename(outpoint1,featurepoint_temp);
				}
				printf("[IR] done L%d-R%d %.1fs\n", j, k, omp_get_wtime() - t0_pair);
				fflush(stdout);
			}
			remove(featurepoint1);
			rename(featurepoint_temp,featurepoint1);
		}
		printf("[IR] finished elapsed=%.1fs\n", omp_get_wtime() - t0_ir);
		fflush(stdout);
		//i结束后，进行匹配点筛选和重排，加载拼接系数并计算fs系数。
		//仿射系数计算(由TichuCX代替)
	}
	printf("Iterative_refinement end!\n");
}


//CCD间连接点
void Observation::Generate_matchPoint_Between_CCD(float* fs_c){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int CCD_num = cfg_.CCD_num;
	char* filepath1=filepath;

	const PipelineConfig::InterCcdMatchParams& inter_cfg = cfg_.inter_ccd_match;
	int intra_w_size=inter_cfg.intra_window_size;
	int intra_ser_range=inter_cfg.intra_search_range;
	int intra_batch_size_r=inter_cfg.intra_batch_size_r;
	int intra_batch_size_c=inter_cfg.intra_batch_size_c;
	float intra_thres=inter_cfg.intra_cc_threshold;
	int cross_w_size=inter_cfg.cross_window_size;
	int cross_ser_range=inter_cfg.cross_search_range;
	float cross_thres=inter_cfg.cross_cc_threshold;
	bool use_grid_controls=inter_cfg.control_use_grid();
	int mark=inter_cfg.run_global_refine ? 1 : 0;

	int beginCCD=cfg_.ccd_begin();
	int endCCD=cfg_.ccd_end()-1;  // 相邻对 j 与 j+1，j 上界为 ccd_end-1
	if(endCCD<=beginCCD){
		printf("[inter_ccd] CCD range [%d,%d) too small for inter-CCD match\n",
			cfg_.ccd_begin(), cfg_.ccd_end());
		fflush(stdout);
		return;
	}
	const int n_pairs = endCCD - beginCCD;
	const double t0_all = omp_get_wtime();
	// 嵌套 OpenMP 关闭：外层 pair 并行时，内层点并行自动退化为单线程
	omp_set_nested(0);
	const bool parallel_pairs = n_pairs >= omp_get_max_threads();
	printf("[inter_ccd] begin pairs=%d j in [%d,%d) mark=%d threads=%d pair_parallel=%d "
		"intra(w=%d ser=%d thr=%.2f batch=%dx%d) cross(w=%d ser=%d thr=%.2f control=%s)\n",
		n_pairs, beginCCD, endCCD, mark, omp_get_max_threads(), (int)parallel_pairs,
		intra_w_size, intra_ser_range, intra_thres, intra_batch_size_r, intra_batch_size_c,
		cross_w_size, cross_ser_range, cross_thres, use_grid_controls ? "grid" : "feature");
	fflush(stdout);

	MatrixXd fs0(3,3);
	fs0 << fs_c[0],fs_c[1],fs_c[2],fs_c[3],fs_c[4],fs_c[5],0,0,1;
	MatrixXd fs1 = fs0.inverse();
	float fs_c_inv[6];
	fs_c_inv[0]=fs1(0,0); fs_c_inv[1]=fs1(0,1); fs_c_inv[2]=fs1(0,2);
	fs_c_inv[3]=fs1(1,0); fs_c_inv[4]=fs1(1,1); fs_c_inv[5]=fs1(1,2);

	auto count_match_lines = [](const char* path) -> int {
		FILE* f = fopen(path, "r");
		if(!f) return 0;
		int n = 0;
		char line[2048];
		while(fgets(line, sizeof(line), f)) ++n;
		fclose(f);
		return n;
	};

	//左影像序列：邻 CCD 密格网（按 CCD 对并行）
	printf("[inter_ccd] stage=intra_L pairs=%d\n", n_pairs);
	fflush(stdout);
	const double t0_intra_L = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic) if(parallel_pairs)
	for(int j=beginCCD;j<endCCD;j++){
		const double t0 = omp_get_wtime();
		const int done = j - beginCCD + 1;
		#pragma omp critical(inter_ccd_progress)
		{
			printf("[inter_ccd] intra_L %d-%d  %d/%d (%.0f%%) elapsed=%.1fs\n",
				j, j+1, done, n_pairs, 100.0 * done / n_pairs,
				omp_get_wtime() - t0_intra_L);
			fflush(stdout);
		}
		ImageMatch IMtemp;
		char imgL_path[2048];
		snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j);
		char imgR_path[2048];
		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j+1);
		char outpoint1[2048];
		snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra.txt", filepath1, xulie_ID1, xulie_ID1, j, j+1);

		IMtemp.intra_CCD_match_B(imgL_path,imgR_path,intra_w_size,intra_ser_range,intra_thres,intra_batch_size_r,intra_batch_size_c,j,outpoint1);
		#pragma omp critical(inter_ccd_progress)
		{
			printf("[inter_ccd] intra_L %d-%d done points=%d %.1fs\n",
				j, j+1, count_match_lines(outpoint1), omp_get_wtime() - t0);
			fflush(stdout);
		}
	}
	printf("[inter_ccd] intra_L finished elapsed=%.1fs\n", omp_get_wtime() - t0_intra_L);
	fflush(stdout);
	if(mark==1){
		printf("[inter_ccd] Tichu_CX(intra_L) …\n");
		fflush(stdout);
		Tichu_CX(3,fs_c,0);
	}

	//左序列：航带间连接（cross，按 CCD 对并行）
	printf("[inter_ccd] stage=cross_L pairs=%d\n", n_pairs);
	fflush(stdout);
	const double t0_cross_L = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic) if(parallel_pairs)
	for(int j=beginCCD;j<endCCD;j++){
		const double t0 = omp_get_wtime();
		const int done = j - beginCCD + 1;
		#pragma omp critical(inter_ccd_progress)
		{
			printf("[inter_ccd] cross_L pair %d-%d  %d/%d (%.0f%%) elapsed=%.1fs\n",
				j, j+1, done, n_pairs, 100.0 * done / n_pairs,
				omp_get_wtime() - t0_cross_L);
			fflush(stdout);
		}
		ImageMatch IMtemp;
		char imgL_path[2048];
		snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j);
		char imgR_path[2048];
		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j+1);
		char outpoint1[2048];
		char featurepoint1[2048];

		snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra_match.txt", filepath1, xulie_ID1, xulie_ID1, j, j+1);
		if(mark==0){
			remove(outpoint1);
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra.txt", filepath1, xulie_ID1, xulie_ID1, j, j+1);
			rename(featurepoint1,outpoint1);
		}
		snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra__.txt", filepath1, xulie_ID1, xulie_ID1, j, j+1);

		FILE *fp=fopen(outpoint1,"r");
		FILE *fp1=fopen(featurepoint1,"w");
		if(fp==NULL || fp1==NULL){
			if(fp!=NULL){
				fclose(fp);
			}
			if(fp1!=NULL){
				fclose(fp1);
			}
			#pragma omp critical(inter_ccd_progress)
			{
				printf("[inter_ccd] Error opening file: %s or %s\n", outpoint1, featurepoint1);
				fflush(stdout);
			}
			continue;
		}
		int n_seed = 0;
		int imgID,mimgID;
		float row,col,mrow,mcol,mscore;
		while(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore)==7){
			fprintf(fp1,"%d %f %f\n",0,row,col);
			++n_seed;
		}
		fclose(fp);
		fclose(fp1);

		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, beginCCD);
		snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/0/%s_RED%d_.txt", filepath1, xulie_ID1, xulie_ID1, j);
		IMtemp.limit_grid_global(imgL_path, imgR_path, cross_w_size, cross_ser_range, cross_thres, fs_c, 0, featurepoint1, outpoint1, use_grid_controls);
		remove(featurepoint1);
		rename(outpoint1,featurepoint1);
		#pragma omp critical(inter_ccd_progress)
		{
			printf("[inter_ccd] cross_L %d-%d done seeds=%d out=%d %.1fs\n",
				j, j+1, n_seed, count_match_lines(featurepoint1), omp_get_wtime() - t0);
			fflush(stdout);
		}
	}
	printf("[inter_ccd] cross_L finished elapsed=%.1fs\n", omp_get_wtime() - t0_cross_L);
	fflush(stdout);

	//右影像序列：邻 CCD 密格网（按 CCD 对并行）
	printf("[inter_ccd] stage=intra_R pairs=%d\n", n_pairs);
	fflush(stdout);
	const double t0_intra_R = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic) if(parallel_pairs)
	for(int j=beginCCD;j<endCCD;j++){
		const double t0 = omp_get_wtime();
		const int done = j - beginCCD + 1;
		#pragma omp critical(inter_ccd_progress)
		{
			printf("[inter_ccd] intra_R %d-%d  %d/%d (%.0f%%) elapsed=%.1fs\n",
				j, j+1, done, n_pairs, 100.0 * done / n_pairs,
				omp_get_wtime() - t0_intra_R);
			fflush(stdout);
		}
		ImageMatch IMtemp;
		char imgL_path[2048];
		snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, j);
		char imgR_path[2048];
		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, j+1);
		char outpoint1[2048];
		snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra.txt", filepath1, xulie_ID2, xulie_ID2, j, j+1);

		IMtemp.intra_CCD_match_B(imgL_path,imgR_path,intra_w_size,intra_ser_range,intra_thres,intra_batch_size_r,intra_batch_size_c,j,outpoint1);
		#pragma omp critical(inter_ccd_progress)
		{
			printf("[inter_ccd] intra_R %d-%d done points=%d %.1fs\n",
				j, j+1, count_match_lines(outpoint1), omp_get_wtime() - t0);
			fflush(stdout);
		}
	}
	printf("[inter_ccd] intra_R finished elapsed=%.1fs\n", omp_get_wtime() - t0_intra_R);
	fflush(stdout);
	if(mark==1){
		printf("[inter_ccd] Tichu_CX(intra_R) …\n");
		fflush(stdout);
		Tichu_CX(3,fs_c,1);
	}

	//右序列：航带间连接（cross，按 CCD 对并行）
	printf("[inter_ccd] stage=cross_R pairs=%d\n", n_pairs);
	fflush(stdout);
	const double t0_cross_R = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic) if(parallel_pairs)
	for(int j=beginCCD;j<endCCD;j++){
		const double t0 = omp_get_wtime();
		const int done = j - beginCCD + 1;
		#pragma omp critical(inter_ccd_progress)
		{
			printf("[inter_ccd] cross_R pair %d-%d  %d/%d (%.0f%%) elapsed=%.1fs\n",
				j, j+1, done, n_pairs, 100.0 * done / n_pairs,
				omp_get_wtime() - t0_cross_R);
			fflush(stdout);
		}
		ImageMatch IMtemp;
		char imgL_path[2048];
		snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, j);
		char imgR_path[2048];
		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, j+1);
		char outpoint1[2048];
		char featurepoint1[2048];

		snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra_match.txt", filepath1, xulie_ID2, xulie_ID2, j, j+1);
		if(mark==0){
			remove(outpoint1);
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra.txt", filepath1, xulie_ID2, xulie_ID2, j, j+1);
			rename(featurepoint1,outpoint1);
		}
		snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra__.txt", filepath1, xulie_ID2, xulie_ID2, j, j+1);
		FILE *fp=fopen(outpoint1,"r");
		FILE *fp1=fopen(featurepoint1,"w");
		if(fp==NULL || fp1==NULL){
			if(fp!=NULL){
				fclose(fp);
			}
			if(fp1!=NULL){
				fclose(fp1);
			}
			#pragma omp critical(inter_ccd_progress)
			{
				printf("[inter_ccd] Error opening file: %s or %s\n", outpoint1, featurepoint1);
				fflush(stdout);
			}
			continue;
		}
		int n_seed = 0;
		int imgID,mimgID;
		float row,col,mrow,mcol,mscore;
		while(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore)==7){
			fprintf(fp1,"%d %f %f\n",0,row,col);
			++n_seed;
		}
		fclose(fp);
		fclose(fp1);

		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, beginCCD);
		snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/0/%s_RED%d_.txt", filepath1, xulie_ID2, xulie_ID2, j);
		IMtemp.limit_grid_global(imgL_path, imgR_path, cross_w_size, cross_ser_range, cross_thres, fs_c_inv, 0, featurepoint1, outpoint1, use_grid_controls);
		remove(featurepoint1);
		rename(outpoint1,featurepoint1);
		#pragma omp critical(inter_ccd_progress)
		{
			printf("[inter_ccd] cross_R %d-%d done seeds=%d out=%d %.1fs\n",
				j, j+1, n_seed, count_match_lines(featurepoint1), omp_get_wtime() - t0);
			fflush(stdout);
		}
	}
	printf("[inter_ccd] cross_R finished elapsed=%.1fs\n", omp_get_wtime() - t0_cross_R);
	printf("[inter_ccd] all stages done total=%.1fs\n", omp_get_wtime() - t0_all);
	fflush(stdout);
}

void Observation::JitterShow(){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	char* filepath1=filepath;

	int imgID,row,col,mimgID,mrow,mcol;
	float mscore;

	ImageMatch IMtemp;

	//左影像序列
	for(int j=0;j<CCD_num-1;j++){
		printf("intra_CCD_L:%d-%d……\n",j,j+1);
		char imgL_path[2048];
		snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j);
		char imgR_path[2048];
		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j+1);
		char outpoint1[2048];
		snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra.txt", filepath1, xulie_ID1, xulie_ID1, j, j+1);

		IMtemp.intra_CCD_match(imgL_path,imgR_path,15,20,0.75,1,5,j,outpoint1);
	}//*/

	//右影像序列
	for(int j=0;j<CCD_num-1;j++){
		printf("intra_CCD_R:%d-%d……\n",j,j+1);
		char imgL_path[2048];
		snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, j);
		char imgR_path[2048];
		snprintf(imgR_path, sizeof(imgR_path), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID2, xulie_ID2, j+1);
		char outpoint1[2048];
		snprintf(outpoint1, sizeof(outpoint1), "%s/%s/downsample/0/%s_RED%d_%d_intra.txt", filepath1, xulie_ID2, xulie_ID2, j, j+1);

		IMtemp.intra_CCD_match(imgL_path,imgR_path,15,20,0.75,1,5,j,outpoint1);
	}/**/
}

void Observation::Img_Preprcess(int seq_index){
	const char* filepath = cfg_.filepath;
	const char* xulie_ID = seq_index == 0 ? cfg_.xulie_ID1 : cfg_.xulie_ID2;
	int rows = seq_index == 0 ? cfg_.rows1 : cfg_.rows2;
	int cols = cfg_.cols;
	int CCD_num=cfg_.CCD_num;
	//影像预处理
	std::string src_path = std::string(filepath) + xulie_ID + "/src";

	std::string prefix = std::string(xulie_ID) + "_RED";
	//①降采样
	std::string downsample_path = std::string(filepath) + xulie_ID + "/downsample";
	int status = mkdir(downsample_path.c_str(), 0755);
	ImageProcess IPtemp;
	for(int level=1; level<5; level++){
		std::string ds_path = downsample_path + "/" + std::to_string(level);
		status = mkdir(ds_path.c_str(), 0755);
		int batchsize = pow(double(2), double(level));
		for(int ccd=0; ccd<CCD_num; ccd++){
			std::string img_path = src_path + "/" + prefix + "_RED" + std::to_string(ccd) + ".tif";
			std::string dst_path = ds_path + "/" + prefix + "_RED" + std::to_string(ccd) + ".tif";
			IPtemp.Down_sample(mutable_cstr(img_path), batchsize, mutable_cstr(dst_path));
		}
	}

	//②拼接
	std::string mosaic_path0 = std::string(filepath) + xulie_ID + "/mosaic";
	status = mkdir(mosaic_path0.c_str(), 0755);
	//(!status) ? (printf("Directory created\n")) : (printf("Unable to create directory\n"));

	int OverlapSamples=cfg_.mosaic.overlap_samples;
	xulie_mosaic1(mutable_cstr(src_path), mutable_cstr(prefix), OverlapSamples, seq_index);

	//③拼接影像降采样
	std::string mosaic_path2 = mosaic_path0 + "/mosaic1.tif";

	for(int j=1;j<5;j++){
		std::string ds_path = mosaic_path0 + "/mosaic_ds" + std::to_string(j) + ".tif";

		//降采样
		int batchsize=pow(double(2),double(j));
		Down_sample(mutable_cstr(mosaic_path2), batchsize, mutable_cstr(ds_path));
	}
}
//物方匹配
void Observation::ground_match(){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	char* filepath1=filepath;
	int CCD_num=cfg_.CCD_num;

	//输出采样外方位元素进行拟合
	char outeopath[2048];
	char eopzpath[2048];
	snprintf(outeopath, sizeof(outeopath), "../data/EO/%s.txt", xulie_ID1);
	snprintf(eopzpath, sizeof(eopzpath), "../data/EO/%s_pz.txt", xulie_ID1);
	EO::OutEO2txt(outeopath, eopzpath);
	snprintf(outeopath, sizeof(outeopath), "../data/EO/%s.txt", xulie_ID2);
	snprintf(eopzpath, sizeof(eopzpath), "../data/EO/%s_pz.txt", xulie_ID2);
	EO::OutEO2txt(outeopath, eopzpath);

	//拟合外方位元素并存储
	double Poly_C[30];
	char poly_path[2048];

	snprintf(outeopath, sizeof(outeopath), "../data/EO/%s/%s_RED5_0.txt", xulie_ID1, xulie_ID1);
	snprintf(poly_path, sizeof(poly_path), "../data/EO/%s/polyCC.txt", xulie_ID1);
	Polynomial3_EO(outeopath,Poly_C,poly_path);

	snprintf(outeopath, sizeof(outeopath), "../data/EO/%s/%s_RED5_0.txt", xulie_ID2, xulie_ID2);
	snprintf(poly_path, sizeof(poly_path), "../data/EO/%s/polyCC.txt", xulie_ID2);
	Polynomial3_EO(outeopath,Poly_C,poly_path);

	//预测左影像对应的地面点
	//float Z;
	//LoadDEM("E:/Mars_VS/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",10,10,&Z);

	float EO[6];
	float IO[10];
	float GC[3];
	//double* Poly_C = new double[30];

	///////////////////////////////////////////////////////
	//计算每个兴趣点的地面坐标
	int ch=6;
	int Localmax_win[5]={513,129,65,49,37};
	for(int i=4;i>=4;i--){
		FILE *fp_f,*fp_eo,*fp_io,*fp_poly;

		//提取EO拟合参数
		//char* poly_path = new char[160];
		snprintf(poly_path, sizeof(poly_path), "../data/EO/%s/polyCC.txt", xulie_ID1);
		fp_poly = fopen(poly_path, "r");
		for(int i=0;i<6;i++){
			for(int j=0;j<5;j++){
				if(fscanf(fp_poly,"%lf ",&Poly_C[i*5+j]) != 1) break;
			}
		}
		fclose(fp_poly);

		//读取内方位参数
		fp_io = fopen("../data/IO/IO.txt", "r");

		//提取特征点并求出其对应地面坐标
		for(int j=0;j<CCD_num;j++){
			char imgL_path[2048];
			snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID1, i, xulie_ID1, j);
			char featurepoint1[2048];
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/ground/%d/%s_RED%d.txt", filepath1, xulie_ID1, i, xulie_ID1, j);
			std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
			ImageMatch IM;
			IM.Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			//读取对应的内方位元素
			if(!load_io_row(fp_io, IO)){
				printf("[WARN][ground_match] IO 行读取失败，跳过 CCD%d\n", j);
				continue;
			}

			//读取对应CCD影像的起始采样时间
			double beginET,LR;
			char EOpath[2048];
			snprintf(EOpath, sizeof(EOpath), "../data/EO/%s/%s_RED%d_0.txt", xulie_ID1, xulie_ID1, j);
			fp_eo=fopen(EOpath,"r");
			if(fp_eo == NULL || fscanf(fp_eo,"%lf %lf\n",&beginET,&LR) != 2){
				if(fp_eo) fclose(fp_eo);
				continue;
			}
			fclose(fp_eo);

			fp_f=fopen(featurepoint1,"w");
			for(int ii=0;ii<KeyPoint_x1.size();ii++){
				KeyPoint_x1[ii]=KeyPoint_x1[ii]*16;
				KeyPoint_y1[ii]=KeyPoint_y1[ii]*16;
				double et=beginET+KeyPoint_x1[ii]*1*LR;
				Get_PolyEO(et,Poly_C,EO);
				//Get_PolyEO1(et,KeyPoint_x1[ii],Poly_C,EO);
				Get_groundtruth(const_cast<char*>("E:/Mars_VS/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"), KeyPoint_y1[ii], KeyPoint_x1[ii], 1, 128, 0, EO, IO, GC);
				fprintf(fp_f,"%d %d %d %f %f %f\n",0,KeyPoint_x1[ii],KeyPoint_y1[ii],GC[0],GC[1],GC[2]);
			}
			std::vector<int>().swap(KeyPoint_x1);
			std::vector<int>().swap(KeyPoint_y1);
			fclose(fp_f);
		}


		rewind(fp_io);
		//提取EO拟合参数
		snprintf(poly_path, sizeof(poly_path), "../data/EO/%s/polyCC.txt", xulie_ID2);
		fp_poly = fopen(poly_path, "r");
		for(int i=0;i<6;i++){
			for(int j=0;j<5;j++){
				if(fscanf(fp_poly,"%lf ",&Poly_C[i*5+j]) != 1) break;
			}
		}
		fclose(fp_poly);
		for(int j=0;j<CCD_num;j++){
			char imgL_path[2048];
			snprintf(imgL_path, sizeof(imgL_path), "%s/%s/downsample/%d/%s_RED%d.tif", filepath, xulie_ID2, i, xulie_ID2, j);
			char featurepoint1[2048];
			snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/ground/%d/%s_RED%d.txt", filepath1, xulie_ID2, i, xulie_ID2, j);

			std::vector<int> KeyPoint_x1,KeyPoint_y1;  //x为行，y为列
			ImageMatch IM;
			IM.Feature_Detection(imgL_path,ch,Localmax_win[i],KeyPoint_x1,KeyPoint_y1);

			//读取对应的内方位元素
			if(!load_io_row(fp_io, IO)){
				printf("[WARN][ground_match] IO 行读取失败，跳过 CCD%d\n", j);
				continue;
			}

			//读取对应CCD影像的起始采样时间
			double beginET,LR;
			char EOpath[2048];
			snprintf(EOpath, sizeof(EOpath), "../data/EO/%s/%s_RED%d_0.txt", xulie_ID2, xulie_ID2, j);
			//sprintf( EOpath, "%s%d%s", "../data/EO/1650/PSP_001777_1650_RED", j, "_0.txt");
			fp_eo=fopen(EOpath,"r");
			if(fp_eo == NULL || fscanf(fp_eo,"%lf %lf\n",&beginET,&LR) != 2){
				if(fp_eo) fclose(fp_eo);
				continue;
			}
			fclose(fp_eo);

			fp_f=fopen(featurepoint1,"w");
			for(int ii=0;ii<KeyPoint_x1.size();ii++){
				KeyPoint_x1[ii]=KeyPoint_x1[ii]*16;
				KeyPoint_y1[ii]=KeyPoint_y1[ii]*16;
				double et=beginET+KeyPoint_x1[ii]*1*LR;
				Get_PolyEO(et,Poly_C,EO);
				//Get_PolyEO1(et,KeyPoint_x1[ii],Poly_C,EO);
				Get_groundtruth(const_cast<char*>("E:/Mars_VS/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"), KeyPoint_y1[ii], KeyPoint_x1[ii], 1, 128, 4000, EO, IO, GC);
				fprintf(fp_f,"%d %d %d %f %f %f\n",0,KeyPoint_x1[ii],KeyPoint_y1[ii],GC[0],GC[1],GC[2]);
			}
			std::vector<int>().swap(KeyPoint_x1);
			std::vector<int>().swap(KeyPoint_y1);
			fclose(fp_f);
		}
		fclose(fp_io);
	}


	//Get_PolyEO(et,Poly_C,EO);
	Get_groundtruth(const_cast<char*>("E:/Mars_VS/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"), 0, 0, 1, 128, 0, EO, IO, GC);
}

//生成格式文件用于BA
bool Observation::prepare_for_BA(int mark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	char* filepath1=filepath;
	int CCD_num=cfg_.CCD_num;
	const int ccd_check = std::min(std::max(cfg_.ccd_begin(), 0), std::max(cfg_.CCD_num - 1, 0));

	char outeopath[2048];
	char eopzpath[2048];

	// OutEO2txt 失败时可能留下仅表头的残缺 RED*_0.txt；必须校验采样行数
	auto eo_file_ok = [](const char* path) -> bool {
		FILE* fp = fopen(path, "r");
		if(fp == NULL) return false;
		char line[2048];
		if(fgets(line, sizeof(line), fp) == NULL){ fclose(fp); return false; }
		int n = 0;
		while(fgets(line, sizeof(line), fp) != NULL){
			double et,Xs,Ys,Zs,phi,w,ka;
			if(sscanf(line, "%lf %lf %lf %lf %lf %lf %lf",
				&et,&Xs,&Ys,&Zs,&phi,&w,&ka) == 7){
				++n;
			}
		}
		fclose(fp);
		return n >= 10;
	};
	auto poly_file_ok = [](const char* path) -> bool {
		FILE* fp = fopen(path, "r");
		if(fp == NULL) return false;
		int n = 0;
		double a1,a2,a3,a4,a5;
		while(fscanf(fp, "%lf %lf %lf %lf %lf\n", &a1,&a2,&a3,&a4,&a5) == 5){
			++n;
		}
		fclose(fp);
		return n >= 6;
	};

	char eo_check1[2048], eo_check2[2048];
	snprintf(eo_check1, sizeof(eo_check1), "../data/EO/%s/%s_RED%d_0.txt", xulie_ID1, xulie_ID1, ccd_check);
	snprintf(eo_check2, sizeof(eo_check2), "../data/EO/%s/%s_RED%d_0.txt", xulie_ID2, xulie_ID2, ccd_check);
	const bool reuse_eo = eo_file_ok(eo_check1) && eo_file_ok(eo_check2);
	if(reuse_eo){
		printf("[INFO][prepare_for_BA] 复用已有 EO 时间序列（跳过 OutEO2txt/SPICE） mark=%d\n", mark);
	} else {
		snprintf(outeopath, sizeof(outeopath), "../data/EO/%s.txt", xulie_ID1);
		snprintf(eopzpath, sizeof(eopzpath), "../data/EO/%s_pz.txt", xulie_ID1);
		EO::OutEO2txt(outeopath, eopzpath,200);
		snprintf(outeopath, sizeof(outeopath), "../data/EO/%s.txt", xulie_ID2);
		snprintf(eopzpath, sizeof(eopzpath), "../data/EO/%s_pz.txt", xulie_ID2);
		EO::OutEO2txt(outeopath, eopzpath,200);
	}

	if(!eo_file_ok(eo_check1) || !eo_file_ok(eo_check2)){
		fprintf(stderr,
			"[ERROR][prepare_for_BA] EO 时间序列无效或未生成，已中止 BA 准备。\n"
			"  检查: %s\n"
			"  检查: %s\n"
			"  请确认从 bin/ 运行 Mars，并检查 data/EO/{id}_pz.txt 中 SPICE 内核是否完整\n"
			"  （常见：spk/*.bsp 下载失败只剩几 KB，需重新 --download）。\n",
			eo_check1, eo_check2);
		return false;
	}

	//拟合外方位元素并存储
	double Poly_C[30];
	char poly_path[2048];
	char poly1[2048];
	char poly2[2048];
	snprintf(poly1, sizeof(poly1), "../data/EO/%s/polyCC.txt", xulie_ID1);
	snprintf(poly2, sizeof(poly2), "../data/EO/%s/polyCC.txt", xulie_ID2);
	const bool reuse_poly = reuse_eo && poly_file_ok(poly1) && poly_file_ok(poly2);
	if(reuse_poly){
		printf("[INFO][prepare_for_BA] 复用已有 polyCC.txt\n");
	} else {
		snprintf(outeopath, sizeof(outeopath), "../data/EO/%s/%s_RED%d_0.txt", xulie_ID1, xulie_ID1, ccd_check);
		snprintf(poly_path, sizeof(poly_path), "../data/EO/%s/polyCC.txt", xulie_ID1);
		Polynomial3_EO(outeopath,Poly_C,poly_path);

		snprintf(outeopath, sizeof(outeopath), "../data/EO/%s/%s_RED%d_0.txt", xulie_ID2, xulie_ID2, ccd_check);
		snprintf(poly_path, sizeof(poly_path), "../data/EO/%s/polyCC.txt", xulie_ID2);
		Polynomial3_EO(outeopath,Poly_C,poly_path);
	}

	char* matchfile = filepath;
	char EOfile[] = "../data/EO";
	char IOtxt[] = "../data/IO/IO.txt";
	char observetxt[2048];
	snprintf(observetxt, sizeof(observetxt), "../data/observedata/%s_%s.txt", xulie_ID1, xulie_ID2);
	int observe_count=0;
	if(mark==0){
		SetObserveTxT_feaBA(matchfile,EOfile,IOtxt,xulie_ID1,xulie_ID2,observetxt,&observe_count);
	}
	else if(mark==1){
		SetObserveTxT_gridFI(matchfile,EOfile,IOtxt,xulie_ID1,xulie_ID2,observetxt,&observe_count);
	}
	else{
		SetObserveTxT2(matchfile,EOfile,IOtxt,xulie_ID1,xulie_ID2,observetxt,&observe_count);
	}
	return true;
}
//生成观测值文件
void Observation::SetObserveTxT_feaBA(char* matchfile,char* EOfile,char* IOtxt,char* xulie_ID1,char* xulie_ID2,char* observetxt,int* base){
	//int rows1=40000;
	//int rows2=80000;
	//读取内方位元素
	int CCD_num=cfg_.CCD_num;
	int TDI=128;
	int BIN=1;
	int count=0;
	float IO[10][10];
	if(!load_io_table(IOtxt, 10, IO, "SetObserveTxT_feaBA")){
		return;
	}

	char matchtxt[2048];
	char matchtxt1[2048];
	FILE *fp1=fopen(observetxt,"w");
	if(fp1 == NULL){
		printf("[ERROR][SetObserveTxT_feaBA] 无法创建观测文件: %s\n", observetxt);
		return;
	}
	float xp,yp;
	double et;
	float EO[6];
	float GC[3];
	double Poly_C[30];
	double polyL[30], polyR[30];
	std::vector<double> beginETL(CCD_num, 0.0), LRL(CCD_num, 0.0);
	std::vector<double> beginETR(CCD_num, 0.0), LRR(CCD_num, 0.0);
	std::vector<char> eoL_ok(CCD_num, 0), eoR_ok(CCD_num, 0);
	{
		char ppath[2048], eopath[2048];
		snprintf(ppath, sizeof(ppath), "%s/%s/polyCC.txt", EOfile, xulie_ID1);
		snprintf(eopath, sizeof(eopath), "%s/%s/polyCC.txt", EOfile, xulie_ID2);
		if(!load_polyCC_cache(ppath, polyL) || !load_polyCC_cache(eopath, polyR)){
			printf("[ERROR][SetObserveTxT_feaBA] 无法预加载 polyCC: %s / %s\n", ppath, eopath);
			fclose(fp1);
			return;
		}
		for(int j=0;j<CCD_num;j++){
			snprintf(eopath, sizeof(eopath), "%s/%s/%s_RED%d_0.txt", EOfile, xulie_ID1, xulie_ID1, j);
			eoL_ok[j] = load_eo_begin_lr(eopath, &beginETL[j], &LRL[j]) ? 1 : 0;
			snprintf(eopath, sizeof(eopath), "%s/%s/%s_RED%d_0.txt", EOfile, xulie_ID2, xulie_ID2, j);
			eoR_ok[j] = load_eo_begin_lr(eopath, &beginETR[j], &LRR[j]) ? 1 : 0;
		}
		printf("[INFO][SetObserveTxT_feaBA] 已缓存 polyCC + 各 CCD EO 头 (beginET,LR)\n");
	}

	char LimgCoor[2048];
	snprintf(LimgCoor, sizeof(LimgCoor), "../data/observedata/%s_%s_LCfea.txt", xulie_ID1, xulie_ID2);
	FILE *fp_lc=fopen(LimgCoor,"w");
	float H=-2200;

	int count0=0;
	const int ccd_begin = cfg_.ccd_begin();
	const int ccd_end = cfg_.ccd_end();
	//航带间
	for(int i=ccd_begin;i<ccd_end;i++){
		snprintf(matchtxt, sizeof(matchtxt), "%s/%s/downsample/0/%s_RED%d_match.txt", matchfile, xulie_ID1, xulie_ID1, i);
		char matchtxt_[2048];
		snprintf(matchtxt_, sizeof(matchtxt_), "%s/%s/downsample/0/%s_RED%d_match_.txt", matchfile, xulie_ID1, xulie_ID1, i);

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt_,"w");
		if(fp == NULL || fp_ == NULL){
			printf("[WARN][SetObserveTxT_feaBA] 跳过缺失匹配文件: %s\n", matchtxt);
			if(fp) fclose(fp);
			if(fp_) fclose(fp_);
			continue;
		}
		if(!eoL_ok[i]){
			printf("[WARN][SetObserveTxT_feaBA] 缺少左 CCD%d EO 头，跳过\n", i);
			fclose(fp); fclose(fp_);
			continue;
		}
		int bj,imgID;
		float row,col,mrow,mcol;
		float mscore;
		while(fscanf(fp,"%d ",&bj) == 1){
			if(bj==1){
				if(count0%10==0){
					if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore) != 6) break;
					if(imgID < 0 || imgID >= CCD_num || !eoR_ok[imgID]){
						count0++;
						continue;
					}
					IO_correct(col, BIN, TDI, IO[i], &xp, &yp);
					et=beginETL[i]+row*LRL[i];
					memcpy(Poly_C, polyL, sizeof(polyL));
					Get_PolyEO(et,Poly_C,EO);
					Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
					fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);
					fprintf(fp_,"%d %f %f %f %d\n",count, row, col, GC[2], imgID); //生成地面匹配控制文件
					fprintf(fp_lc,"%d %f %f\n",count, row, col);

					IO_correct(mcol, BIN, TDI, IO[imgID], &xp, &yp);
					et=beginETR[imgID]+mrow*LRR[imgID];
					memcpy(Poly_C, polyR, sizeof(polyR));
					Get_PolyEO(et,Poly_C,EO);
					Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
					fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
					count++;
				}
				else{
					if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore) != 6) break;
				}
				count0++;
			}
			else{
				if(fscanf(fp,"%f %f\n",&row,&col) != 2) break;
			}
		}
		fclose(fp);
		fclose(fp_);
	}//*/

	std::cout<<"航带间连接点数量："<<count<<std::endl;

	count0=0;
	//CCD间:左影像
	for(int i=ccd_begin;i<ccd_end-1;i++){
		snprintf(matchtxt, sizeof(matchtxt), "%s/%s/downsample/0/%s_RED%d_%d_intra_match.txt", matchfile, xulie_ID1, xulie_ID1, i, i+1);
		snprintf(matchtxt1, sizeof(matchtxt1), "%s/%s/downsample/0/%s_RED%d_%d_intra__.txt", matchfile, xulie_ID1, xulie_ID1, i, i+1);

		int imgID,mimgID;
		float row,col,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt1,"r");
		if(fp == NULL || fp_ == NULL){
			if(fp) fclose(fp);
			if(fp_) fclose(fp_);
			continue;
		}
		while(true){
			int bj;
			if(fscanf(fp_,"%d ",&bj) != 1) break;
			if(bj==1){
				if(count0%2==0){
					//带间
					if(fscanf(fp_,"%f %f %d %f %f %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore) != 6) break;
					if(mimgID >= 0 && mimgID < CCD_num && eoR_ok[mimgID]){
						IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
						et=beginETR[mimgID]+mrow*LRR[mimgID];
						memcpy(Poly_C, polyR, sizeof(polyR));
						Get_PolyEO(et,Poly_C,EO);
						Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
						fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
					}

					//读取CCD间数据
					if(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore) != 7) break;
					if(imgID >= 0 && imgID < CCD_num && eoL_ok[imgID] &&
					   mimgID >= 0 && mimgID < CCD_num && eoL_ok[mimgID]){
						IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
						et=beginETL[imgID]+row*LRL[imgID];
						memcpy(Poly_C, polyL, sizeof(polyL));
						Get_PolyEO(et,Poly_C,EO);
						Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
						fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

						//CCD间
						IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
						et=beginETL[mimgID]+mrow*LRL[mimgID];
						memcpy(Poly_C, polyL, sizeof(polyL));
						Get_PolyEO(et,Poly_C,EO);
						Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
						fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);
						count++;
					}
				}
				else{
					if(fscanf(fp_,"%f %f %d %f %f %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore) != 6) break;
					if(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore) != 7) break;
				}
				count0++;
			}
			else{
				if(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore) != 7) break;
				if(fscanf(fp_,"%f %f\n",&row,&col) != 2) break;
			}
		}
		fclose(fp);
		fclose(fp_);
	}//*/


	//CCD间：右影像
	for(int i=ccd_begin;i<ccd_end-1;i++){
		snprintf(matchtxt, sizeof(matchtxt), "%s/%s/downsample/0/%s_RED%d_%d_intra_match.txt", matchfile, xulie_ID2, xulie_ID2, i, i+1);
		snprintf(matchtxt1, sizeof(matchtxt1), "%s/%s/downsample/0/%s_RED%d_%d_intra__.txt", matchfile, xulie_ID2, xulie_ID2, i, i+1);

		int imgID,mimgID;
		float row,col,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt1,"r");
		if(fp == NULL || fp_ == NULL){
			if(fp) fclose(fp);
			if(fp_) fclose(fp_);
			continue;
		}
		while(true){
			int bj;
			if(fscanf(fp_,"%d ",&bj) != 1) break;
			if(bj==1){
				if(count0%2==0){
					//带间
					if(fscanf(fp_,"%f %f %d %f %f %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore) != 6) break;
					if(mimgID >= 0 && mimgID < CCD_num && eoL_ok[mimgID]){
						IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
						et=beginETL[mimgID]+mrow*LRL[mimgID];
						memcpy(Poly_C, polyL, sizeof(polyL));
						Get_PolyEO(et,Poly_C,EO);
						Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
						fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);
						fprintf(fp_lc,"%d %f %f\n",count, mrow, mcol);
					}

					//读取CCD间数据
					if(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore) != 7) break;
					if(imgID >= 0 && imgID < CCD_num && eoR_ok[imgID] &&
					   mimgID >= 0 && mimgID < CCD_num && eoR_ok[mimgID]){
						IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
						et=beginETR[imgID]+row*LRR[imgID];
						memcpy(Poly_C, polyR, sizeof(polyR));
						Get_PolyEO(et,Poly_C,EO);
						Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
						fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);

						//CCD间
						IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
						et=beginETR[mimgID]+mrow*LRR[mimgID];
						memcpy(Poly_C, polyR, sizeof(polyR));
						Get_PolyEO(et,Poly_C,EO);
						Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
						fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
						count++;
					}
				}
				else{
					if(fscanf(fp_,"%f %f %d %f %f %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore) != 6) break;
					if(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore) != 7) break;
				}
				count0++;
			}
			else{
				if(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore) != 7) break;
				if(fscanf(fp_,"%f %f\n",&row,&col) != 2) break;
			}
		}
		fclose(fp);
		fclose(fp_);
	}//*/
	std::cout<<"CCD间连接点数量："<<count<<std::endl;

	fclose(fp1);
	fclose(fp_lc);
	*base = count;
}

void Observation::SetObserveTxT_gridFI(char* matchfile,char* EOfile,char* IOtxt,char* xulie_ID1,char* xulie_ID2,char* observetxt,int* base){
	//int rows1=40000;
	//int rows2=80000;
	int cols=cfg_.cols;
	//读取内方位元素
	int CCD_num=cfg_.CCD_num;
	int TDI=128;
	int BIN=1;
	int count=0;
	float IO[10][10];
	if(!load_io_table(IOtxt, CCD_num, IO, "SetObserveTxT_gridFI")){
		return;
	}

	char matchtxt[2048];
	char matchtxt1[2048];
	FILE *fp1=fopen(observetxt,"w");
	if(fp1 == NULL){
		printf("[ERROR][SetObserveTxT_gridFI] 无法创建观测文件: %s\n", observetxt);
		return;
	}
	float xp,yp;
	double et;
	float EO[6];
	float GC[3];
	double Poly_C[30];
	double polyL[30], polyR[30];
	std::vector<double> beginETL(CCD_num, 0.0), LRL(CCD_num, 0.0);
	std::vector<double> beginETR(CCD_num, 0.0), LRR(CCD_num, 0.0);
	std::vector<char> eoL_ok(CCD_num, 0), eoR_ok(CCD_num, 0);
	{
		char ppath[2048], eopath[2048];
		snprintf(ppath, sizeof(ppath), "%s/%s/polyCC.txt", EOfile, xulie_ID1);
		snprintf(eopath, sizeof(eopath), "%s/%s/polyCC.txt", EOfile, xulie_ID2);
		if(!load_polyCC_cache(ppath, polyL) || !load_polyCC_cache(eopath, polyR)){
			printf("[ERROR][SetObserveTxT_gridFI] 无法预加载 polyCC: %s / %s\n", ppath, eopath);
			fclose(fp1);
			return;
		}
		for(int j=0;j<CCD_num;j++){
			snprintf(eopath, sizeof(eopath), "%s/%s/%s_RED%d_0.txt", EOfile, xulie_ID1, xulie_ID1, j);
			eoL_ok[j] = load_eo_begin_lr(eopath, &beginETL[j], &LRL[j]) ? 1 : 0;
			snprintf(eopath, sizeof(eopath), "%s/%s/%s_RED%d_0.txt", EOfile, xulie_ID2, xulie_ID2, j);
			eoR_ok[j] = load_eo_begin_lr(eopath, &beginETR[j], &LRR[j]) ? 1 : 0;
		}
		printf("[INFO][SetObserveTxT_gridFI] 已缓存 polyCC + 各 CCD EO 头 (beginET,LR)\n");
	}

	char LimgCoor[2048];
	snprintf(LimgCoor, sizeof(LimgCoor), "../data/observedata/%s_%s_LCgrid.txt", xulie_ID1, xulie_ID2);
	FILE *fp_lc=fopen(LimgCoor,"w");

	float H=-2200;
	const int ccd_begin = cfg_.ccd_begin();
	const int ccd_end = cfg_.ccd_end();
	//航带间
	for(int i=ccd_begin;i<ccd_end;i++){
		snprintf(matchtxt, sizeof(matchtxt), "%s/%s/downsample/0/%s_RED%d_match.txt", matchfile, xulie_ID1, xulie_ID1, i);
		char matchtxt_[2048];
		snprintf(matchtxt_, sizeof(matchtxt_), "%s/%s/downsample/0/%s_RED%d_match_.txt", matchfile, xulie_ID1, xulie_ID1, i);

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt_,"w");
		if(fp == NULL || fp_ == NULL){
			printf("[WARN][SetObserveTxT_gridFI] 跳过缺失匹配文件: %s\n", matchtxt);
			if(fp) fclose(fp);
			if(fp_) fclose(fp_);
			continue;
		}
		if(!eoL_ok[i]){
			printf("[WARN][SetObserveTxT_gridFI] 缺少左 CCD%d EO 头，跳过\n", i);
			fclose(fp); fclose(fp_);
			continue;
		}
		int bj,imgID;
		float row,col,mrow,mcol;
		float mscore;
		int count0=0;

		while(fscanf(fp,"%d ",&bj) == 1){
			if(bj==1){
				if(count0%2==0){
					if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore) != 6) break;
					if(col<cols-16 && imgID >= 0 && imgID < CCD_num && eoR_ok[imgID]){
						IO_correct(col, BIN, TDI, IO[i], &xp, &yp);
						et=beginETL[i]+row*LRL[i];
						memcpy(Poly_C, polyL, sizeof(polyL));
						Get_PolyEO(et,Poly_C,EO);
						Get_groundtruth(xp, yp, EO, 11995.48, 3396190+H, GC);
						fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);
						fprintf(fp_,"%d %f %f %f %d\n",count, row, col, GC[2], imgID); //生成地面匹配控制文件
						fprintf(fp_lc,"%d %f %f %d\n",count, row, col,i);

						IO_correct(mcol, BIN, TDI, IO[imgID], &xp, &yp);
						et=beginETR[imgID]+mrow*LRR[imgID];
						memcpy(Poly_C, polyR, sizeof(polyR));
						Get_PolyEO(et,Poly_C,EO);
						Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
						fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);

						count++;
					}
				}
				else{
					if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&imgID,&mrow,&mcol,&mscore) != 6) break;
				}
				count0++;
			}
			else{
				if(fscanf(fp,"%f %f\n",&row,&col) != 2) break;
			}
		}
		fclose(fp);
		fclose(fp_);
	}//*/


	//CCD间:左影像
	/*for(int i=0;i<CCD_num-1;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s%d%s", matchfile, "/", xulie_ID1, "/downsample/0/",xulie_ID1, "_RED", i,"_", i+1, "_intra_match.txt" );
		sprintf( matchtxt1, "%s%s%s%s%s%s%d%s%d%s", matchfile, "/", xulie_ID1, "/downsample/0/",xulie_ID1, "_RED", i,"_", i+1, "_intra__.txt" );

		double beginET,LR;
		int imgID,mimgID;
		float row,col,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt1,"r");
		while(!feof(fp) && !feof(fp_)){
			int bj;
			fscanf(fp_,"%d ",&bj);
			if(bj==1){
				FILE* fp_eo;
				double a1,a2,a3,a4,a5;

				//带间
				fscanf(fp_,"%f %f %d %f %f %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID2, "/",xulie_ID2, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID2, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);

				//读取CCD间数据
				fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID1, "/",xulie_ID1, "_RED", imgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+row*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID1, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

				//CCD间
				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID1, "/",xulie_ID1, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID1, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
				//fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

				count++;
			}
			else{
				fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
				fscanf(fp_,"%f %f\n",&row,&col);
			}
		}
		fclose(fp);
		fclose(fp_);
	}//*/


	//CCD间：右影像
	/*for(int i=0;i<10-1;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s%d%s", matchfile, "/", xulie_ID2, "/downsample/0/",xulie_ID2, "_RED", i,"_", i+1, "_intra_match.txt" );
		sprintf( matchtxt1, "%s%s%s%s%s%s%d%s%d%s", matchfile, "/", xulie_ID2, "/downsample/0/",xulie_ID2, "_RED", i,"_", i+1, "_intra__.txt" );

		double beginET,LR;
		int imgID,mimgID;
		float row,col,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt1,"r");
		while(!feof(fp) && !feof(fp_)){
			int bj;
			fscanf(fp_,"%d ",&bj);
			if(bj==1){
				FILE* fp_eo;
				double a1,a2,a3,a4,a5;

				//带间
				fscanf(fp_,"%f %f %d %f %f %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID1, "/",xulie_ID1, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID1, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

				//读取CCD间数据
				fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID2, "/",xulie_ID2, "_RED", imgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+row*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID2, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);

				//CCD间
				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID2, "/",xulie_ID2, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID2, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC);
				//fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);

				count++;
			}
			else{
				fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
				fscanf(fp_,"%f %f\n",&row,&col);
			}
		}
		fclose(fp);
		fclose(fp_);
	}//*/

	fclose(fp1);
	fclose(fp_lc);
	*base = count;
}

void Observation::SetObserveTxT2(char* matchfile,char* EOfile,char* IOtxt,char* xulie_ID1,char* xulie_ID2,char* observetxt,int* base){
	GDALAllRegister();
	//定义变量
	GDALDataType Type0 = GDT_Float32;
	GDALDataset *poDataset;
	GDALRasterBand *poBand;
	int nXSize1,nYSize1,nXSize2,nYSize2,nBands;
	float *paf1,*paf2;
	uchar* data1,*data2;

	int rows1=cfg_.rows1;
	int rows2=cfg_.rows2;
	//读取内方位元素
	int CCD_num=cfg_.CCD_num;
	int TDI=128;
	int BIN=1;
	int count=0;
	float IO[10][10];
	if(!load_io_table(IOtxt, 10, IO, "SetObserveTxT2")){
		return;
	}

	FILE *fp1=fopen(observetxt,"w");
	if(fp1 == NULL){
		printf("[ERROR][SetObserveTxT2] 无法创建观测文件: %s\n", observetxt);
		return;
	}
	float xp,yp;
	double et;
	float EO[6];
	float GC[3];
	double Poly_C[30];
	double polyL[30], polyR[30];
	std::vector<double> beginETL(CCD_num, 0.0), LRL(CCD_num, 0.0);
	std::vector<double> beginETR(CCD_num, 0.0), LRR(CCD_num, 0.0);
	std::vector<char> eoL_ok(CCD_num, 0), eoR_ok(CCD_num, 0);
	{
		char ppath[2048], eopath[2048];
		snprintf(ppath, sizeof(ppath), "%s/%s/polyCC.txt", EOfile, xulie_ID1);
		snprintf(eopath, sizeof(eopath), "%s/%s/polyCC.txt", EOfile, xulie_ID2);
		if(!load_polyCC_cache(ppath, polyL) || !load_polyCC_cache(eopath, polyR)){
			printf("[ERROR][SetObserveTxT2] 无法预加载 polyCC: %s / %s\n", ppath, eopath);
			fclose(fp1);
			return;
		}
		for(int j=0;j<CCD_num;j++){
			snprintf(eopath, sizeof(eopath), "%s/%s/%s_RED%d_0.txt", EOfile, xulie_ID1, xulie_ID1, j);
			eoL_ok[j] = load_eo_begin_lr(eopath, &beginETL[j], &LRL[j]) ? 1 : 0;
			snprintf(eopath, sizeof(eopath), "%s/%s/%s_RED%d_0.txt", EOfile, xulie_ID2, xulie_ID2, j);
			eoR_ok[j] = load_eo_begin_lr(eopath, &beginETR[j], &LRR[j]) ? 1 : 0;
		}
	}

	//航带间
	for(int i=0;i<CCD_num;i++){
		//读取视差图
		char imgpara_path[2048];
		snprintf(imgpara_path, sizeof(imgpara_path), "%s/%s/downsample/0/%s_RED%d_dense.tif", matchfile, xulie_ID1, xulie_ID1, i);
		poDataset = (GDALDataset*) GDALOpen(imgpara_path, GA_ReadOnly);
		if( poDataset == NULL )
		{
			printf( "File1: %s不能打开！\n",imgpara_path);
			return;
		}
		if(poDataset->GetRasterCount()<1){
			printf("视差影像波段小于1！\n");
			return;
		}
		nXSize1 = poDataset->GetRasterBand(1)->GetXSize();
		nYSize1 = poDataset->GetRasterBand(1)->GetYSize();
		nBands = poDataset->GetRasterCount();
		Type0 = poDataset->GetRasterBand(1)->GetRasterDataType();

		double beginET,LR;
		int bj,row,col,imgID,mrow,mcol;
		float mscore;
		float temp;
		int count0=0;

		for(int ii=0;ii<nYSize1;ii+=40){
			for(int jj=0;jj<nXSize1;jj+=40){
				row=ii;col=jj;
				if(poDataset->GetRasterBand(4)->RasterIO(GF_Read, jj, ii, 1, 1,
					&mscore, 1, 1, Type0, 0, 0 ) != CE_None) continue;
				if(poDataset->GetRasterBand(2)->RasterIO(GF_Read, jj, ii, 1, 1,
					&temp, 1, 1, Type0, 0, 0 ) != CE_None) continue;
				mrow=int(temp);
				if(poDataset->GetRasterBand(1)->RasterIO(GF_Read, jj, ii, 1, 1,
					&temp, 1, 1, Type0, 0, 0 ) != CE_None) continue;
				imgID=int(temp);
				if(mscore>0 && imgID >= 0 && imgID < CCD_num && eoL_ok[i] && eoR_ok[imgID]){

					if(poDataset->GetRasterBand(3)->RasterIO(GF_Read, jj, ii, 1, 1,
					&temp, 1, 1, Type0, 0, 0 ) != CE_None) continue;
					mcol=int(temp);

					IO_correct(col, BIN, TDI, IO[i], &xp, &yp);
					et=beginETL[i]+row*LRL[i];
					memcpy(Poly_C, polyL, sizeof(polyL));
					Get_PolyEO(et,Poly_C,EO);
					ImageMatch().Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
					fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

					IO_correct(mcol, BIN, TDI, IO[imgID], &xp, &yp);
					et=beginETR[imgID]+mrow*LRR[imgID];
					memcpy(Poly_C, polyR, sizeof(polyR));
					Get_PolyEO(et,Poly_C,EO);
					ImageMatch().Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
					fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);
					count++;
				}
			}
		}
		GDALClose(poDataset);
	}


	//CCD间:左影像
	/*for(int i=0;i<CCD_num-1;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s%d%s", matchfile, "/", xulie_ID1, "/downsample/0/",xulie_ID1, "_RED", i,"_", i+1, "_intra.txt" );
		sprintf( matchtxt1, "%s%s%s%s%s%s%d%s%d%s", matchfile, "/", xulie_ID1, "/downsample/0/",xulie_ID1, "_RED", i,"_", i+1, "_intra__.txt" );

		double beginET,LR;
		int imgID,row,col,mimgID,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt1,"r");
		while(!feof(fp) && !feof(fp_)){
			int bj;
			fscanf(fp_,"%d ",&bj);
			if(bj==1){
				FILE* fp_eo;
				double a1,a2,a3,a4,a5;

				//带间
				fscanf(fp_,"%d %d %d %d %d %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID2, "/",xulie_ID2, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID2, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				ImageMatch().Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);

				//读取CCD间数据
				fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID1, "/",xulie_ID1, "_RED", imgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+row*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID1, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				ImageMatch().Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

				//CCD间
				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID1, "/",xulie_ID1, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID1, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				ImageMatch().Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

				count++;
			}
			else{
				fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
				fscanf(fp_,"%d %d\n",&row,&col);
			}
		}
		fclose(fp);
		fclose(fp_);
	}//*/


	//CCD间：右影像
	/*for(int i=0;i<CCD_num-1;i++){
		sprintf( matchtxt, "%s%s%s%s%s%s%d%s%d%s", matchfile, "/", xulie_ID2, "/downsample/0/",xulie_ID2, "_RED", i,"_", i+1, "_intra.txt" );
		sprintf( matchtxt1, "%s%s%s%s%s%s%d%s%d%s", matchfile, "/", xulie_ID2, "/downsample/0/",xulie_ID2, "_RED", i,"_", i+1, "_intra__.txt" );

		double beginET,LR;
		int imgID,row,col,mimgID,mrow,mcol;
		float mscore;

		FILE *fp=fopen(matchtxt,"r");
		FILE *fp_=fopen(matchtxt1,"r");
		while(!feof(fp) && !feof(fp_)){
			int bj;
			fscanf(fp_,"%d ",&bj);
			if(bj==1){
				FILE* fp_eo;
				double a1,a2,a3,a4,a5;

				//带间
				fscanf(fp_,"%d %d %d %d %d %f\n",&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID1, "/",xulie_ID1, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID1, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				ImageMatch().Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 0, GC[0], GC[1], GC[2]);

				//读取CCD间数据
				fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
				IO_correct(col, BIN, TDI, IO[imgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID2, "/",xulie_ID2, "_RED", imgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+row*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID2, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				ImageMatch().Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);

				//CCD间
				IO_correct(mcol, BIN, TDI, IO[mimgID], &xp, &yp);
				sprintf( EOtxt, "%s%s%s%s%s%s%d%s", EOfile, "/", xulie_ID2, "/",xulie_ID2, "_RED", mimgID, "_0.txt" );
				fp_eo=fopen(EOtxt,"r");
				fscanf(fp_eo,"%lf %lf\n",&beginET,&LR);
				fclose(fp_eo);
				et=beginET+mrow*LR;

				sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID2, "/polyCC.txt" );
				fp_eo=fopen(EOtxt,"r");
				for(int ii=0;ii<6;ii++){
					fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5);
					Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
					Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
				}
				fclose(fp_eo);
				Get_PolyEO(et,Poly_C,EO);
				ImageMatch().Get_groundtruth1(xp, yp, EO, 11995.48, 3396190, GC);
				fprintf(fp1,"%d %f %f %lf %d %f %f %f\n",count, xp, yp, et, 1, GC[0], GC[1], GC[2]);

				count++;
			}
			else{
				fscanf(fp,"%d %d %d %d %d %d %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&mscore);
				fscanf(fp_,"%d %d\n",&row,&col);
			}
		}
		fclose(fp);
		fclose(fp_);
	}//*/

	fclose(fp1);
	*base = count;
}

void Observation::GetImageGroundRange(char* EOfile,char* IOtxt,char* xulie_ID1,char* outfile,int rows, int cols, float H){
	//int rows1=40000;
	//int rows2=80000;
	//读取内方位元素
	int CCD_num=cfg_.CCD_num;
	int TDI=128;
	int BIN=1;
	int count=0;

	// Intrinsic Parameters
	cout<<IOtxt<<endl;
	float IO[10][10];
	if(!load_io_table(IOtxt, 10, IO, "GetImageGroundRange")){
		return;
	}

	// ET and LR
	char EOtxt[2048];
	char ET_LR[2048];
	snprintf(ET_LR, sizeof(ET_LR), "%s/%s/ET_LR.txt", EOfile, xulie_ID1);
	FILE* fp_etlr=fopen(ET_LR,"w");
	for(int i=0;i<CCD_num;i++){
		snprintf(EOtxt, sizeof(EOtxt), "%s/%s/%s_RED%d_0.txt", EOfile, xulie_ID1, xulie_ID1, i);
		FILE* fp_eo_t=fopen(EOtxt,"r");
		double temp1,temp2;
		if(fp_eo_t && fscanf(fp_eo_t,"%lf %lf\n",&temp1,&temp2) == 2){
			fprintf(fp_etlr,"%.12lf %.12lf\n",temp1,temp2);
		}
		if(fp_eo_t) fclose(fp_eo_t);
	}
	fclose(fp_etlr);

	std::vector<double> beginET(CCD_num, 0.0);
	std::vector<double> LR(CCD_num, 0.0);
	snprintf(EOtxt, sizeof(EOtxt), "%s/%s/ET_LR.txt", EOfile, xulie_ID1);
	FILE* fp_eo=fopen(EOtxt,"r");
	for(int i=0;i<CCD_num;i++){
		if(fp_eo == NULL || fscanf(fp_eo,"%lf %lf\n",&(beginET[i]),&(LR[i])) != 2) break;
	}
	if(fp_eo) fclose(fp_eo);

	// PolyCC
	double Poly_C[30];
	snprintf(EOtxt, sizeof(EOtxt), "%s/%s/polyCC.txt", EOfile, xulie_ID1);
	fp_eo=fopen(EOtxt,"r");
	double a1,a2,a3,a4,a5;
	for(int ii=0;ii<6;ii++){
		if(fp_eo == NULL || fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5) != 5) break;
		Poly_C[ii*5+0]=a1;Poly_C[ii*5+1]=a2;Poly_C[ii*5+2]=a3;
		Poly_C[ii*5+3]=a4;Poly_C[ii*5+4]=a5;
	}
	if(fp_eo) fclose(fp_eo);

	float xp,yp;
	double et;
	float EO[6];
	float GC[3];
	// float H=-2200;

	int CornerR[4] = {0,rows-1,0,rows-1};
	int CornerC[4] = {0,0,cols-1,cols-1};

	//CCD0:(0,0) (rows-1,0)
	//CCDn:(0,cols-1) (rows-1,cols-1)
	FILE* fp_out=fopen(outfile,"w");
	double E,N,HH;
	for(int i=0;i<4;i++){
		et=beginET[(CCD_num-1)*(i/2)]+CornerR[i]*LR[(CCD_num-1)*(i/2)];
		Get_PolyEO(et,Poly_C,EO);
		IO_correct(CornerC[i], BIN, TDI, IO[i], &xp, &yp);
		Get_groundtruth(xp, yp, EO, 11995.48, 3396190+H, GC);
		ConstSpiceDouble rectan[3]={GC[0],GC[1],GC[2]};
		recgeo_c(rectan,3396190,0,&E,&N,&HH);
		cout<<"spice func:"<<E<<" "<<N<<" "<<HH<<endl;
		double rectan0[3]={GC[0],GC[1],GC[2]};
		rec2geo(rectan0,3396190.0,0.0,&E,&N,&HH);
		cout<<"my func: "<<E<<" "<<N<<" "<<HH<<endl;
		double r=(90.0-N*180/3.1425926)/180.0*53347;
		double c=(E*180/3.1425926+180.0)/360.0*106694;
		fprintf(fp_out,"%d %.6lf %.6lf %.6lf %d %d\n",i,E*180/3.1415926,N*180/3.1415926,HH,int(r+0.5),int(c+0.5));
	}
	fclose(fp_out);
}

void Observation::Obs_main(){
	char* srcfilepath = const_cast<char*>(cfg_.srcfilepath);
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int rows1 = cfg_.rows1;
	int rows2 = cfg_.rows2;
	int CCD_num=cfg_.CCD_num;
	int cols1=cfg_.cols;int cols2=cfg_.cols;
	int cols=cfg_.cols;
	//序列影像 pds 转半幅 tif
	cout<<"xulie_pds2tif!"<<endl;
	ImageProcess IP;
	IP.xulie_process(cfg_);/*//*/

	//CCD 内拼接
	cout<<"intra_ccd_mosaic!"<<endl;
	IP.intra_ccd_mosaic(cfg_);/*//*/

	//序列影像降采样（2，4，8，16）
	cout<<"xulie_downsample!"<<endl;
	xulie_downsample();/*//*/

	//原始分辨率拼接&输出每级拼接参数及影像到downsample文件夹
	cout<<"xulie_mosaic!"<<endl;
	prepare_feature_match_mosaic();/*//*/

	//序列影像分层匹配（分辨率由低到高）
	/*int mark=0;
	float* fs_c = new float[6];
	if(mark==0){
		fenfu_match1(filepath,xulie_ID1,xulie_ID2,rows1,rows2,fs_c);
	}
	else if(mark==1){
		Compute_fsc(filepath,xulie_ID1,xulie_ID2,80000,80000,fs_c);
		//fs_c[0]=1;fs_c[1]=0;fs_c[2]=0;fs_c[3]=0;fs_c[4]=1;fs_c[5]=0;
		fenfu_match1(filepath,xulie_ID1,xulie_ID2,rows1,rows2,fs_c,mark);
	}
	*/
}


//生成核线影像
void Observation::EpipolarImage(){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	ImageMatch IM;
	//以CCD5影像为例
	int CCDid=5;
	//先把坐标系转到局部表面坐标系LVCS
	//计算局部坐标系
	char* matchfile = filepath;
	char EOfile[] = "../data/EO";
	char IOtxt[] = "../data/IO/IO.txt";

	char EOtxt[2048];

	int CCD_num=cfg_.CCD_num;
	int TDI=128;
	int BIN=1;
	int count=0;

	//匹配对
	float p1[2] = {20000.000000f, 1056.000000f};
	float p2[2] = {40744.000000f, 689.000000f};

	//内方位元素：左右影像相同
	float IO[10][10];
	if(!load_io_table(IOtxt, 10, IO, "EpipolarImage")){
		return;
	}

	//外方位元素：左右影像不同（用拟合参数）
	double beginET[2]={219232467.954159080000,217456589.837160530000};
	double LR[2]={0.000082187500,0.000084437500};

	double Poly_C[2][30];
	for(int i=0;i<2;i++){
		char* xulie_ID;
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else{
			xulie_ID=xulie_ID2;
		}
		snprintf(EOtxt, sizeof(EOtxt), "%s/%s/polyCC.txt", EOfile, xulie_ID);
		FILE* fp_eo=fopen(EOtxt,"r");
		double a1,a2,a3,a4,a5;
		for(int ii=0;ii<6;ii++){
			if(fp_eo == NULL || fscanf(fp_eo,"%lf %lf %lf %lf %lf\n",&a1,&a2,&a3,&a4,&a5) != 5) break;
			Poly_C[i][ii*5+0]=a1;Poly_C[i][ii*5+1]=a2;Poly_C[i][ii*5+2]=a3;
			Poly_C[i][ii*5+3]=a4;Poly_C[i][ii*5+4]=a5;
		}
		if(fp_eo) fclose(fp_eo);
	}

	//LVCS坐标系转换矩阵计算（以影像1的中间采样时刻像主点成像地面点为中心）
	double et;
	et=beginET[0]+20000*LR[0];
	float EO[6];
	Get_PolyEO(et,Poly_C[0],EO);

	float xp,yp;
	float GC0[3];
	IO_correct(1024, BIN, TDI, IO[CCDid], &xp, &yp);
	Get_groundtruth(xp, yp, EO, 11995.48, 3396190, GC0);


	//计算坐标系旋转矩阵
	const double pi = acos(-1.0);
	double X0=GC0[0];double Y0=GC0[1];double Z0=GC0[2];
	double Lat=asin(Z0/sqrt(X0*X0+Y0*Y0+Z0*Z0));
	double Lon=atan(Y0/X0);
	if(X0<0){
		Lon = Lon + pi;
	}

	float dR[9]={static_cast<float>(-sin(Lon))         , static_cast<float>(cos(Lon)),       0,
		          static_cast<float>(-cos(Lon)*sin(Lat)), static_cast<float>(-sin(Lon)*sin(Lat)), static_cast<float>(cos(Lat)),
		          static_cast<float>( cos(Lon)*cos(Lat)), static_cast<float>( sin(Lon)*cos(Lat)), static_cast<float>(sin(Lat))};
	float* dR0=dR;
	MatrixXf dR_ = (Map<MatrixXf>(dR0,3,3)).transpose();

	//匹配对投影到Z=0平面（LVCS）
	float GC[3];
	IO_correct(p1[1], BIN, TDI, IO[CCDid], &xp, &yp);
	et=beginET[0]+p1[0]*LR[0];
	Get_PolyEO(et,Poly_C[0],EO);
	Get_groundtruth(xp, yp, EO, dR_, GC0, 11995.48, 3396190, GC);


	int imgC[2];
	G2I(GC,IO[CCDid],Poly_C[0],imgC,beginET[0],LR[0],dR_,GC0);
	//用拟合系数计算外方位元素并投影地面点，再反投影到影像二
	cout<<GC0[0]<<" "<<GC0[1]<<" "<<GC0[2]<<" "<<imgC[0]<<" "<<imgC[1]<<endl;
}


//其他函数
//XYZ转为NEH
void Observation::GetLocalNEH(char* ProjFile,char* XYZfile,char* NEHfile){
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
	if(fp_gt == NULL || fp_neh == NULL){
		printf("[ERROR][GetLocalNEH] 无法打开输入或输出文件: %s -> %s\n", XYZfile, NEHfile);
		if(fp_gt) fclose(fp_gt);
		if(fp_neh) fclose(fp_neh);
		GDALClose(poDataset);
		return;
	}
	if(fscanf(fp_gt,"%d\n",&ground_num) != 1){
		printf("[ERROR][GetLocalNEH] 地面点数量读取失败: %s\n", XYZfile);
		fclose(fp_gt);
		fclose(fp_neh);
		GDALClose(poDataset);
		return;
	}
	fprintf(fp_neh,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		if(fscanf(fp_gt,"%lf %lf %lf %lf\n",&X,&Y,&Z,&res) != 4){
			printf("[WARN][GetLocalNEH] 地面点行读取失败: %s line=%d\n", XYZfile, i + 2);
			break;
		}
		My_rec2NEH(projRef,double(X),double(Y),double(Z),&N,&E,&H);
		fprintf(fp_neh,"%lf %lf %lf %lf\n",N,E,H,res);
	}
	fclose(fp_gt);
	fclose(fp_neh);
	GDALClose(poDataset);
}
//匹配结果显示
void Observation::Match_Result_Show(char* featurepoint1,int j,float* fs_c,int mark){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int CCD_num=cfg_.CCD_num;
	//读取拼接系数
	std::vector<int> mosaic_c1(CCD_num * 4);
	std::vector<int> mosaic_c2(CCD_num * 4);
	if(!load_mosaic_coefficients(filepath, xulie_ID1, CCD_num, mosaic_c1.data()) ||
	   !load_mosaic_coefficients(filepath, xulie_ID2, CCD_num, mosaic_c2.data())){
		return;
	}

	ImageMatch IM;


	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
	FILE *fp=fopen(featurepoint1,"r");
	if(fp==NULL){
		cout<<"featurepoint1 file open error!"<<endl;
		return;
	}
	if(mark==0){
		int bj,mimgID;float row,col,mrow,mcol;
		float m_score;
		double sfr=pow(double(2),double(0))/pow(double(2),double(4));
		while(fscanf(fp,"%d ",&bj) == 1){
			if(bj==1){
				if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&mimgID,&mrow,&mcol,&m_score) != 6){
					break;
				}
				KeyPoint_x1.push_back(int(row+mosaic_c1[j*4+0]));
				KeyPoint_y1.push_back(int(col+mosaic_c1[j*4+2]));
				KeyPoint_x2.push_back(int(mrow+mosaic_c2[mimgID*4+0]));
				KeyPoint_y2.push_back(int(mcol+mosaic_c2[mimgID*4+2]));
				KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
				KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
				KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[mimgID*4+0])*sfr));
				KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[mimgID*4+2])*sfr));
			}
			else{
				if(fscanf(fp,"%f %f\n",&row,&col) != 2){
					break;
				}
			}
		}
		char imgL_path1[2048];
		snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
		char imgR_path1[2048];
		snprintf(imgR_path1, sizeof(imgR_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID2);
		ImageMatch().drawMatch2(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,1);
	}
	else if(mark==1){
		int imgID,mimgID;float row,col,mrow,mcol;
		float m_score;
		double sfr=pow(double(2),double(0))/pow(double(2),double(4));
		while(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&m_score) == 7){
			KeyPoint_x1.push_back(int(row+mosaic_c1[imgID*4+0]));
			KeyPoint_y1.push_back(int(col+mosaic_c1[imgID*4+2]));
			KeyPoint_x2.push_back(int(mrow+mosaic_c2[mimgID*4+0]));
			KeyPoint_y2.push_back(int(mcol+mosaic_c2[mimgID*4+2]));
			KeyPoint_x11.push_back(int(double(row+mosaic_c1[imgID*4+0])*sfr));
			KeyPoint_y11.push_back(int(double(col+mosaic_c1[imgID*4+2])*sfr));
			KeyPoint_x22.push_back(int(double(mrow+mosaic_c1[mimgID*4+0])*sfr));
			KeyPoint_y22.push_back(int(double(mcol+mosaic_c1[mimgID*4+2])*sfr));
		}

		char imgL_path1[2048];
		snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
		char imgR_path1[2048];
		snprintf(imgR_path1, sizeof(imgR_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
		ImageMatch().drawMatch2(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,1);
	}
	else if(mark==2){
		int imgID,mimgID;float row,col,mrow,mcol;
		float m_score;
		double sfr=pow(double(2),double(0))/pow(double(2),double(4));
		while(fscanf(fp,"%d %f %f %d %f %f %f\n",&imgID,&row,&col,&mimgID,&mrow,&mcol,&m_score) == 7){
			KeyPoint_x1.push_back(int(row));
			KeyPoint_y1.push_back(int(col));
			KeyPoint_x2.push_back(int(mrow));
			KeyPoint_y2.push_back(int(mcol));
			KeyPoint_x11.push_back(int(double(row+mosaic_c1[imgID*4+0])*sfr));
			KeyPoint_y11.push_back(int(double(col+mosaic_c1[imgID*4+2])*sfr));
			KeyPoint_x22.push_back(int(double(mrow+mosaic_c1[mimgID*4+0])*sfr));
			KeyPoint_y22.push_back(int(double(mcol+mosaic_c1[mimgID*4+2])*sfr));
		}

		char imgL_path1[2048];
		snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j);
		char imgR_path1[2048];
		snprintf(imgR_path1, sizeof(imgR_path1), "%s/%s/downsample/0/%s_RED%d.tif", filepath, xulie_ID1, xulie_ID1, j+1);
		IM.drawMatch2(imgL_path1,imgR_path1,KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2,4);
	}
	else{
		ImageMatch IM;
		int bj,mimgID;float row,col,mrow,mcol;
		float m_score;
		int count=0;
		double sfr=pow(double(2),double(0))/pow(double(2),double(4));
		while(fscanf(fp,"%d ",&bj) == 1){
			if(bj==1){
				if(fscanf(fp,"%f %f %d %f %f %f\n",&row,&col,&mimgID,&mrow,&mcol,&m_score) != 6){
					break;
				}
				if(count%100==0){
					KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
					KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
					//KeyPoint_x22.push_back(int(double(mrow+mosaic_c2[mimgID*4+0])*sfr));
					//KeyPoint_y22.push_back(int(double(mcol+mosaic_c2[mimgID*4+2])*sfr));
					KeyPoint_x22.push_back(int(double(fs_c[2]+fs_c[0]*(row+mosaic_c1[j*4+0])+fs_c[1]*(col+mosaic_c1[j*4+2]))*sfr));
					KeyPoint_y22.push_back(int(double(fs_c[5]+fs_c[3]*(row+mosaic_c1[j*4+0])+fs_c[4]*(col+mosaic_c1[j*4+2]))*sfr));
				}
			}
			else{
				if(fscanf(fp,"%f %f\n",&row,&col) != 2){
					break;
				}
				if(count%100==0){
					KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
					KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
					KeyPoint_x22.push_back(int(double(fs_c[2]+fs_c[0]*(row+mosaic_c1[j*4+0])+fs_c[1]*(col+mosaic_c1[j*4+2]))*sfr));
					KeyPoint_y22.push_back(int(double(fs_c[5]+fs_c[3]*(row+mosaic_c1[j*4+0])+fs_c[4]*(col+mosaic_c1[j*4+2]))*sfr));
				}
			}
			count++;
		}
		char imgL_path1[2048];
		snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
		char imgR_path1[2048];
		snprintf(imgR_path1, sizeof(imgR_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID2);
		IM.drawMatch2(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,1);
	}
	fclose(fp);
}

void Observation::Draw_FeaturePoint_betweenCCD(int j,int seq_index){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID = const_cast<char*>(seq_index == 0 ? cfg_.xulie_ID1 : cfg_.xulie_ID2);
	int CCD_num = cfg_.CCD_num;
	if(j < 0 || j >= CCD_num - 1){
		printf("[WARN][CCD间绘制] CCD 对索引越界: %d\n", j);
		return;
	}

	std::vector<int> mosaic_c(CCD_num * 4);
	if(!load_mosaic_coefficients(filepath, xulie_ID, CCD_num, mosaic_c.data())){
		return;
	}

	char featurepoint[2048];
	snprintf(featurepoint, sizeof(featurepoint), "%s/%s/downsample/0/%s_RED%d_%d_intra_match.txt",
		filepath, xulie_ID, xulie_ID, j, j + 1);
	std::vector<int> keypoint_rows_ds4;
	std::vector<int> keypoint_cols_ds4;
	const int matched_count = append_inter_ccd_match_points(featurepoint, j, mosaic_c.data(), keypoint_rows_ds4, keypoint_cols_ds4);
	if(matched_count <= 0){
		return;
	}

	char mosaic_ds4_path[2048];
	snprintf(mosaic_ds4_path, sizeof(mosaic_ds4_path), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID);
	char outpath[2048];
	snprintf(outpath, sizeof(outpath), "%s/%s/downsample/0/%s_RED%d_%d_intra_match_ds4.tif",
		filepath, xulie_ID, xulie_ID, j, j + 1);

	ImageMatch IM;
	IM.draw_fea(mosaic_ds4_path, keypoint_rows_ds4, keypoint_cols_ds4, 1, outpath);
	printf("[INFO][CCD间绘制] 已输出 %s, 匹配点 %d\n", outpath, matched_count);
}

void Observation::Draw_InterCCDMatch_OnMosaic(int seq_index){
	(void)seq_index;
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int CCD_num = cfg_.CCD_num;

	std::vector<int> mosaic_c1(CCD_num * 4);
	std::vector<int> mosaic_c2(CCD_num * 4);
	if(!load_mosaic_coefficients(filepath, xulie_ID1, CCD_num, mosaic_c1.data()) ||
	   !load_mosaic_coefficients(filepath, xulie_ID2, CCD_num, mosaic_c2.data())){
		return;
	}

	ImageMatch IM;
	std::vector<int> keypoint_x1_ds4;
	std::vector<int> keypoint_y1_ds4;
	std::vector<int> keypoint_x2_ds4;
	std::vector<int> keypoint_y2_ds4;
	int total_match_count = 0;
	char imgL_path1[2048];
	char imgR_path1[2048];
	snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
	snprintf(imgR_path1, sizeof(imgR_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID2);

	for(int j=cfg_.ccd_begin(); j<cfg_.ccd_end()-1; ++j){
		char featurepoint[2048];
		snprintf(featurepoint, sizeof(featurepoint), "%s/%s/downsample/0/%s_RED%d_%d_intra__.txt",
			filepath, xulie_ID1, xulie_ID1, j, j + 1);

		const int matched_count = append_inter_seq_match_pairs(featurepoint, j, mosaic_c1.data(), mosaic_c2.data(), CCD_num,
			keypoint_x1_ds4, keypoint_y1_ds4, keypoint_x2_ds4, keypoint_y2_ds4);
		if(matched_count <= 0){
			continue;
		}
		total_match_count += matched_count;
	}

	if(!keypoint_x1_ds4.empty()){
		const int draw_stride = 10;
		std::vector<int> sampled_x1_ds4;
		std::vector<int> sampled_y1_ds4;
		std::vector<int> sampled_x2_ds4;
		std::vector<int> sampled_y2_ds4;
		for(size_t i=0; i<keypoint_x1_ds4.size(); i+=draw_stride){
			sampled_x1_ds4.push_back(keypoint_x1_ds4[i]);
			sampled_y1_ds4.push_back(keypoint_y1_ds4[i]);
			sampled_x2_ds4.push_back(keypoint_x2_ds4[i]);
			sampled_y2_ds4.push_back(keypoint_y2_ds4[i]);
		}

		std::vector<int> match(sampled_x1_ds4.size(), 1);
		char outpath[2048];
		snprintf(outpath, sizeof(outpath), "../out/intra__.tif");
		IM.drawMatch3(imgL_path1, imgR_path1, outpath,
			sampled_x1_ds4, sampled_y1_ds4, sampled_x2_ds4, sampled_y2_ds4, match.data(), 1);
		printf("[INFO][CCD间绘制] 已输出 %s, 来源 %s 的 Intra__, 总匹配点 %d, 绘制点 %zu (步长=%d)\n",
			outpath, xulie_ID1, total_match_count, sampled_x1_ds4.size(), draw_stride);
	}
	else{
		printf("[WARN][CCD间绘制] 未找到可绘制的 Intra__ 结果: %s\n", xulie_ID1);
	}
}

void Observation::Feature_Show(char* featurepoint1,int j){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	int CCD_num=cfg_.CCD_num;
	//读取拼接系数
	std::vector<int> mosaic_c1(CCD_num * 4);
	if(!load_mosaic_coefficients(filepath, xulie_ID1, CCD_num, mosaic_c1.data())){
		cout<<"mosaic file not exist!"<<endl;
		return;
	}

	ImageMatch IM;

	std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
	std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
	FILE *fp=fopen(featurepoint1,"r");
	if(fp==NULL){
		cout<<"feature file not exist!"<<endl;
		return;
	}

	int bj,row,col,mimgID,mrow,mcol;
	float m_score;
	double sfr=pow(double(2),double(0))/pow(double(2),double(4));
	while(fscanf(fp,"%d ",&bj) == 1){
		if(bj==1){
			if(fscanf(fp,"%d %d %d %d %d %f\n",&row,&col,&mimgID,&mrow,&mcol,&m_score) != 6){
				break;
			}
			KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
			KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
		}
		else{
			if(fscanf(fp,"%d %d\n",&row,&col) != 2){
				break;
			}
			KeyPoint_x11.push_back(int(double(row+mosaic_c1[j*4+0])*sfr));
			KeyPoint_y11.push_back(int(double(col+mosaic_c1[j*4+2])*sfr));
		}
	}
	char imgL_path1[2048];
	snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
	IM.draw_fea(imgL_path1,KeyPoint_x11,KeyPoint_y11,1);

	fclose(fp);
}

void Observation::Res_Show(){
	char* filepath = const_cast<char*>(cfg_.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg_.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg_.xulie_ID2);
	int CCD_num=cfg_.CCD_num;
	//读取拼接系数
	std::vector<int> mosaic_c1(CCD_num * 4);
	if(!load_mosaic_coefficients(filepath, xulie_ID1, CCD_num, mosaic_c1.data())){
		cout<<"mosaic file not exist!"<<endl;
		return;
	}

	ImageMatch IM;

	std::vector<int> KeyPoint_x11,KeyPoint_y11;
	std::vector<double> resx,resy;
	char featurepoint1[2048];
	snprintf(featurepoint1, sizeof(featurepoint1), "../data/result/%s_%s_res.txt", xulie_ID1, xulie_ID2);
	FILE *fp=fopen(featurepoint1,"r");
	if(fp==NULL){
		cout<<"res file not exist!"<<endl;
		return;
	}
	int imgID;
	double row,col,resxt,resyt;
	double sfr=pow(double(2),double(0))/pow(double(2),double(4));
	while(fscanf(fp,"%lf %lf %lf %lf %d\n",&row,&col,&resxt,&resyt,&imgID) == 5){
		KeyPoint_x11.push_back(int(double(row+mosaic_c1[imgID*4+0])*sfr));
		KeyPoint_y11.push_back(int(double(col+mosaic_c1[imgID*4+2])*sfr));
		resx.push_back(resxt*sfr);
		resy.push_back(resyt*sfr);
	}
	char imgL_path1[2048];
	snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/0/mosaic_ds4.tif", filepath, xulie_ID1);
	IM.draw_res(imgL_path1,KeyPoint_x11,KeyPoint_y11,resx,resy,1);

	fclose(fp);
}
