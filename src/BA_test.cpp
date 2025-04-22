// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "ceres/rotation.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

template <typename T>
//phi，w，k旋转
void My_AngleAxisRotatePoint(T camera[3],T point[3],T p[3]){
	T phi=camera[0];T w=camera[1];T k=camera[2];
	T R_inv[9];

	R_inv[0]=cos(phi)*cos(k)-sin(phi)*sin(w)*sin(k);
	R_inv[3]=-cos(phi)*sin(k)-sin(phi)*sin(w)*cos(k);
	R_inv[3]=-R_inv[3];
	R_inv[6]=-sin(phi)*cos(w);

	R_inv[1]=cos(w)*sin(k);
	R_inv[1]=-R_inv[1];
	R_inv[4]=cos(w)*cos(k);
	R_inv[7]=-sin(w);
	R_inv[7]=-R_inv[7];

	R_inv[2]=sin(phi)*cos(k)+cos(phi)*sin(w)*sin(k);
	R_inv[5]=-sin(phi)*sin(k)+cos(phi)*sin(w)*cos(k);
	R_inv[5]=-R_inv[5];
	R_inv[8]=cos(phi)*cos(w);

	for(int i=0;i<3;i++){
		p[i]=T(0);
		for(int j=0;j<3;j++){
			p[i] += R_inv[i*3+j]*point[j];
		}
	}
}

/*
struct Mycost {
	Mycost(double observed_x, double observed_y, double et, double* EO_0, double* Poly_CR,double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), EO_0(EO_0), Poly_CR(Poly_CR), focal(focal){}

	template <typename T>
	bool operator()(const T* const Poly_CT,
		const T* const point,
		T* residuals) const {

			T Poly_C[18];
			for(int i=0;i<6;i++){
				if(i<3){
					Poly_C[i*3+0]=T(Poly_CT[i*3+0]);
					Poly_C[i*3+1]=T(Poly_CT[i*3+1]);
					Poly_C[i*3+2]=T(Poly_CT[i*3+2]);
				}
				else{
					Poly_C[i*3+0]=T(Poly_CR[(i-3)*3+0]);
					Poly_C[i*3+1]=T(Poly_CR[(i-3)*3+1]);
					Poly_C[i*3+2]=T(Poly_CR[(i-3)*3+2]);
				}
			}

			T camera_R[3],camera_T[3];
			for(int i=0;i<6;i++){
				T cam_temp=EO_0[i]+Poly_C[3*i+0]*T(et)+Poly_C[3*i+1]*pow(T(et),2)+Poly_C[3*i+2]*pow(T(et),3);
				if(i<3){
					camera_R[i]=T(cam_temp);
				}
				else{
					camera_T[i-3]=T(cam_temp);
				}
			}

			T p[3];
			//ceres::AngleAxisRotatePoint(camera, point, p);
			T point_[3];
			point_[0]=point[0];point_[1]=point[1];point_[2]=point[2];
			My_AngleAxisRotatePoint(camera_R, point_, p);


			// camera[3,4,5] are the translation.
			T pc[3];
			T pc0[3];
			pc0[0]=camera_T[0];pc0[1]=camera_T[1];pc0[2]=camera_T[2];

			My_AngleAxisRotatePoint(camera_R, pc0, pc);
			p[0] -= pc[0]; p[1] -= pc[1]; p[2] -= pc[2];

			T xp =  -p[0] / p[2];
			T yp =  -p[1] / p[2];

			// Apply second and fourth order radial distortion.
			//const T& l1 = 0;//camera[7];
			//const T& l2 = 0;//camera[8];
			//T r2 = xp*xp + yp*yp;
			T distortion = T(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

			// Compute final projected point position.
			//T focal = T(1.0);
			T predicted_x = focal * distortion * xp;
			T predicted_y = focal * distortion * yp;

			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double* EO_0, double* Poly_CR, double focal) {
			return (new ceres::AutoDiffCostFunction<Mycost, 2, 9, 3>(
				new Mycost(observed_x, observed_y, et, EO_0, Poly_CR, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double* Poly_CR;
	double* EO_0;
	double focal;
};*/

//光束法交会
struct Projcost{
	Projcost(double observed_x, double observed_y, double et, double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), focal(focal){}

	template <typename T>
	bool operator()(const T* const point,
		const T* const Poly_EO,
		T* residuals) const {

			T camera_R[3],camera_T[3];
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_C[4*i+0]+Poly_C[4*i+1]*T(et)+Poly_C[4*i+2]*pow(T(et),2)+Poly_C[4*i+3]*pow(T(et),3);
				T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				if(i<3){
					camera_R[i]=T(cam_temp);
				}
				else{
					camera_T[i-3]=T(cam_temp);
				}
			}

			T p[3];
			//ceres::AngleAxisRotatePoint(camera, point, p);
			T point_[3];
			point_[0]=point[0];point_[1]=point[1];point_[2]=point[2];
			My_AngleAxisRotatePoint(camera_R, point_, p);


			// camera[3,4,5] are the translation.
			T pc[3];
			T pc0[3];
			pc0[0]=camera_T[0];pc0[1]=camera_T[1];pc0[2]=camera_T[2];

			My_AngleAxisRotatePoint(camera_R, pc0, pc);
			p[0] -= pc[0]; p[1] -= pc[1]; p[2] -= pc[2];

			T xp =  -p[0] / p[2];
			T yp =  -p[1] / p[2];

			// Apply second and fourth order radial distortion.
			//const T& l1 = 0;//camera[7];
			//const T& l2 = 0;//camera[8];
			//T r2 = xp*xp + yp*yp;
			T distortion = T(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

			// Compute final projected point position.
			//T focal = T(1.0);
			T predicted_x = focal * distortion * xp;
			T predicted_y = focal * distortion * yp;

			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double focal) {
			return (new ceres::AutoDiffCostFunction<Projcost, 2, 3, 24>(
				new Projcost(observed_x, observed_y, et, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double focal;
};

//轨道参数观测方程
struct Orbitalcost{
	Orbitalcost(double et, double* obs_EO)
		: et(et), obs_EO(obs_EO){}

	template <typename T>
	bool operator()(const T* const Poly_EO,
		T* residuals) const {

			for(int i=0;i<6;i++){
				if(i<3){
					//residuals[i] = T(171.4286)*((Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3))
					//	-(Poly_C[4*i+0]+Poly_C[4*i+1]*T(et)+Poly_C[4*i+2]*pow(T(et),2)+Poly_C[4*i+3]*pow(T(et),3)));
					residuals[i] = T(171.4286*171.4286)*((Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3))
						-(obs_EO[i+3]));
				}
				else{
					//residuals[i] = T(0.006)*((Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3))
					//	-(Poly_C[4*i+0]+Poly_C[4*i+1]*T(et)+Poly_C[4*i+2]*pow(T(et),2)+Poly_C[4*i+3]*pow(T(et),3)));
					residuals[i] = T(0.006)*((Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3))
						-(obs_EO[i-3]));
				}
			}
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double et, double* obs_EO) {
			return (new ceres::AutoDiffCostFunction<Orbitalcost, 6, 24>(
				new Orbitalcost(et, obs_EO)));
	}

	double et;
	double* obs_EO;
};


//前方交会
struct ForInsec_cost{
	ForInsec_cost(double observed_x, double observed_y, double et, double* Poly_C,double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), Poly_C(Poly_C), focal(focal){}

	template <typename T>
	bool operator()(const T* const point, 
		T* residuals) const {

			T camera_R[3],camera_T[3];
			for(int i=0;i<6;i++){
				T cam_temp=Poly_C[4*i+0]+Poly_C[4*i+1]*T(et)+Poly_C[4*i+2]*pow(T(et),2)+Poly_C[4*i+3]*pow(T(et),3);
				if(i<3){
					camera_R[i]=T(cam_temp);
				}
				else{
					camera_T[i-3]=T(cam_temp);
				}
			}

			T p[3];
			//ceres::AngleAxisRotatePoint(camera, point, p);
			T point_[3];
			point_[0]=point[0];point_[1]=point[1];point_[2]=point[2];
			My_AngleAxisRotatePoint(camera_R, point_, p);


			// camera[3,4,5] are the translation.
			T pc[3];
			T pc0[3];
			pc0[0]=camera_T[0];pc0[1]=camera_T[1];pc0[2]=camera_T[2];

			My_AngleAxisRotatePoint(camera_R, pc0, pc);
			p[0] -= pc[0]; p[1] -= pc[1]; p[2] -= pc[2];

			T xp =  -p[0] / p[2];
			T yp =  -p[1] / p[2];

			// Apply second and fourth order radial distortion.
			//const T& l1 = 0;//camera[7];
			//const T& l2 = 0;//camera[8];
			//T r2 = xp*xp + yp*yp;
			T distortion = T(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

			// Compute final projected point position.
			//T focal = T(1.0);
			T predicted_x = focal * distortion * xp;
			T predicted_y = focal * distortion * yp;

			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double* Poly_C, double focal) {
			return (new ceres::AutoDiffCostFunction<ForInsec_cost, 2, 3>(
				new ForInsec_cost(observed_x, observed_y, et, Poly_C, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double* Poly_C;
	double focal;
};

//计算残差
void Compu_ProjRES(double observed_x, double observed_y, double et, double* Poly_C,double focal,double* point,double* resdual){
	double camera_R[3],camera_T[3];
	for(int i=0;i<6;i++){
		double cam_temp=Poly_C[4*i+0]+Poly_C[4*i+1]*(et)+Poly_C[4*i+2]*pow((et),2)+Poly_C[4*i+3]*pow((et),3);
		if(i<3){
			camera_R[i]=double(cam_temp);
		}
		else{
			camera_T[i-3]=double(cam_temp);
		}
	}

	double p[3];
	//ceres::AngleAxisRotatePoint(camera, point, p);
	double point_[3];
	point_[0]=point[0];point_[1]=point[1];point_[2]=point[2];
	My_AngleAxisRotatePoint(camera_R, point_, p);

	// camera[3,4,5] are the translation.
	double pc[3];
	double pc0[3];
	pc0[0]=camera_T[0];pc0[1]=camera_T[1];pc0[2]=camera_T[2];

	My_AngleAxisRotatePoint(camera_R, pc0, pc);
	p[0] -= pc[0]; p[1] -= pc[1]; p[2] -= pc[2];

	double xp =  -p[0] / p[2];
	double yp =  -p[1] / p[2];

	// Apply second and fourth order radial distortion.
	//const T& l1 = 0;//camera[7];
	//const T& l2 = 0;//camera[8];
	//T r2 = xp*xp + yp*yp;
	double distortion = double(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

	// Compute final projected point position.
	//T focal = T(1.0);
	double predicted_x = focal * distortion * xp;
	double predicted_y = focal * distortion * yp;

	// The error is the difference between the predicted and observed position.
	double residuals[2];
	residuals[0] = predicted_x - double(observed_x);
	residuals[1] = predicted_y - double(observed_y);
	*resdual = sqrt(residuals[0]*residuals[0]+residuals[1]*residuals[1]);
}

/*
struct EOcost {
	EOcost(double* Poly_C,double* EO_0,double et)
		: Poly_C(Poly_C), EO_0(EO_0), et(et) {}

	template <typename T>
	bool operator()(const T* const Poly_C_,
		T* residuals) const {
			for(int i=0;i<6;i++){
				residuals[i]= EO_0[i]+Poly_C[3*i+0]*T(et)+Poly_C[3*i+1]*pow(T(et),2)+Poly_C[3*i+2]*pow(T(et),3)-(EO_0[i]+Poly_C_[3*i+0]*T(et)+Poly_C_[3*i+1]*pow(T(et),2)+Poly_C_[3*i+2]*pow(T(et),3));
			}
			return true;
	}


	static ceres::CostFunction* Create(double* Poly_C, double* EO_0, double et) {
		return (new ceres::AutoDiffCostFunction<EOcost, 6, 18>(
			new EOcost(Poly_C, EO_0, et)));
	}

	double* Poly_C;
	double *EO_0;
	double et;
};

struct EOcost1 {
	EOcost1(double* Poly_C)
		: Poly_C(Poly_C) {}

	template <typename T>
	bool operator()(const T* const Poly_C_,
		T* residuals) const {
			for(int i=0;i<18;i++){
				residuals[i]= Poly_C[i]-Poly_C_[i];
			}
			return true;
	}


	static ceres::CostFunction* Create(double* Poly_C) {
		return (new ceres::AutoDiffCostFunction<EOcost1, 18, 18>(
			new EOcost1(Poly_C)));
	}

	double* Poly_C;
};*/


int BAtest1(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* FItxt,char* GTtxt,char* EOretxt) {
	printf("Begin BAtest!\n");
	// The variable to solve for with its initial value. It will be
	// mutated in place by the solver.
	//double x = 0.5;
	//const double initial_x = x;

	//数据准备
	std::vector<double> match_point;
	std::vector<double> GT_initial;
	FILE *fp=fopen(observetxt,"r");
	double temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf\n",&temp1,&temp2,&temp3,&temp4,&temp5,&temp6,&temp7,&temp8);
		match_point.push_back(temp1);
		match_point.push_back(temp2);
		match_point.push_back(temp3);
		match_point.push_back(temp4);
		match_point.push_back(temp5);
		GT_initial.push_back(temp6);
		GT_initial.push_back(temp7);
		GT_initial.push_back(temp8);
	}
	fclose(fp);

	//int match_num=100;
	//double * match_point = new double[match_num*5];

	int image_num=2;
	int CCnum[2];
	CCnum[0]=80;CCnum[1]=160;
	//外方位元素观测
	double LR[2];

	char* EOtxt = new char[80];
	double ** poly_EO = new double *[image_num];
	double ** poly_CE = new double *[image_num];
	double ** poly_CR = new double *[image_num];
	double ** poly_CT = new double *[image_num];
	double ** obs_EO = new double *[image_num];
	double ** obs_et = new double *[image_num];
	char* xulie_ID;
	double *et0 = new double[2];
	for(int i=0;i<image_num;i++){
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else{
			xulie_ID=xulie_ID2;
		}

		//读取拟合结果
		sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID, "\\polyCC.txt" );
		FILE* fp_eo=fopen(EOtxt,"r");
		poly_EO[i] = new double[24];
		poly_CE[i] = new double[24];
		poly_CR[i] = new double[12];
		poly_CT[i] = new double[12];
		double temp,temp1;
		int jj;
		for(int j=0;j<6;j++){
			if(j<3){
				jj=j+3;
			}
			else{
				jj=j-3;
			}
			fscanf(fp_eo,"%lf",&temp);
			et0[i]=temp;
			//fscanf(fp_eo," %lf",&temp);
			//EO_0[i][jj]=temp;
			for(int k=0;k<4;k++){
				fscanf(fp_eo," %lf",&temp);
				poly_EO[i][jj*4+k] = temp;
				poly_CE[i][jj*4+k] = temp;
				if(jj<3){
					poly_CR[i][jj*4+k] = temp;
				}
				else{
					poly_CT[i][(jj-3)*4+k] = temp;
				}
			}
			fscanf(fp_eo,"\n");
		}
		fclose(fp_eo);

		//读取观测EO
		sprintf( EOtxt, "%s%s%s%s%s%s", EOfile, "\\", xulie_ID, "\\", xulie_ID, "_RED5_0.txt");
		fp_eo=fopen(EOtxt,"r");
		obs_EO[i] = new double[CCnum[i]*6];
		obs_et[i] = new double[CCnum[i]];
		fscanf(fp_eo,"%lf %lf\n",&temp,&temp1);
		LR[i]=temp1;
		for(int j=0;j<CCnum[i];j++){
			fscanf(fp_eo,"%lf",&temp);
			obs_et[i][j]=temp;
			for(int k=0;k<6;k++){
				fscanf(fp_eo," %lf",&temp);
				obs_EO[i][j*6+k] = temp;
			}
			fscanf(fp_eo,"\n");
		}
		fclose(fp_eo);
	}

	int ground_num = match_point.size()/5/2;
	double ** ground_point = new double *[ground_num];
	for(int i=0;i<ground_num;i++){
		ground_point[i] = new double[3];
		for(int j=0;j<3;j++){
			ground_point[i][j]=(GT_initial[i*2*3+j]+GT_initial[i*2*3+j+3])/2;
		}
		//memset(ground_point[i],0.0,3*sizeof(double));
	}

	for(int i=0;i<ground_num;i++){
		ground_point[i] = new double[3];
		for(int j=0;j<3;j++){
			ground_point[i][j]=GT_initial[i*2*3+j+3];
		}
		//memset(ground_point[i],0.0,3*sizeof(double));
	}


	double focal=-11995.48;

	//前方交会////////////////////////////////////////////////////////////
   Problem problem;
	for (int i=0; i<match_point.size()/5; i++) {
		ceres::CostFunction* cost_function =
			ForInsec_cost::Create(
			match_point[5*i+1],
			match_point[5*i+2],
			match_point[5*i+3]-et0[int(match_point[5*i+4])],
			poly_CE[int(match_point[5*i+4])],
			focal);
		problem.AddResidualBlock(cost_function,
			new ceres::CauchyLoss(0.5),
			//NULL,
			ground_point[int(match_point[5*i])]);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations=50;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << "\n";

	//计算残差
	double *resdual = new double[ground_num];
	for(int ii=0;ii<ground_num;ii++){
		double res1,res2;
		res1=res2=0;
		int i=2*ii;
		Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&res1);
		i=2*ii+1;
		Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&res2);
		resdual[ii]=sqrt(res1*res1+res2*res2);
	}

	FILE *fp_fi=fopen(FItxt,"w");
	fprintf(fp_fi,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		fprintf(fp_fi,"%lf %lf %lf %lf\n",ground_point[i][0],ground_point[i][1],ground_point[i][2],resdual[i]);
	}
	fclose(fp_fi);

	for(int i=0;i<ground_num;i++){
		//ground_point[i] = new double[3];
		for(int j=0;j<3;j++){
			//ground_point[i][j]=GT_initial[i*2*3+j+3];
		}
		//memset(ground_point[i],0.0,3*sizeof(double));
	}

	//光束法平差////////////////////////////////////////////////////////////
	Problem problem1;
	//共线条件方程
	for (int i=0; i<match_point.size()/5; i++) {
		ceres::CostFunction* cost_function =
			Projcost::Create(
			match_point[5*i+1],
			match_point[5*i+2],
			match_point[5*i+3]-et0[int(match_point[5*i+4])],
			focal);
		problem1.AddResidualBlock(cost_function,
			new ceres::CauchyLoss(1.0),
			//NULL,
			ground_point[int(match_point[5*i])],
			poly_EO[int(match_point[5*i+4])]);
	}

	//轨道参数方程
	for(int ii=0;ii<2;ii++){
		for(int i=0;i<CCnum[ii];i++){
			double * eo_temp=new double[6];
			for(int j=0;j<6;j++){
				eo_temp[j]=obs_EO[ii][i*6+j];
			}
			ceres::CostFunction* cost_function =
				Orbitalcost::Create(
				obs_et[ii][i]-et0[ii],
				eo_temp);
			problem1.AddResidualBlock(cost_function,
				new ceres::CauchyLoss(1.0),
				//NULL,
				poly_EO[ii]);
		}
	}

	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_SCHUR;
	options1.minimizer_progress_to_stdout = true;
	options1.max_num_iterations=50;
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	std::cout << summary1.FullReport() << "\n";/**/
	/*
	//二次前方交会
	Problem problem2;
	for (int i=0; i<match_point.size()/5; i++) {
		ceres::CostFunction* cost_function =
			ForInsec_cost::Create(
			match_point[5*i+1],
			match_point[5*i+2],
			match_point[5*i+3]-et0[int(match_point[5*i+4])],
			poly_EO[int(match_point[5*i+4])],
			focal);
		problem2.AddResidualBlock(cost_function,
			new ceres::CauchyLoss(0.5),
			//NULL,
			ground_point[int(match_point[5*i])]);
	}

	ceres::Solver::Options options2;
	options2.linear_solver_type = ceres::DENSE_SCHUR;
	options2.minimizer_progress_to_stdout = true;
	options2.max_num_iterations=50;
	ceres::Solver::Summary summary2;
	ceres::Solve(options2, &problem2, &summary2);

	std::cout << summary2.FullReport() << "\n";*/
	
	for(int ii=0;ii<ground_num;ii++){
		double res1,res2;
		res1=res2=0;
		int i=2*ii;
		Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&res1);
		i=2*ii+1;
		Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&res2);
		resdual[ii]=sqrt(res1*res1+res2*res2);
	}

	FILE *fp_gt=fopen(GTtxt,"w");
	fprintf(fp_gt,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		fprintf(fp_gt,"%lf %lf %lf %lf\n",ground_point[i][0],ground_point[i][1],ground_point[i][2],resdual[i]);
	}
	fclose(fp_gt);

	FILE *fp_eo=fopen(EOretxt,"w");
	fprintf(fp_eo,"%d\n",image_num);
	for(int i=0;i<image_num;i++){
		for(int j=0;j<6;j++){
			if(j==5){
				fprintf(fp_eo,"%.12lf %.12lf %.12lf %.12lf %.12lf\n\n",et0[i],poly_EO[i][j*4+0],poly_EO[i][j*4+1],poly_EO[i][j*4+2],poly_EO[i][j*4+3]);
			}
			else{
				fprintf(fp_eo,"%.12lf %.12lf %.12lf %.12lf %.12lf\n",et0[i],poly_EO[i][j*4+0],poly_EO[i][j*4+1],poly_EO[i][j*4+2],poly_EO[i][j*4+3]);
			}
		}
	}
	fclose(fp_eo);
	return 0;
}
