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

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
	template <typename T> bool operator()(const T* const x, T* residual) const {
		residual[0] = 10.0 - x[0];
		return true;
	}
};

struct cost_function_define
{
	cost_function_define(Point3d p1,Point3d p2):_p1(p1),_p2(p2){}
	template<typename T>
	bool operator()(const T* const cere_r,const T* const cere_t,T* residual)const
	{
		T p_1[3];
		T p_2[3];
		p_1[0]=T(_p1.x);
		p_1[1]=T(_p1.y);
		p_1[2]=T(_p1.z);
		ceres::AngleAxisRotatePoint(cere_r,p_1,p_2);
		p_2[0]=p_2[0]+cere_t[0];
		p_2[1]=p_2[1]+cere_t[1];
		p_2[2]=p_2[2]+cere_t[2];
		const T x=p_2[0]/p_2[2];
		const T y=p_2[1]/p_2[2];
		const T u=x*520.9+325.1;
		const T v=y*521.0+249.7;
		T p_3[3];
		p_3[0]=T(_p2.x);
		p_3[1]=T(_p2.y);
		p_3[2]=T(_p2.z);
		const T x1=p_3[0]/p_3[2];
		const T y1=p_3[1]/p_3[2];
		const T u1=x1*520.9+325.1;
		const T v1=y1*521.0+249.7;
		residual[0]=u-u1;
		residual[1]=v-v1;
		return true;
	}
	Point3d _p1,_p2;
};


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

//输入：内定向以后的像面坐标(/f进行归一化)以及星历时间
//camera：phi w k Xs Ys Zs的拟合参数（每个4个，共4*6=24个）；Point：3*1（地面点）
struct SnavelyReprojectionError {
	SnavelyReprojectionError(double observed_x, double observed_y, double et, double* EO_0, double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), EO_0(EO_0),focal(focal) {}

	template <typename T>
	bool operator()(const T* const Poly_C,
		const T* const point,
		T* residuals) const {

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
		const double observed_y, const double et, double* EO_0, double focal) {
			return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 18, 3>(
				new SnavelyReprojectionError(observed_x, observed_y, et, EO_0, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double* EO_0;
	double focal;
};


//前方交会
struct SnavelyReprojectionError1 {
	SnavelyReprojectionError1(double observed_x, double observed_y, double et, double* EO_0, double* Poly_C,double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), EO_0(EO_0), Poly_C(Poly_C), focal(focal){}

	template <typename T>
	bool operator()(const T* const point,
		T* residuals) const {

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
		const double observed_y, const double et, double* EO_0, double* Poly_C, double focal) {
			return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError1, 2, 3>(
				new SnavelyReprojectionError1(observed_x, observed_y, et, EO_0, Poly_C, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double* Poly_C;
	double* EO_0;
	double focal;
};

//后方交会
struct SnavelyReprojectionError2 {
	SnavelyReprojectionError2(double observed_x, double observed_y, double et, double* EO_0, double* point, double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), EO_0(EO_0), point(point), focal(focal) {}

	template <typename T>
	bool operator()(const T* const Poly_C,
		T* residuals) const {

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
			point_[0]=T(point[0]);point_[1]=T(point[1]);point_[2]=T(point[2]);
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
		const double observed_y, const double et, double* EO_0, double* point, double focal) {
			return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError2, 2, 18>(
				new SnavelyReprojectionError2(observed_x, observed_y, et, EO_0, point, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double* point;
	double* EO_0;
	double  focal;
};

struct EOcost0 {
	EOcost0(double* Poly_C,double* EO_0,double et)
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
		return (new ceres::AutoDiffCostFunction<EOcost0, 6, 18>(
			new EOcost0(Poly_C, EO_0, et)));
	}

	double* Poly_C;
	double *EO_0;
	double et;
};



int BAtest(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* GTtxt,char* EOretxt) {

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
	char* EOtxt = new char[80];
	double ** poly_EO = new double *[image_num];
	double ** poly_C = new double *[image_num];
	double ** EO_0 = new double *[image_num];
	char* xulie_ID;
	double *et0 = new double[2];
	for(int i=0;i<image_num;i++){
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else{
			xulie_ID=xulie_ID2;
		}
		sprintf( EOtxt, "%s%s%s%s", EOfile, "\\", xulie_ID, "\\polyCC.txt" );
		FILE* fp_eo=fopen(EOtxt,"r");
		poly_EO[i] = new double[18];
		poly_C[i] = new double[18];
		EO_0[i] = new double[6];
		double temp;
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
			fscanf(fp_eo," %lf",&temp);
			EO_0[i][jj]=temp;
			for(int k=0;k<3;k++){
				fscanf(fp_eo," %lf",&temp);
				poly_EO[i][jj*3+k] = temp;
				poly_C[i][jj*3+k] = temp;
			}
			fscanf(fp_eo,"\n");
		}
		//memset(poly_EO[i],0,24*sizeof(double));
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
			ground_point[i][j]=(GT_initial[i*2*3+j]+GT_initial[i*2*3+j+3])/2;
		}
		//memset(ground_point[i],0.0,3*sizeof(double));
	}

	double focal=-11995.48;

	//前方交会////////////////////////////////////////////////////////////
    Problem problem1;
	for (int i=0; i<match_point.size()/5; i++) {
		ceres::CostFunction* cost_function =
			SnavelyReprojectionError1::Create(
			match_point[5*i+1],
			match_point[5*i+2],
			match_point[5*i+3]-et0[int(match_point[5*i+4])],
			EO_0[int(match_point[5*i+4])],
			poly_EO[int(match_point[5*i+4])],
		    focal);
		problem1.AddResidualBlock(cost_function,
			NULL,
			ground_point[int(match_point[5*i])]);
	}

	ceres::Solver::Options options1;
	options1.linear_solver_type = ceres::DENSE_SCHUR;
	options1.minimizer_progress_to_stdout = true;
	options1.max_num_iterations=50;
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	std::cout << summary1.FullReport() << "\n";/**/

	
	//后方交会////////////////////////////////////////////////////////////
	Problem problem2;
	for (int i=0; i<match_point.size()/5; i+=1) {
		ceres::CostFunction* cost_function =
			SnavelyReprojectionError2::Create(
			match_point[5*i+1],
			match_point[5*i+2],
			match_point[5*i+3]-et0[int(match_point[5*i+4])],
			EO_0[int(match_point[5*i+4])],
			ground_point[int(match_point[5*i])]);
		problem2.AddResidualBlock(cost_function,
			NULL,
			poly_EO[int(match_point[5*i+4])]);
	}

	ceres::Solver::Options options2;
	options2.linear_solver_type = ceres::DENSE_SCHUR;
	options2.minimizer_progress_to_stdout = true;
	options2.max_num_iterations=20;
	ceres::Solver::Summary summary2;
	ceres::Solve(options2, &problem2, &summary2);

	std::cout << summary2.FullReport() << "\n";
	/**/
	
	
	Problem problem;
	//光束法平差//////////////////////////////////////////////////
	for (int i=0; i<match_point.size()/5; i++) {
		ceres::CostFunction* cost_function =
			SnavelyReprojectionError::Create(
			match_point[5*i+1],
			match_point[5*i+2],
			match_point[5*i+3]-et0[int(match_point[5*i+4])],
			EO_0[int(match_point[5*i+4])],
			focal);
		problem.AddResidualBlock(cost_function,
			NULL,
			poly_EO[int(match_point[5*i+4])],
			ground_point[int(match_point[5*i])]);

		ceres::CostFunction* cost_function1 =
			EOcost0::Create(poly_C[int(match_point[5*i+4])],
			EO_0[int(match_point[5*i+4])],
			match_point[5*i+3]-et0[int(match_point[5*i+4])]);
		problem1.AddResidualBlock(cost_function1,
			NULL,
			poly_EO[int(match_point[5*i+4])]);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations=3;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << "\n";
	

	FILE *fp_gt=fopen(GTtxt,"w");
	fprintf(fp_gt,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		fprintf(fp_gt,"%lf %lf %lf\n",ground_point[i][0],ground_point[i][1],ground_point[i][2]);
	}
	fclose(fp_gt);

	FILE *fp_eo=fopen(EOretxt,"w");
	fprintf(fp_eo,"%d\n",image_num);
	for(int i=0;i<image_num;i++){
		for(int j=0;j<6;j++){
			if(j==5){
				fprintf(fp_eo,"%.12lf %.12lf %.12lf %.12lf %.12lf\n\n",et0[i],EO_0[i][j],poly_EO[i][j*3+0],poly_EO[i][j*3+1],poly_EO[i][j*3+2]);
			}
			else{
				fprintf(fp_eo,"%.12lf %.12lf %.12lf %.12lf %.12lf\n",et0[i],EO_0[i][j],poly_EO[i][j*3+0],poly_EO[i][j*3+1],poly_EO[i][j*3+2]);
			}
		}
	}
	fclose(fp_eo);
	return 0;
}
