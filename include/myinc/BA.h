#ifndef _BA_H_
#define _BA_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "Observation.h"
#include "EO.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#define init_poly_num 24
#define poly_num 318
#define rate_BA 1
template <typename T>
//phi，w，k旋转
void My_AngleAxisRotatePoint(T camera[3],T point[3],T p[3]);

// rec:XYZ; re:长轴 f:扁率 lon,lat:弧度单位制
template <typename T>
void rec2geo(T rec[3],T re,T f,T* lon,T* lat,T* H){
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

template <typename T>
void Get_polyEO_T_IO(T et, T Poly_C[24], T EO[6]);

template <typename T>
void Get_polyEO_T0(T et, T Poly_C[poly_num], T EO[6]);

template <typename T>
void Get_polyEO_T(T et, T Poly_C[poly_num], T EO[6]);

template <typename T>
void Get_polyEO_T1(T et, T Poly_C[poly_num], T EO[6]);

template <typename T>
void Get_IOJitter_T(T et, T Poly_C[poly_num], T IOJitter[2]);

struct Projcost0{
	Projcost0(double observed_x, double observed_y, double et, double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), focal(focal){}

	template <typename T>
	bool operator()(const T* const point,
		const T* const Poly_EO,
		T* residuals) const {
			//int poly_num = 42;

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[24];
			for(int i=0;i<24;i++){
				Poly_temp[i]=Poly_EO[i];
			}
			Get_polyEO_T0(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				T cam_temp=T(EO_temp[i]);
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
			return (new ceres::AutoDiffCostFunction<Projcost0, 2, 3, 24>(
				new Projcost0(observed_x, observed_y, et, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double focal;
};

struct Projcost1{
	Projcost1(double observed_x, double observed_y, double et, double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), focal(focal){}

	template <typename T>
	bool operator()(const T* const point,
		const T* const Poly_EO,
		const T* const IO,
		T* residuals) const {
			//int poly_num = 42;

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[24];
			for(int i=0;i<24;i++){
				Poly_temp[i]=Poly_EO[i];
			}
			Get_polyEO_T0(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				T cam_temp=T(EO_temp[i]);
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
			T predicted_x = focal * distortion * (xp-IO[0]*sin(IO[1]*et+IO[2]));
			T predicted_y = focal * distortion * (yp-IO[3]*sin(IO[4]*et+IO[5]));

			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double focal) {
			return (new ceres::AutoDiffCostFunction<Projcost1, 2, 3, 24, 6>(
				new Projcost1(observed_x, observed_y, et, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double focal;
};

struct Projcost{
	Projcost(double observed_x, double observed_y, double et, double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), focal(focal){}

	template <typename T>
	bool operator()(const T* const point,
		const T* const Poly_EO,
		T* residuals) const {
			//int poly_num = 42;

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[poly_num];
			for(int i=0;i<poly_num;i++){
				Poly_temp[i]=Poly_EO[i];
			}
			Get_polyEO_T(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				T cam_temp=T(EO_temp[i]);
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
			residuals[1] = T(rate_BA)*(predicted_y - T(observed_y));
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double focal) {
			return (new ceres::AutoDiffCostFunction<Projcost, 2, 3, poly_num>(
				new Projcost(observed_x, observed_y, et, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double focal;
};

struct IO_Projcost{
	IO_Projcost(double observed_x, double observed_y, double et, double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), focal(focal){}

	template <typename T>
	bool operator()(const T* const point,
		const T* const Poly_EO,
		const T* const Poly_IO,
		T* residuals) const {

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[24];
			for(int i=0;i<24;i++){
				Poly_temp[i]=Poly_EO[i];
			}
			Get_polyEO_T_IO(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				T cam_temp=T(EO_temp[i]);
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
			T IOJitter_temp[2];
			T IOPoly_temp[poly_num];
			for(int i=0;i<poly_num;i++){
				IOPoly_temp[i]=T(Poly_IO[i]);
			}
			Get_IOJitter_T(T(et),IOPoly_temp,IOJitter_temp);
			T predicted_x = focal * distortion * (xp+IOJitter_temp[0]);
			T predicted_y = focal * distortion * (yp+IOJitter_temp[1]);

			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);
			//residuals[2] = T(0.0000000001)*IOJitter_temp[0];
			//residuals[3] = T(0.0000000001)*IOJitter_temp[1];
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double focal) {
			return (new ceres::AutoDiffCostFunction<IO_Projcost, 2, 3, 24, poly_num>(
				new IO_Projcost(observed_x, observed_y, et, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double focal;
};

//这个有问题
struct J_Projcost{
	J_Projcost(double observed_x, double observed_y, double et, double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), focal(focal){}

	template <typename T>
	bool operator()(const T* const point,
		const T* const Poly_EO,
		const T* const Jitter_EO,
		T* residuals) const {
			//int poly_num = 42;

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[poly_num+24];
			int pn=poly_num+24;
			for(int i=0;i<6;i++){
				for(int j=0;j<pn/6;j++){
					if(j<4){
						Poly_temp[i*pn/6+j]=Poly_EO[i*4+j];
					}
					else{
						Poly_temp[i*pn/6+j]=Jitter_EO[i*(pn/6-4)+j-4];
					}
				}
			}

			Get_polyEO_T(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				T cam_temp=T(EO_temp[i]);
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
			return (new ceres::AutoDiffCostFunction<J_Projcost, 2, 3, 24, poly_num>(
				new J_Projcost(observed_x, observed_y, et, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double focal;
};

struct Orbitalcost0{
	Orbitalcost0(double et, double* obs_EO)
		: et(et), obs_EO(obs_EO){}

	template <typename T>
	bool operator()(const T* const Poly_EO,
		T* residuals) const {

			T EO_temp[6];
			T Poly_temp[24];
			for(int i=0;i<24;i++){
				Poly_temp[i]=Poly_EO[i];
			}
			Get_polyEO_T0(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				if(i<3){
					residuals[i] = T(20*1)*(EO_temp[i]
					-(obs_EO[i+3]));
				}
				else{
					residuals[i] = T(0.006*1)*(EO_temp[i]
					-(obs_EO[i-3]));
				}
			}

			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double et, double* obs_EO) {
		return (new ceres::AutoDiffCostFunction<Orbitalcost0, 6, 24>(
			new Orbitalcost0(et, obs_EO)));
	}

	double et;
	double* obs_EO;
};

struct Orbitalcost{
	Orbitalcost(double et, double* obs_EO)
		: et(et), obs_EO(obs_EO){}

	template <typename T>
	bool operator()(const T* const Poly_EO,
		T* residuals) const {

			T EO_temp[6];
			T Poly_temp[poly_num];
			for(int i=0;i<poly_num;i++){
				Poly_temp[i]=Poly_EO[i];
			}
			Get_polyEO_T0(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				if(i<3){
					residuals[i] = T(20*1)*(EO_temp[i]
					-(obs_EO[i+3]));
				}
				else{
					residuals[i] = T(0.006*1)*(EO_temp[i]
					-(obs_EO[i-3]));
				}
			}

			//Jitter项
			Get_polyEO_T1(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				if(i<2){
					residuals[i+6] = T(20*0.0001)*(EO_temp[i]);
				}
				else{
					residuals[i+6] = T(0.006*10000)*(EO_temp[i]);
				}
			}
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double et, double* obs_EO) {
		return (new ceres::AutoDiffCostFunction<Orbitalcost, 12, poly_num>(
			new Orbitalcost(et, obs_EO)));
	}

	double et;
	double* obs_EO;
};

//这个有问题
struct J_Orbitalcost{
	J_Orbitalcost(double et, double* obs_EO)
		: et(et), obs_EO(obs_EO){}

	template <typename T>
	bool operator()(const T* const Poly_EO,
		const T* const Jitter_EO,
		T* residuals) const {

			T EO_temp[6];
			T Poly_temp[poly_num+24];
			int pn=poly_num+24;
			for(int i=0;i<6;i++){
				for(int j=0;j<pn/6;j++){
					if(j<4){
						Poly_temp[i*pn/6+j]=Poly_EO[i*4+j];
					}
					else{
						Poly_temp[i*pn/6+j]=Jitter_EO[i*(pn/6-4)+j-4];
					}
				}
			}
			Get_polyEO_T(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				if(i<3){
					residuals[i] = T(20*1)*(EO_temp[i]
					-(obs_EO[i+3]));
				}
				else{
					residuals[i] = T(0.006*1)*(EO_temp[i]
					-(obs_EO[i-3]));
				}
			}

			//Jitter项
			Get_polyEO_T1(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				if(i<2){
					residuals[i+6] = T(0);//T(20*0.000001)*(EO_temp[i]);
				}
				else if(i<3){
					residuals[i+6] = T(20*1000)*(EO_temp[i]);
				}
				else{
					residuals[i+6] = T(0.006*100000)*(EO_temp[i]);
				}
			}
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double et, double* obs_EO) {
		return (new ceres::AutoDiffCostFunction<J_Orbitalcost, 12, 24, poly_num>(
			new J_Orbitalcost(et, obs_EO)));
	}

	double et;
	double* obs_EO;
};

//约束EO拟合式系数的情况
struct Orbitalcost1{
	Orbitalcost1(double* Poly_EO0)
		: Poly_EO0(Poly_EO0){}

	template <typename T>
	bool operator()(const T* const Poly_EO,
		T* residuals) const {

			for(int i=0;i<poly_num;i++){
				if(i<poly_num/2){
					residuals[i]=T(20*100000000000000)*T(Poly_EO0[i]-Poly_EO[i]);
				}
				else{
					residuals[i]=T(0.006*100000000000)*T(Poly_EO0[i]-Poly_EO[i]);
				}
			}
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(double* Poly_EO0) {
		return (new ceres::AutoDiffCostFunction<Orbitalcost1, poly_num, poly_num>(
			new Orbitalcost1(Poly_EO0)));
	}

	double* Poly_EO0;
};

struct Orbitalcost2{
	Orbitalcost2(double et, double* obs_EO)
		: et(et), obs_EO(obs_EO){}

	template <typename T>
	bool operator()(const T* const Poly_EO,
		T* residuals) const {

			T EO_temp[6];
			T Poly_temp[poly_num];
			for(int i=0;i<poly_num;i++){
				Poly_temp[i]=Poly_EO[i];
			}

			Get_polyEO_T0(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				if(i<3){
					residuals[i] = T(20*1)*(EO_temp[i]
					-(obs_EO[i+3]));
				}
				else{
					residuals[i] = T(0.006*1)*(EO_temp[i]
					-(obs_EO[i-3]));
				}
			}

			////Jitter项
			//Get_polyEO_T1(T(et),Poly_temp,EO_temp);
			//for(int i=0;i<6;i++){
			// if(i<2){
			// residuals[i+6] = T(20*0.001)*abs(EO_temp[i])*EO_temp[i];
			// }
			// else{
			// residuals[i+6] = T(0.006*100000)*abs(EO_temp[i])*EO_temp[i];
			// }
			//}

			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double et, double* obs_EO) {
		return (new ceres::AutoDiffCostFunction<Orbitalcost2, 6, poly_num>(
			new Orbitalcost2(et, obs_EO)));
	}

	double et;
	double* obs_EO;
};

struct PointControlError_H{
	PointControlError_H(double H0)
		: H0(H0){}

	template <typename T>
	bool operator()(const T* const point,
		T* residuals) const {
			//translate to local coordinate(NEH)
			T E,N,H;
			T rectan[3]={point[0],point[1],point[2]};
			rec2geo(rectan,T(3396190),T(0),&E,&N,&H); 

			//control H
			T cov_p = T(0.001);
			residuals[0] = T(3)*cov_p*(H-T(H0));

			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(double H0) {
		return (new ceres::AutoDiffCostFunction<PointControlError_H, 1, 3>(
			new PointControlError_H(H0)));
	}

	double H0;
};

struct PointControlError{
	PointControlError(double* prev_XYZ)
		: prev_XYZ(prev_XYZ){}

	template <typename T>
	bool operator()(const T* const point,
		T* residuals) const {
			//translate to local coordinate(NEH)
			T E,N,H;
			T E0,N0,H0;
			T rectan[3]={(point[0]),(point[1]),(point[2])};
			rec2geo(rectan,T(3396190),T(0),&E,&N,&H); 

			// ConstSpiceDouble rectan0[3]={prev_XYZ[0],prev_XYZ[1],prev_XYZ[2]};
			// recgeo_c(rectan0,3396190,0,&E0,&N0,&H0); 
			T rectan0[3]={T(prev_XYZ[0]),T(prev_XYZ[1]),T(prev_XYZ[2])};
			rec2geo(rectan0,T(3396190),T(0),&E0,&N0,&H); 

			//control NEH
			T cov_p = T(0.0000001);
			//residuals[0] = cov_p*T(E-E0);
			//residuals[1] = cov_p*T(N-N0);
			residuals[0] = T(3.0)*cov_p*(H-T(H0));

			return true;
	}


	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(double* prev_XYZ) {
		return (new ceres::AutoDiffCostFunction<PointControlError, 1, 3>(
			new PointControlError(prev_XYZ)));
	}

	double* prev_XYZ;
};

struct Jittercost{
	Jittercost(double et, double* obs_EO)
		: et(et), obs_EO(obs_EO){}

	template <typename T>
	bool operator()(const T* const Poly_EO,
		T* residuals) const {

			T EO_temp[6];
			T Poly_temp[poly_num];
			for(int i=0;i<poly_num;i++){
				Poly_temp[i]=T(Poly_EO[i]);
			}
			Get_polyEO_T(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				if(i<3){
					residuals[i] = T(20*1)*(EO_temp[i]);
				}
				else{
					residuals[i] = T(0.006*1)*(EO_temp[i]);
				}
			}
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double et, double* obs_EO) {
		return (new ceres::AutoDiffCostFunction<Jittercost, 6, poly_num>(
			new Jittercost(et, obs_EO)));
	}

	double et;
	double* obs_EO;
};

struct IOcost{
	IOcost(double* IO_0)
		: IO_0(IO_0){}

	template <typename T>
	bool operator()(const T* const IO,
		T* residuals) const {

			for(int i=0;i<6;i++){
				if(i<3){
					residuals[i] = T(1*1)*(IO_0[i]-IO[i]);
				}
				else{
					residuals[i] = T(1*1)*(IO_0[i]-IO[i]);
				}
			}
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(double* IO_0) {
		return (new ceres::AutoDiffCostFunction<IOcost, 6, 6>(
			new IOcost(IO_0)));
	}

	double* IO_0;
};

struct ForInsec_cost{
	ForInsec_cost(double observed_x, double observed_y, double et, double* Poly_C,double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), Poly_C(Poly_C), focal(focal){}

	template <typename T>
	bool operator()(const T* const point, 
		T* residuals) const {
			//int poly_num = 42;

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[poly_num];
			for(int i=0;i<poly_num;i++){
				Poly_temp[i]=T(Poly_C[i]);
			}
			Get_polyEO_T(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				T cam_temp=T(EO_temp[i]);
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
			residuals[1] = T(rate_BA)*(predicted_y - T(observed_y));
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

struct ForInsec_cost_initial{
	ForInsec_cost_initial(double observed_x, double observed_y, double et, double* Poly_C,double focal)
		: observed_x(observed_x), observed_y(observed_y), et(et), Poly_C(Poly_C), focal(focal){}

	template <typename T>
	bool operator()(const T* const point, 
		T* residuals) const {
			//int poly_num = 42;

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[init_poly_num];
			for(int i=0;i<init_poly_num;i++){
				Poly_temp[i]=T(Poly_C[i]);
			}
			Get_polyEO_T_IO(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				T cam_temp=T(EO_temp[i]);
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
			residuals[1] = T(rate_BA)*(predicted_y - T(observed_y));
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double* Poly_C, double focal) {
			return (new ceres::AutoDiffCostFunction<ForInsec_cost_initial, 2, 3>(
				new ForInsec_cost_initial(observed_x, observed_y, et, Poly_C, focal)));
	}

	double observed_x;
	double observed_y;
	double et;
	double* Poly_C;
	double focal;
};

struct ForInsec_cost1{
	ForInsec_cost1(double observed_x, double observed_y, double et, double* Poly_C,double focal,double* IO)
		: observed_x(observed_x), observed_y(observed_y), et(et), Poly_C(Poly_C), focal(focal), IO(IO){}

	template <typename T>
	bool operator()(const T* const point, 
		T* residuals) const {
			//int poly_num = 42;

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[poly_num];
			for(int i=0;i<poly_num;i++){
				Poly_temp[i]=T(Poly_C[i]);
			}
			Get_polyEO_T(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				T cam_temp=T(EO_temp[i]);
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
			T predicted_x = focal * distortion * (xp-IO[0]*sin(IO[1]*et+IO[2]));
			T predicted_y = focal * distortion * (yp-IO[3]*sin(IO[4]*et+IO[5]));

			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double* Poly_C, double focal,double* IO) {
			return (new ceres::AutoDiffCostFunction<ForInsec_cost1, 2, 3>(
				new ForInsec_cost1(observed_x, observed_y, et, Poly_C, focal,IO)));
	}

	double observed_x;
	double observed_y;
	double et;
	double* Poly_C;
	double focal;
	double* IO;
};

struct IO_ForInseccost{
	IO_ForInseccost(double observed_x, double observed_y, double et, double focal,double* Poly_EO,double* Poly_IO)
		: observed_x(observed_x), observed_y(observed_y), et(et), focal(focal), Poly_EO(Poly_EO), Poly_IO(Poly_IO){}

	template <typename T>
	bool operator()(const T* const point,
		T* residuals) const {

			T camera_R[3],camera_T[3];
			T EO_temp[6];
			T Poly_temp[24];
			for(int i=0;i<24;i++){
				Poly_temp[i]=T(Poly_EO[i]);
			}
			Get_polyEO_T_IO(T(et),Poly_temp,EO_temp);
			for(int i=0;i<6;i++){
				//T cam_temp=Poly_EO[4*i+0]+Poly_EO[4*i+1]*T(et)+Poly_EO[4*i+2]*pow(T(et),2)+Poly_EO[4*i+3]*pow(T(et),3);
				T cam_temp=T(EO_temp[i]);
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
			T IOJitter_temp[2];
			T IOPoly_temp[poly_num];
			for(int i=0;i<poly_num;i++){
				IOPoly_temp[i]=T(Poly_IO[i]);
			}
			Get_IOJitter_T(T(et),IOPoly_temp,IOJitter_temp);
			T predicted_x = focal * distortion * (xp+IOJitter_temp[0]);
			T predicted_y = focal * distortion * (yp+IOJitter_temp[1]);

			// The error is the difference between the predicted and observed position.
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);
			return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double et, double focal,double* Poly_EO,double* Poly_IO) {
			return (new ceres::AutoDiffCostFunction<IO_ForInseccost, 2, 3>(
				new IO_ForInseccost(observed_x, observed_y, et, focal,Poly_EO,Poly_IO)));
	}

	double observed_x;
	double observed_y;
	double et;
	double focal;
	double* Poly_EO;
	double* Poly_IO;
};

template <typename T>
void Compu_ProjRES0(T observed_x, T observed_y, T et, T* Poly_C,T focal,T* point,T* resdual);
template <typename T>
void Compu_ProjRES(T observed_x, T observed_y, T et, T* Poly_C,T focal,T* point,T* resdual,int mark); 
template <typename T>
void Compu_ProjRES(T observed_x, T observed_y, T et, T* Poly_C, T* Poly_IO, T focal,T* point,T* resdual);

void Compu_ProjRES1(double observed_x, double observed_y, double et, double* Poly_C,double focal,double* point,double* resdual);
int BAtest1(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* FItxt,char* GTtxt,char* EOretxt);

int Block_adjustment(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* GTtxt,char* EOreFile);
int Jitter_adjustment(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* GTtxt,char* EOreFile);
int Forward_intersection(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* FItxt,char* GTtxt,char* EOretxt);
int Forward_intersection(char* observetxt,char* EOfile,double** poly_IO,char* xulie_ID1,char* xulie_ID2,char* FItxt);
int Forward_intersection_initial(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* FItxt);

int IO_adjustment(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* GTtxt,char* EOreFile);

int BA_main(const PipelineConfig& cfg);

//#include "../src/BA.cpp"
//#include "../src/BA_test.cpp"
#endif
