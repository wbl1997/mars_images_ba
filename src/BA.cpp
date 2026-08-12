#include "BA.h"
#include <algorithm>
#include <omp.h>

namespace {
const PipelineConfig::BaParams* g_ba_params = nullptr;

const PipelineConfig::BaParams& ba_cfg() {
	static const PipelineConfig::BaParams kDefault;
	return g_ba_params ? *g_ba_params : kDefault;
}

void ConfigureCeresThreads(ceres::Solver::Options& options) {
	int n = 1;
#ifdef _OPENMP
	n = std::max(1, omp_get_max_threads());
#endif
	options.num_threads = n;
	printf("[INFO][Ceres] num_threads=%d\n", n);
}

ceres::LossFunction* MakeBaCauchyLoss() {
	return new ceres::CauchyLoss(static_cast<double>(ba_cfg().cauchy_loss_ba));
}

ceres::LossFunction* MakeFiCauchyLoss() {
	return new ceres::CauchyLoss(static_cast<double>(ba_cfg().cauchy_loss_fi));
}
}  // namespace

template <typename T>
//phi，w，k旋转
void My_AngleAxisRotatePoint(T camera[3],T point[3],T p[3]){
	T phi=-camera[0];T w=camera[1];T k=camera[2];
	T R_inv[9];

	R_inv[0]=cos(phi)*cos(k)-sin(phi)*sin(w)*sin(k);
	R_inv[3]=-cos(phi)*sin(k)-sin(phi)*sin(w)*cos(k);
	//R_inv[3]=-R_inv[3];
	R_inv[6]=-sin(phi)*cos(w);

	R_inv[1]=cos(w)*sin(k);
	//R_inv[1]=-R_inv[1];
	R_inv[4]=cos(w)*cos(k);
	R_inv[7]=-sin(w);
	//R_inv[7]=-R_inv[7];

	R_inv[2]=sin(phi)*cos(k)+cos(phi)*sin(w)*sin(k);
	R_inv[5]=-sin(phi)*sin(k)+cos(phi)*sin(w)*cos(k);
	//R_inv[5]=-R_inv[5];
	R_inv[8]=cos(phi)*cos(w);

	for(int i=0;i<3;i++){
		p[i]=T(0);
		for(int j=0;j<3;j++){
			p[i] += R_inv[i*3+j]*point[j];
		}
	}
}

template <typename T>
void Get_polyEO_T_IO(T et, T Poly_C[24], T EO[6]){
	int n=24/6;
	for(int i=0;i<6;i++){
		T t=et;
		T t0=T(0.0);
		EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
	}
}

template <typename T>
void Get_polyEO_T0(T et, T Poly_C[poly_num], T EO[6]){
	int n=poly_num/6;
	for(int i=0;i<6;i++){
		T t=et;
		T t0=T(0.0);
		EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
	}
}

template <typename T>
void Get_polyEO_T(T et, T Poly_C[poly_num], T EO[6]){
	for(int i=0;i<6;i++){
		T t=et;
		T t0=T(0.0);
		if(poly_num/6==7){
			if(i<6){
				EO[i]=Poly_C[7*i+0]+Poly_C[7*i+1]*(t-t0)+Poly_C[7*i+2]*pow((t-t0),2)+Poly_C[7*i+3]*pow((t-t0),3)+Poly_C[7*i+4]*cos(Poly_C[7*i+5]*(t-t0)+Poly_C[7*i+6]); //余弦函数
			}
			else{
				EO[i]=Poly_C[7*i+0]+Poly_C[7*i+1]*(t-t0)+Poly_C[7*i+2]*pow((t-t0),2)+Poly_C[7*i+3]*pow((t-t0),3); //余弦函数
			}
		}
		else if(poly_num/6==8){
			EO[i]=Poly_C[8*i+0]+Poly_C[8*i+1]*(t-t0)+Poly_C[8*i+2]*pow((t-t0),2)+Poly_C[8*i+3]*pow((t-t0),3)+Poly_C[8*i+4]*sin((t-t0))+Poly_C[8*i+5]*sin(T(2.0)*(t-t0))+Poly_C[8*i+6]*sin(T(3.0)*(t-t0))+Poly_C[8*i+7]*sin(T(4.0)*(t-t0)); //傅里叶级数
		}
		else if(poly_num/6==10){
			int n=poly_num/6;
			if(i<3){
				EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3)+Poly_C[n*i+4]*cos(Poly_C[n*i+5]*(t-t0)+Poly_C[n*i+6])+Poly_C[n*i+7]*cos(Poly_C[n*i+8]*(t-t0)+Poly_C[n*i+9]); //余弦函数
			}
			else{
				EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
			}
		}
		else if(poly_num/6==13 || poly_num/6==16 || poly_num/6==22){
			int n=poly_num/6;
			EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
			for(int j=4;j<n;j+=3){
				EO[i]+=T(0.0001)*(Poly_C[n*i+j]*cos(6.2832*Poly_C[n*i+j+1]*(t-t0)+Poly_C[n*i+j+2]));
				//cout<<Poly_C[n*i+j]<<" "<<Poly_C[n*i+j+1]<<" "<<Poly_C[n*i+j+2]<<endl;
			}
		}
		else if(poly_num/6==25){
			int n=poly_num/6;
			EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
			for(int j=4;j<n-1;j++){
				if(j%2==0){
					EO[i]+=T(pow(-1.0,0))*Poly_C[n*i+j]*cos(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,0))*Poly_C[n*i+j]*sin(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		else if(poly_num/6==21){
			int n=poly_num/6;
			EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
			for(int j=4;j<n-1;j++){
				if(j%2==0){
					EO[i]+=T(pow(-1.0,0))*T(0.0001)*Poly_C[n*i+j]*cos(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,0))*T(0.0001)*Poly_C[n*i+j]*sin(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		else if(poly_num/6==15){
			int n=poly_num/6;
			EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
			for(int j=4;j<n-1;j++){
				if(j%2==0){
					EO[i]+=T(pow(-1.0,i+1))*T(0.0001)*Poly_C[n*i+j]*cos(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,i+1))*T(0.0001)*Poly_C[n*i+j]*sin(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		else if(poly_num/6==31 || poly_num/6==47 || poly_num/6>47){
			int n=poly_num/6;
			EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
			for(int j=4;j<n-1;j++){
				if(j<(4+n-1)/2){
					EO[i]+=T(pow(-1.0,0))*T(0.0001)*Poly_C[n*i+j]*cos(T(int((j-4)+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,0))*T(0.0001)*Poly_C[n*i+j]*sin(T(int((j-(4+n-1)/2)+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		else if(poly_num/6>10){
			//printf("%s\n","ddd");
			int n=poly_num/6;
			EO[i]=Poly_C[n*i+0]+Poly_C[n*i+1]*(t-t0)+Poly_C[n*i+2]*pow((t-t0),2)+Poly_C[n*i+3]*pow((t-t0),3);
			for(int j=4;j<n-1;j++){
				if(j%2==0){
					EO[i]+=T(pow(-1.0,i))*Poly_C[n*i+j]*cos(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,i))*Poly_C[n*i+j]*sin(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		else if(poly_num/6==4){
			EO[i]=Poly_C[4*i+0]+Poly_C[4*i+1]*(t-t0)+Poly_C[4*i+2]*pow((t-t0),2)+Poly_C[4*i+3]*pow((t-t0),3);
		}//*/
		//EO[i]=Poly_C[4*i+0]+Poly_C[4*i+1]*(t-t0)+Poly_C[4*i+2]*T(sin(t-t0))+Poly_C[4*i+3]*T(sin(T(2.0)*(t-t0)));
	}
}

//Jitter项
template <typename T>
void Get_polyEO_T1(T et, T Poly_C[poly_num], T EO[6]){
	for(int i=0;i<6;i++){
		T t=et;
		T t0=T(0.0);
		if(poly_num/6==7){
			if(i<6){
				//EO[i]=Poly_C[7*i+4]*sin(Poly_C[7*i+5]*(t-t0)+Poly_C[7*i+6]); //正弦函数
				EO[i]=Poly_C[7*i+4]*cos(Poly_C[7*i+5]*(t-t0)+Poly_C[7*i+6]); //余弦函数
			}
			else{
				EO[i]=T(0); 
			}
		}
		else if(poly_num/6==8){
			EO[i]=Poly_C[8*i+4]*sin((t-t0))+Poly_C[8*i+5]*sin(T(2.0)*(t-t0))+Poly_C[8*i+6]*sin(T(3.0)*(t-t0))+Poly_C[8*i+7]*sin(T(4.0)*(t-t0)); //傅里叶级数
		}
		else if(poly_num/6==10){
			int n=poly_num/6;
			if(i<3){
				//EO[i]=Poly_C[n*i+4]*sin(Poly_C[n*i+5]*(t-t0)+Poly_C[n*i+6])+Poly_C[n*i+7]*sin(Poly_C[n*i+8]*(t-t0)+Poly_C[n*i+9]); //正弦函数
				EO[i]=Poly_C[n*i+4]*cos(Poly_C[n*i+5]*(t-t0)+Poly_C[n*i+6])+Poly_C[n*i+7]*cos(Poly_C[n*i+8]*(t-t0)+Poly_C[n*i+9]); //余弦函数
			}
			else{
				EO[i]=T(0);
			}
		}
		else if(poly_num/6==13 || poly_num/6==16 || poly_num/6==22){
			int n=poly_num/6;
			EO[i]=T(0);
			for(int j=4;j<n;j+=3){
				EO[i]+=T(0.0001)*(abs(Poly_C[n*i+j])*Poly_C[n*i+j]*cos(6.2832*Poly_C[n*i+j+1]*(t-t0)+Poly_C[n*i+j+2]));
			}
		}
		else if(poly_num/6==22){
			int n=poly_num/6;
			int nt=(n-4)/3;
			EO[i]=T(0);
			for(int j=4;j<4+(n-4)/3;j+=1){
				EO[i]+=T(0.0001)*(Poly_C[n*i+j]*cos(6.2832*Poly_C[n*i+j+nt]*(t-t0)+Poly_C[n*i+j+2*nt]));
			}
		}
		else if(poly_num/6==25){
			int n=poly_num/6;
			EO[i]=T(0);
			for(int j=4;j<n-1;j++){
				if(j%2==0){
					EO[i]+=T(pow(-1.0,0))*Poly_C[n*i+j]*cos(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,0))*Poly_C[n*i+j]*sin(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		//else if(poly_num/6==22){
		// int n=poly_num/6;
		// EO[i]=T(0);
		// EO[i]+=Poly_C[n*i+4]*T(0.0000);
		// for(int j=5;j<n-1;j++){
		// if(j%2==1){
		// EO[i]+=T(0.0001)*Poly_C[n*i+j]*cos(T(int((j-5)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
		// }
		// else{
		// EO[i]+=T(0.0001)*Poly_C[n*i+j]*sin(T(int((j-5)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
		// }
		// }
		//}
		else if(poly_num/6==21){
			int n=poly_num/6;
			EO[i]=T(0.0);
			for(int j=4;j<n-1;j++){
				if(j%2==0){
					EO[i]+=T(pow(-1.0,1))*T(0.0001)*Poly_C[n*i+j]*cos(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,1))*T(0.0001)*Poly_C[n*i+j]*sin(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		else if(poly_num/6==15){
			int n=poly_num/6;
			EO[i]=T(0);
			for(int j=4;j<n-1;j++){
				if(j%2==0){
					EO[i]+=T(pow(-1.0,i+1))*T(0.0001)*Poly_C[n*i+j]*cos(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,i+1))*T(0.0001)*Poly_C[n*i+j]*sin(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		else if(poly_num/6==31 || poly_num/6==47 || poly_num/6>47){
			int n=poly_num/6;
			EO[i]=T(0);
			for(int j=4;j<n-1;j++){
				if(j<(4+n-1)/2){
					EO[i]+=T(pow(-1.0,0))*T(0.0001)*Poly_C[n*i+j]*cos(T(int((j-4)+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
				else{
					EO[i]+=T(pow(-1.0,0))*T(0.0001)*Poly_C[n*i+j]*sin(T(int((j-(4+n-1)/2)+1))*T(Poly_C[n*i+n-1])*(t-t0));
				}
			}
		}
		else if(poly_num/6>10){
			int n=poly_num/6;
			EO[i]=T(0);
			for(int j=4;j<n-1;j++){
				if(j%2==0){
					EO[i]+=Poly_C[n*i+j]*cos(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
					//EO[i]+=abs(Poly_C[n*i+j]);
				}
				else{
					EO[i]+=Poly_C[n*i+j]*sin(T(int((j-4)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
					//EO[i]+=abs(Poly_C[n*i+j]);
				}
			}
		}
		else if(poly_num/6==4){
			EO[i]=T(0);
		}//*/
	}
}


template <typename T>
void Get_IOJitter_T(T et, T Poly_C[poly_num], T IOJitter[2]){
	T t=et;
	T t0=T(0.0);
	for(int i=0;i<2;i++){
		int ii=i;
		//int ii=abs(1-i);
		int n=poly_num/2;
		IOJitter[i]=T(0);
		for(int j=0;j<n-1;j++){
			if(j%2==0){
				IOJitter[ii]-=Poly_C[n*i+j]*cos(T(int((j)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
			}
			else{
				IOJitter[ii]-=Poly_C[n*i+j]*sin(T(int((j)/2+1))*T(Poly_C[n*i+n-1])*(t-t0));
			}
		}
	}
}

template <typename T>
void Compu_ProjRES0(T observed_x, T observed_y, T et, T* Poly_C,T focal,T* point,T* resdual){
	T camera_R[3],camera_T[3];
	T EO_temp[6];
	T Poly_temp[poly_num];
	for(int i=0;i<poly_num;i++){
		Poly_temp[i]=T(Poly_C[i]);
	}
	Get_polyEO_T0(T(et),Poly_temp,EO_temp);
	for(int i=0;i<6;i++){
		T cam_temp=EO_temp[i];
		//T cam_temp=Poly_C[4*i+0]+Poly_C[4*i+1]*(et)+Poly_C[4*i+2]*sin(et)+Poly_C[4*i+3]*sin(2*et);
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

	//Apply second and fourth order radial distortion.
	//const T& l1 = 0;//camera[7];
	//const T& l2 = 0;//camera[8];
	//T r2 = xp*xp + yp*yp;
	T distortion = T(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

	// Compute final projected point position.
	//T focal = T(1.0);
	T predicted_x = focal * distortion * xp;
	T predicted_y = focal * distortion * yp;

	// The error is the difference between the predicted and observed position.
	T residuals[2];
	residuals[0] = predicted_x - T(observed_x);
	residuals[1] = predicted_y - T(observed_y);
	*resdual = sqrt(residuals[0]*residuals[0]+residuals[1]*residuals[1]);
}

template <typename T>
void Compu_ProjRES(T observed_x, T observed_y, T et, T* Poly_C,T focal,T* point,T* resdual,int mark){
	T camera_R[3],camera_T[3];
	T EO_temp[6];
	T Poly_temp[poly_num];
	for(int i=0;i<poly_num;i++){
		Poly_temp[i]=T(Poly_C[i]);
	}
	Get_polyEO_T(T(et),Poly_temp,EO_temp);
	for(int i=0;i<6;i++){
		T cam_temp=EO_temp[i];
		//T cam_temp=Poly_C[4*i+0]+Poly_C[4*i+1]*(et)+Poly_C[4*i+2]*sin(et)+Poly_C[4*i+3]*sin(2*et);
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

	//Apply second and fourth order radial distortion.
	//const T& l1 = 0;//camera[7];
	//const T& l2 = 0;//camera[8];
	//T r2 = xp*xp + yp*yp;
	T distortion = T(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

	// Compute final projected point position.
	//T focal = T(1.0);
	T predicted_x = focal * distortion * xp;
	T predicted_y = focal * distortion * yp;

	// The error is the difference between the predicted and observed position.
	T residuals[2];
	residuals[0] = predicted_x - T(observed_x);
	residuals[1] = predicted_y - T(observed_y);
	if(mark==0){
		*resdual = residuals[0];
	}
	else{
		*resdual = residuals[1];
	}
	//*resdual = residuals[1];//sqrt(residuals[0]*residuals[0]+residuals[1]*residuals[1]);
}

template <typename T>
void Compu_ProjRES(T observed_x, T observed_y, T et, T* Poly_C, T* Poly_IO, T focal,T* point,T* resdual){
	T camera_R[3],camera_T[3];
	T EO_temp[6];
	T Poly_temp[24];
	for(int i=0;i<24;i++){
		Poly_temp[i]=T(Poly_C[i]);
	}
	Get_polyEO_T_IO(T(et),Poly_temp,EO_temp);
	for(int i=0;i<6;i++){
		T cam_temp=EO_temp[i];
		//T cam_temp=Poly_C[4*i+0]+Poly_C[4*i+1]*(et)+Poly_C[4*i+2]*sin(et)+Poly_C[4*i+3]*sin(2*et);
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

	//Apply second and fourth order radial distortion.
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
	T residuals[2];
	residuals[0] = predicted_x - T(observed_x);
	residuals[1] = predicted_y - T(observed_y);
	*resdual = residuals[1];//sqrt(residuals[0]*residuals[0]+residuals[1]*residuals[1]);
}


void Compu_ProjRES1(double observed_x, double observed_y, double et, double* Poly_C,double focal,double* point,double* resdual){
	double camera_R[3],camera_T[3];
	for(int i=0;i<6;i++){
		double cam_temp=Poly_C[4*i+0]+Poly_C[4*i+1]*(et)+Poly_C[4*i+2]*pow((et),2)+Poly_C[4*i+3]*pow((et),3);
		//double cam_temp=Poly_C[4*i+0]+Poly_C[4*i+1]*(et)+Poly_C[4*i+2]*sin(et)+Poly_C[4*i+3]*sin(2*et);
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

	//Apply second and fourth order radial distortion.
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


//区域网平差（特征点）
int IO_adjustment(char* observetxt,char* EOfile,double** poly_IO,char* xulie_ID1,char* xulie_ID2,char* GTtxt,char* EOreFile){
	printf("Begin Block_adjustment!\n");

	//数据准备
	std::vector<double> match_point;
	std::vector<double> GT_initial;
	FILE *fp=fopen(observetxt,"r");
	double temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;
	int markC=-1;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf\n",&temp1,&temp2,&temp3,&temp4,&temp5,&temp6,&temp7,&temp8);
		match_point.push_back(temp1);
		match_point.push_back(temp2);
		match_point.push_back(temp3);
		match_point.push_back(temp4);
		match_point.push_back(temp5);
		if(int(temp1)!=markC){
			GT_initial.push_back(temp6);
			GT_initial.push_back(temp7);
			GT_initial.push_back(temp8);
			markC=temp1;
		}
	}
	fclose(fp);


	int image_num=2;
	int CCnum[2];
	CCnum[0]=160;CCnum[1]=200;
	double LR[2]; //外方位元素观测

	char* EOtxt = new char[80];
	double ** poly_EO = new double *[image_num];
	double ** poly_CE = new double *[image_num];
	//double ** poly_IO = new double *[image_num];
	double ** obs_EO = new double *[image_num];
	double ** obs_et = new double *[image_num];
	char* xulie_ID = new char[80];
	double *et0 = new double[2];
	for(int i=0;i<image_num;i++){
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else{
			xulie_ID=xulie_ID2;
		}

		//读取拟合结果
		sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID, "/polyCC.txt" );
		FILE* fp_eo=fopen(EOtxt,"r");
		poly_EO[i] = new double[24];
		poly_CE[i] = new double[24];
		//poly_IO[i] = new double[poly_num];
		memset(poly_EO[i],0,24*sizeof(double));
		memset(poly_CE[i],0,24*sizeof(double));
		memset(poly_IO[i],0,(poly_num)*sizeof(double));
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

			double awptemp[2][6][11] = {{{1.128e-07,1.585e-07,-2.56e-07,1.704e-07,-2.933e-07,1.59e-07,3.263e-07, 6.459e-07,-1.529e-07,-4.805e-07,1.504},   //俯仰角--航向视差决定（dr）
			{1.48e-07,-2.319e-07,2.739e-08,1.042e-07,9.967e-08,2.671e-08,3.034e-07, -3.468e-07,-2.957e-07,-2.783e-08,1.719},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}},
			{{0.02731e-05,-0.03084e-05,1.89e-05,-2.291e-05,-0.07449e-05,-0.09755e-05,0.06342e-05, 0.782e-05,0.06086e-05,0.029e-05,4.201},   //俯仰角--航向视差决定（dr）
			{0.00654e-05,0.0841e-05,0.04644e-05,-0.007466e-05,-0.1409e-05, -0.1224e-05, 0.1241e-05,0.1384e-05,-1.109e-05,3.282e-05,2.028},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}}};//*/
			/*double awptemp[2][6][11] = {{{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},   //俯仰角--航向视差决定（dr）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}},
			{{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},   //俯仰角--航向视差决定（dr）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}}};//*/

			for(int k=0;k<4;k++){
				fscanf(fp_eo," %lf",&temp);
				poly_EO[i][jj*24/6+k] = temp;
				poly_CE[i][jj*24/6+k] = temp;
				//printf("%lf ",poly_EO[i][jj*poly_num/6+k]);
			}
			for(int k=4;k<24/6;k++){
				//poly_EO[i][jj*poly_num/6+k] = jj<3 ? awptemp1[k-4] : awptemp2[k-4];
				poly_EO[i][jj*24/6+k] = awptemp[i][jj][k-4];
				poly_CE[i][jj*24/6+k] = 0;//poly_EO[i][jj*poly_num/6+k];
			}
			fscanf(fp_eo,"\n");
			//printf("\n");
		}
		fclose(fp_eo);


		//读取jitter初值
		/*double polyIOtemp[2][22] = {{1.128e-05,1.585e-05,-2.56e-05,1.704e-05,-2.933e-05,1.59e-05,3.263e-05, 6.459e-05,-1.529e-05,-4.805e-05,1.504,   //俯仰角--航向视差决定（dr）
		1.48e-05,-2.319e-05,0.2739e-05,1.042e-05,0.9967e-05,0.2671e-05,3.034e-05, -3.468e-05,-2.957e-05,-0.2783e-05,1.719},  //翻滚角--旁向视差决定（dc）
		{0.02731e-03,-0.03084e-03,1.89e-03,-2.291e-03,-0.07449e-03,-0.09755e-03,0.06342e-03, 0.782e-03,0.06086e-03,0.029e-03,4.201,     //俯仰角--航向视差决定（dr）
		0.00654e-03,0.0841e-03,0.04644e-03,-0.007466e-03,-0.1409e-03, -0.1224e-03, 0.1241e-03,0.1384e-03,-1.109e-03,3.282e-03,2.028}}; //翻滚角--旁向视差决定（dc）//*/ 
		double polyIOtemp[2][22] = {{0.0005563,-0.002222,0.0001566,0.0008861,-0.002301,0.008228,-0.0003971,-0.001284,-0.0008487,-0.001322,2.726,   //俯仰角--航向视差决定（dr）
			-0.0001989,0.000824,0.0002909,0.0004321,0.000642,0.0003807,-0.01415,0.0009293,0.0004385, -0.001592,1.591},  //翻滚角--旁向视差决定（dc）
		{0.0004071,-0.0002489,0.0008866,-0.001115,0.00289,-0.003135,0.00694,-0.03169,-0.01772,0.005625,2.074,     //俯仰角--航向视差决定（dr）
		0.002203,0.0008621,0.005024,-0.008662,-0.01613,0.04059,-0.001475,0.0001512,-1.196e-05,-0.0001854,3.384}}; //翻滚角--旁向视差决定（dc）//*/ 
		/*double polyIOtemp[2][34] = {{0.0003314,-0.0002566,-0.0003141,0.001853,-0.0002812,0.008543,-0.0005533,-0.0008615,-0.0009071,-0.0007003,-0.001914,0.002682,-0.0004561,0.003927,0.0001145,-0.0006252,2.698,   //俯仰角--航向视差决定（dr）
		-0.001826,6.244e-05,-0.00297,-0.0004808,-0.00244,-0.0005562,-0.002856,0.0005642,-0.002656,-0.000175,-0.002808,0.001761,-0.004559,0.003786,-0.01308,0.008182,0.7802},  //翻滚角--旁向视差决定（dc）
		{0.0004583,-3.776e-05,0.000316,-0.0008113,-0.0009096,0.0007942,0.001841,-0.001739,0.02259,-0.02592,-0.002932,0.02413,-3.142e-05,-0.0005255,0.0009952,0.0007662,1.685,     //俯仰角--航向视差决定（dr）
		-0.001604,-0.0007734,0.001022,-0.001215,-0.001495,0.001352,0.004074,-0.009567,-0.002866,0.004073,-0.0161,0.04055,0.002031,0.001455,-0.0009752,-0.0004053,1.693}};  //翻滚角--旁向视差决定（dc）//*/
		for(int k=0;k<poly_num;k++){
			poly_IO[i][k] = polyIOtemp[i][k];
		}

		//读取观测EO
		sprintf( EOtxt, "%s%s%s%s%s%s", EOfile, "/", xulie_ID, "/", xulie_ID, "_RED5_0.txt");
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

	int ground_num = match_point[5*(int(match_point.size()/5)-1)]+1;
	double ** ground_point = new double *[ground_num];
	for(int i=0;i<ground_num;i++){
		ground_point[i] = new double[3];
		for(int j=0;j<3;j++){
			ground_point[i][j]=GT_initial[i*3+j];
		}
	}

	double focal=-11995.48;

	/////////////////////////////////////////////////////光束法平差////////////////////////////////////////////////////////////
	Problem problem1;
	//共线条件方程
	for (int i=0; i<match_point.size()/5; i++) {
		ceres::CostFunction* cost_function =
			IO_Projcost::Create(
			match_point[5*i+1],
			match_point[5*i+2],
			match_point[5*i+3]-et0[int(match_point[5*i+4])],
			focal);
		problem1.AddResidualBlock(cost_function,
			new ceres::CauchyLoss(1.0),
			//NULL,
			ground_point[int(match_point[5*i])],
			poly_EO[int(match_point[5*i+4])],
			poly_IO[int(match_point[5*i+4])]);
	}

	//轨道参数方程
	for(int ii=0;ii<2;ii++){
		for(int i=0;i<CCnum[ii];i++){
			double * eo_temp=new double[6];
			for(int j=0;j<6;j++){
				eo_temp[j]=obs_EO[ii][i*6+j];
			}
			ceres::CostFunction* cost_function =
				Orbitalcost0::Create(
				obs_et[ii][i]-et0[ii],
				eo_temp);
			problem1.AddResidualBlock(cost_function,
				new ceres::CauchyLoss(1.0),
				//NULL,
				poly_EO[ii]);
		}
	}

	ceres::Solver::Options options1;
	ConfigureCeresThreads(options1);
	options1.linear_solver_type = ceres::DENSE_SCHUR;
	options1.minimizer_progress_to_stdout = true;
	options1.max_num_iterations=100;
	options1.function_tolerance=1e-10;
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	std::cout << summary1.FullReport() << "\n";/**/


	double *resdual = new double[ground_num];
	int count=0;
	double res,temp;
	res=0;
	int js=0;
	/*for(int i=0;i<match_point.size()/5;i++){
	if(match_point[5*i]==count){
	Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
	res+=temp*temp;
	js+=1;
	if(i==match_point.size()/5-1){
	resdual[count]=sqrt(res/js);
	}
	}
	else{
	resdual[count]=sqrt(res/js);
	res=0;
	js=0;
	count++;
	Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
	res+=temp*temp;
	}
	}*/
	for(int i=0;i<match_point.size()/5;i++){
		if(match_point[5*i]==count){
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],poly_IO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
			res=temp;
			js+=1;
			if(i==match_point.size()/5-1){
				resdual[count]=res;
			}
		}
		else{
			resdual[count]=res;
			res=0;
			js=0;
			count++;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],poly_IO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
			res=temp;
		}
	}

	//输出地面点坐标
	FILE *fp_gt=fopen(GTtxt,"w");
	fprintf(fp_gt,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		fprintf(fp_gt,"%lf %lf %lf %lf\n",ground_point[i][0],ground_point[i][1],ground_point[i][2],resdual[i]);
		//delete[] (ground_point[i]);
	}
	fclose(fp_gt);

	//输出外方位元素
	for(int i=0;i<image_num;i++){
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else if(i==1){
			xulie_ID=xulie_ID2;
		}
		char* EOretxt = new char[80];
		sprintf(EOretxt, "%s%s%s%s", EOreFile, "/", xulie_ID, "_EOre.txt" );
		FILE *fp_eo=fopen(EOretxt,"w");
		for(int j=0;j<6;j++){
			for(int k=0;k<24/6+1;k++){
				if(k==0){
					fprintf(fp_eo,"%.12lf",et0[i]);
				}
				else{
					fprintf(fp_eo," %.12lf",poly_EO[i][j*24/6+k-1]);
				}
			}
			fprintf(fp_eo,"\n");
			//delete[] (poly_EO[i]);
			//delete[] (poly_CE[i]);
			//fprintf(fp_eo,"%.12lf %.12lf %.12lf %.12lf %.12lf\n",et0[i],poly_EO[i][j*4+0],poly_EO[i][j*4+1],poly_EO[i][j*4+2],poly_EO[i][j*4+3]);
		}
		fclose(fp_eo);
	}

	std::vector<double>().swap(match_point);
	std::vector<double>().swap(GT_initial);

	delete[] resdual;
	delete[] ground_point;

	return 0;
}
int Block_adjustment(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* GTtxt,char* EOreFile){
	printf("Begin Block_adjustment!\n");

	//数据准备
	std::vector<double> match_point;
	std::vector<double> GT_initial;
	FILE *fp=fopen(observetxt,"r");
	double temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;
	int markC=-1;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf\n",&temp1,&temp2,&temp3,&temp4,&temp5,&temp6,&temp7,&temp8);
		match_point.push_back(temp1);
		match_point.push_back(temp2);
		match_point.push_back(temp3);
		match_point.push_back(temp4);
		match_point.push_back(temp5);
		if(int(temp1)!=markC){
			GT_initial.push_back(temp6);
			GT_initial.push_back(temp7);
			GT_initial.push_back(temp8);
			markC=temp1;
		}
	}
	fclose(fp);


	int image_num=2;
	int CCnum[2];
	CCnum[0]=80;CCnum[1]=160;
	//CCnum[0]=160;CCnum[1]=200;
	double LR[2]; //外方位元素观测

	char* EOtxt = new char[80];
	double ** poly_EO = new double *[image_num];
	double ** poly_CE = new double *[image_num];
	double ** obs_EO = new double *[image_num];
	double ** obs_et = new double *[image_num];
	char* xulie_ID = new char[80];
	double *et0 = new double[2];
	for(int i=0;i<image_num;i++){
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else{
			xulie_ID=xulie_ID2;
		}

		//读取拟合结果
		sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID, "/polyCC.txt" );
		FILE* fp_eo=fopen(EOtxt,"r");
		poly_EO[i] = new double[poly_num];
		poly_CE[i] = new double[poly_num];
		memset(poly_EO[i],0,poly_num*sizeof(double));
		memset(poly_CE[i],0,poly_num*sizeof(double));
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

			/*double awptemp[2][6][18] = {{
			{
			0.007815,1.334102,-2.571324,
			0.004863,2.935024,0.514006,
			0.003402,2.668204,2.843438,
			0.003346,1.067281,1.348238,
			0.003150,1.200692,0.560713,
			0.002609,3.068434,2.678661
			},
			{
			0.010699,1.067281,2.094353,
			0.006379,0.933871,-1.500696,
			0.005639,1.200692,2.300535,
			0.003527,0.800461,-1.296026,
			0.002491,1.467512,2.315042,
			0.002208,1.334102,1.971862

			},
			//{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			//{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0}},{
			{
			0.021812,1.300350,0.985500,
			0.016350,1.560419,-1.257023,
			0.012035,1.365367,-0.585818,
			0.009292,1.625437,3.105977,
			0.008264,1.235332,2.921415,
			0.006960,1.430384,-0.380631
			},
			{
			0.025798,1.560419,-1.465140,
			0.018400,1.625437,3.005049,
			0.009032,1.495402,0.348912,
			0.006611,1.040280,1.347964,
			0.006415,1.430384,-0.748693,
			0.005444,1.105297,-0.123728
			},
			//{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			//{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0}}};//*/

			/*double awptemp[2][6][18] = {{
			{ 0.000988,31.729489,1.402654,
			0.000790,16.797965,-2.204673,
			0.000677,26.130167,3.115754,
			0.000624,18.664405,0.218835,
			0.000386,14.931524,-2.720086,
			0.000358,39.195251,1.242518
			},
			{ 0.001870,31.729489,1.658592,
			0.001153,29.863048,-2.368641,
			0.000844,16.797965,-2.744809,
			0.000791,33.595929,0.443965,
			0.000777,26.130167,-0.026181,
			0.000680,27.996608,-2.502574

			},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0}},{
			//{ 0.000292,1.816706,0.460945,
			// 0.000287,6.358470,-1.471041,
			// 0.000276,5.450117,-0.291258,
			// 0.000221,21.800467,1.222533,
			// 0.000211,18.167056,-0.364409,
			// 0.000196,19.075409,-2.274215
			//},
			//{ 0.000339,1.816706,-2.728376,
			// 0.000262,22.708820,1.247743,
			// 0.000238,5.450117,3.133166,
			// 0.000210,21.800467,-2.328557,
			// 0.000204,20.892114,1.160938,
			// 0.000201,3.633411,1.047107
			//},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0},
			{0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0,   0.0,0.0,0.0}}};//*/


			double awptemp[2][6][17] = {{
				{3.634e-05,4.522e-05,2.772e-05,7.467e-05,0.0001923,4.158e-05,-0.0006079,-0.0008739,-0.00024,-0.000306,-0.0002766,-0.0003867,-0.0005332,-0.0008854,-5.869e-05,-2.978e-05,4.433},   //俯仰角--航向视差决定（dr）
				{0.0002233,4.72e-05,0.0003738,0.0001153,0.0004335,0.0003987,9.218e-05,-0.0001359,0.0007587,-0.000279,-0.001557,-0.002156,1.324e-05,2.567e-05,-0.0001791,-8.649e-05,5.183},  //翻滚角--旁向视差决定（dc）
				//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
				{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
				{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
				{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
				{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
						//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
					{0.0004933,-0.0001916,0.0003532,-0.0005158,0.0001704,-0.0004236,0.0003087,-0.0003022,-0.0001469,-0.0003924,-5.248e-05,-0.0003306,-3.845e-05,-2.013e-05,-0.0001659,0.0001747,0.803},     //俯仰角--航向视差决定（dr）
					//{-0.0001713,6.951e-05,-0.0002915,0.0002894,-0.0001578,0.0001237,-5.635e-05,-9.241e-05,0.0001595,0.0002574,3.755e-05,0.0001828,0.0001332,-0.0001442,0.0001543,3.187e-05,0.843},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/
					/*double awptemp[2][6][11] = {{{1.128e-07,1.585e-07,-2.56e-07,1.704e-07,-2.933e-07,1.59e-07,3.263e-07, 6.459e-07,-1.529e-07,-4.805e-07,1.504},   //俯仰角--航向视差决定（dr）
					{1.48e-07,-2.319e-07,2.739e-08,1.042e-07,9.967e-08,2.671e-08,3.034e-07, -3.468e-07,-2.957e-07,-2.783e-08,1.719},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}},
					{{0.02731e-05,-0.03084e-05,1.89e-05,-2.291e-05,-0.07449e-05,-0.09755e-05,0.06342e-05, 0.782e-05,0.06086e-05,0.029e-05,4.201},   //俯仰角--航向视差决定（dr）
					{0.00654e-05,0.0841e-05,0.04644e-05,-0.007466e-05,-0.1409e-05, -0.1224e-05, 0.1241e-05,0.1384e-05,-1.109e-05,3.282e-05,2.028},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}}};//*/
					/*double awptemp[2][6][12] = {
					{{0.000749,2.67347,-2.40670953,0.000657, 2.970532, 0.2334134,0.00056,4.15874,-2.9385328,0.0007842,5.04990494,1.3545923},   //俯仰角--航向视差决定（dr）
					{5.5138e-04,2.6734,-2.265869,0.00067, 4.1587, -0.3496605,0.00119,4.7528,-2.4918767,0.001825438,5.049904,1.38235},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0},   //对dr、dc影响相对不明确
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0}},
					{{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0},   //俯仰角--航向视差决定（dr）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0},   //对dr、dc影响相对不明确
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0}}};//*/
					/*double awptemp[2][6][11] = {
					{{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},   //俯仰角--航向视差决定（dr）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}},
					{{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},   //俯仰角--航向视差决定（dr）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
					{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}}};//*/
					/*double awptemp[6][6] = {{0.000001,8,-1,     0.00000001, 100, 0},   //俯仰角--航向视差决定（dr）
					{0.0000005,6,-1,     0.00000001, 100, 0},  //翻滚角--旁向视差决定（dc）
					{0.0000001,50,0,         0.0, 0.0, 0.0},   //对dr、dc影响相对不明确
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},
					{0.0,0.0,0.0,           0.0, 0.0, 0.0}};//*/
					/*double awptemp[6][6] = {{0.000001,9.5,1.57,     0.0000001, 100.0, 0.0},  //俯仰角--航向视差决定（dr）
					{0.000001,7.8,1.57,     0.0000001, 100.0, 0.0},  //翻滚角--旁向视差决定（dc）
					{0.0000000,0.0,0.0,     0.0, 0.0, 0.0},  //对dr、dc影响相对不明确
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},
					{0.0,0.0,0.0,           0.0, 0.0, 0.0}};//*/
					/*double awptemp[2][6][6] ={ 
					{{0.0,0.0,0.0,           0.0, 0.0, 0.0},  //俯仰角--航向视差决定（dr）
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},  //对dr、dc影响相对不明确
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},
					{0.0,0.0,0.0,           0.0, 0.0, 0.0}},
					{{0.0,0.0,0.0,           0.0, 0.0, 0.0},  //俯仰角--航向视差决定（dr）
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},  //翻滚角--旁向视差决定（dc）
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},  //对dr、dc影响相对不明确
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},
					{0.0,0.0,0.0,           0.0, 0.0, 0.0},
					{0.0,0.0,0.0,           0.0, 0.0, 0.0}}
					};//*/
					for(int k=0;k<4;k++){
						fscanf(fp_eo," %lf",&temp);
						poly_EO[i][jj*poly_num/6+k] = temp;
						poly_CE[i][jj*poly_num/6+k] = temp;
						//printf("%lf ",poly_EO[i][jj*poly_num/6+k]);
					}
					for(int k=4;k<poly_num/6;k++){
						//poly_EO[i][jj*poly_num/6+k] = jj<3 ? awptemp1[k-4] : awptemp2[k-4];
						poly_EO[i][jj*poly_num/6+k] = (k-4 < 17) ? awptemp[i][jj][k-4] : 0.0;
						poly_CE[i][jj*poly_num/6+k] = 0;//poly_EO[i][jj*poly_num/6+k];
					}
					fscanf(fp_eo,"\n");
					//printf("\n");
		}
		fclose(fp_eo);

		//读取观测EO
		sprintf( EOtxt, "%s%s%s%s%s%s", EOfile, "/", xulie_ID, "/", xulie_ID, "_RED5_0.txt");
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

	int ground_num = match_point[5*(int(match_point.size()/5)-1)]+1;
	double ** ground_point = new double *[ground_num];
	for(int i=0;i<ground_num;i++){
		ground_point[i] = new double[3];
		for(int j=0;j<3;j++){
			ground_point[i][j]=GT_initial[i*3+j];
		}
	}

	double focal=-11995.48;

	/////////////////////////////////////////////////////光束法平差////////////////////////////////////////////////////////////
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
			MakeBaCauchyLoss(),
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
				MakeBaCauchyLoss(),
				//NULL,
				poly_EO[ii]);
		}
	}

	ceres::Solver::Options options1;
	ConfigureCeresThreads(options1);
	options1.linear_solver_type = ceres::DENSE_SCHUR;
	options1.minimizer_progress_to_stdout = true;
	options1.max_num_iterations=ba_cfg().block_max_iterations;
	options1.function_tolerance=ba_cfg().function_tolerance;
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	std::cout << summary1.FullReport() << "\n";/**/


	double *resdual = new double[ground_num];
	int count=0;
	double res,temp;
	res=0;
	int js=0;
	/*for(int i=0;i<match_point.size()/5;i++){
	if(match_point[5*i]==count){
	Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
	res+=temp*temp;
	js+=1;
	if(i==match_point.size()/5-1){
	resdual[count]=sqrt(res/js);
	}
	}
	else{
	resdual[count]=sqrt(res/js);
	res=0;
	js=0;
	count++;
	Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
	res+=temp*temp;
	}
	}*/
	for(int i=0;i<match_point.size()/5;i++){
		if(match_point[5*i]==count){
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp, 0);
			res=temp;
			js+=1;
			if(i==match_point.size()/5-1){
				resdual[count]=res;
			}
		}
		else{
			resdual[count]=res;
			res=0;
			js=0;
			count++;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp, 0);
			res=temp;
		}
	}

	//输出地面点坐标
	FILE *fp_gt=fopen(GTtxt,"w");
	fprintf(fp_gt,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		fprintf(fp_gt,"%lf %lf %lf %lf\n",ground_point[i][0],ground_point[i][1],ground_point[i][2],resdual[i]);
		//delete[] (ground_point[i]);
	}
	fclose(fp_gt);

	//输出外方位元素
	for(int i=0;i<image_num;i++){
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else if(i==1){
			xulie_ID=xulie_ID2;
		}
		char* EOretxt = new char[80];
		sprintf(EOretxt, "%s%s%s%s", EOreFile, "/", xulie_ID, "_EOre.txt" );
		FILE *fp_eo=fopen(EOretxt,"w");
		for(int j=0;j<6;j++){
			for(int k=0;k<poly_num/6+1;k++){
				if(k==0){
					fprintf(fp_eo,"%.12lf",et0[i]);
				}
				else{
					fprintf(fp_eo," %.12lf",poly_EO[i][j*poly_num/6+k-1]);
				}
			}
			fprintf(fp_eo,"\n");
			//delete[] (poly_EO[i]);
			//delete[] (poly_CE[i]);
			//fprintf(fp_eo,"%.12lf %.12lf %.12lf %.12lf %.12lf\n",et0[i],poly_EO[i][j*4+0],poly_EO[i][j*4+1],poly_EO[i][j*4+2],poly_EO[i][j*4+3]);
		}
		fclose(fp_eo);
	}

	std::vector<double>().swap(match_point);
	std::vector<double>().swap(GT_initial);

	delete[] resdual;
	delete[] ground_point;

	return 0;
}

int Jitter_adjustment(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* GTtxt,char* EOreFile){
	printf("Begin Jitter_adjustment!\n");

	//数据准备
	std::vector<double> match_point;
	std::vector<double> GT_initial;
	FILE *fp=fopen(observetxt,"r");
	double temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;
	int markC=-1;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf\n",&temp1,&temp2,&temp3,&temp4,&temp5,&temp6,&temp7,&temp8);
		match_point.push_back(temp1);
		match_point.push_back(temp2);
		match_point.push_back(temp3);
		match_point.push_back(temp4);
		match_point.push_back(temp5);
		if(int(temp1)!=markC){
			GT_initial.push_back(temp6);
			GT_initial.push_back(temp7);
			GT_initial.push_back(temp8);
			markC=temp1;
		}
	}
	fclose(fp);


	int image_num=2;
	int CCnum[2];
	//CCnum[0]=80;CCnum[1]=160;
	CCnum[0]=160;CCnum[1]=200;
	double LR[2]; //外方位元素观测

	char* EOtxt = new char[80];
	char* EOFtxt = new char[80];
	double ** poly_EO = new double *[image_num];
	//double ** poly_CE = new double *[image_num];
	double ** obs_EO = new double *[image_num];
	double ** obs_et = new double *[image_num];
	char* xulie_ID = new char[80];
	double *et0 = new double[2];

	for(int i=0;i<image_num;i++){
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else{
			xulie_ID=xulie_ID2;
		}

		//double awptemp[2][6][poly_num/6-4];
		double awptemp[2][6][100];
		//读取拟合结果
		sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID, "/polyCC.txt" );
		//sprintf(EOtxt, "%s%s%s%s", EOreFile, "/", xulie_ID, "_EOre.txt");
		FILE* fp_eo=fopen(EOtxt,"r");

		sprintf( EOFtxt, "%s%s%s%s%s%s", EOfile, "/", xulie_ID, "/",xulie_ID,"_F.txt");
		FILE* fp_eof=fopen(EOFtxt,"r");
		if(fp_eof == NULL){
			printf("未找到震颤频率文件：%s，跳过Jitter_adjustment\n", EOFtxt);
			if(fp_eo) fclose(fp_eo);
			return -1;
		}
		poly_EO[i] = new double[poly_num];
		//poly_CE[i] = new double[poly_num];
		memset(poly_EO[i],0,poly_num*sizeof(double));
		//memset(poly_CE[i],0,poly_num*sizeof(double));
		double temp,temp1;
		int jj;
		for(int j=0;j<6;j++){
			if(j<3){
				jj=j+3;
			}
			else{
				jj=j-3;
			}
			//jj=j;
			fscanf(fp_eo,"%lf",&temp);
			et0[i]=temp;
			//double awptemp[3] = {36,8,1.5};
			//double awptemp1[3] = {0.0000001,6.5,1.57};
			//double awptemp1[6] = {0.0000001,6.5,-1.57,0,39,0};
			//double awptemp1[6] = {0.0000001,6.5,-1.57,0.0,50,0};
			//double awptemp2[6] = {0.0,0.0,0.0,0.0,0.0,0.0};

			for(int k=0;k<poly_num/6-4;k++){
				if(jj>=2){
					awptemp[i][jj][k]=0;
				}
				else{
					fscanf(fp_eof,"%lf",&temp);
					awptemp[i][jj][k]=temp;
					//cout<<temp<<endl;
				}
			}
			/*double awptemp[2][6][7] = {{
			{0.0004655e-04,-0.002129e-04,0.0001784e-04,0.001003e-04,-0.001086e-04,0.008373e-04,2.713},   //俯仰角--航向视差决定（dr）
			{-0.0001327e-04,-0.0008738e-04,0.001366e-04,0.0003895e-04,-0.01426e-04,0.0009575e-04,2.121},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0}},{
			{-0.00004204e-04,-0.002113e-04,-0.0004778e-04,-0.0009263e-04,0.02226e-04,-0.02574e-04,2.801},     //俯仰角--航向视差决定（dr）
			{0.002213e-04,0.0008987e-04,0.004961e-04,-0.00866e-04,-0.01609e-04,0.04058e-04,3.384},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0}}};//*/


			/*double awptemp[2][6][43] = {{
			{0.0004009,0.0004194,-0.002381,-0.003686,0.0002201,-0.0003876,0.0001056,-0.003751,0.005036,0.0005932,8.464e-05,-2.523e-05,0.0002135,-7.949e-06,-0.0002394,-4.815e-05,-0.0001248,-0.0001693,-0.0002237,-0.0002255,0.0002358,
			-0.0005129,-0.0005287,0.001595,0.007715,-0.0006818,0.00162,4.149e-05,0.001213,0.0005415,-0.0001221,-0.0005406,-0.0005389,-0.0002147,-0.0002852,-8.76e-05,-0.0004069,0.0002774,-0.0001727,0.0001025,0.000112,8.477e-06,
			2.007},   //俯仰角--航向视差决定（dr）
			{-0.000392,9.096e-05,-0.0003999,0.001577,0.0002529,-0.01347,-0.002072,0.0009997,0.0009478,-0.0001721,0.0002492,-0.0003977,0.001503,0.000925,0.001278,0.0001206,-0.0002233,0.001015,0.0005901,0.0006226,0.0002359,
			0.0006546,-0.0005406,0.0003534,-0.0001544,0.001618,-0.0007473,-0.001293,0.0001784,-0.000429,-0.001259,-0.0006894,0.001215,0.0009378,0.0002096,0.0002593,-0.0009739,0.0002273,-0.0002227,0.0002591,4.854e-05,-0.0002077,
			1.066},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}},

			{
			{0.00062,0.0002824,-0.0007695,0.001402,0.02229,-0.002533,-0.0005994,0.0008303,0.002671,4.099e-05,0.0005245,-0.0005703,-6.892e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
			-0.0001,-0.0002327,0.0005865,-0.001637,-0.02638,0.02441,-0.0005232,0.001181,-0.000693,0.006643,0.0004317,0.0002313,-0.0001916,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
			1.641},     //俯仰角--航向视差决定（dr）
			{-0.001575,0.0009531,-0.001311,0.00432,-0.002917,-0.01583,0.002203,-0.0009083,-8.466e-05,-0.0005046,-0.0004745,0.0003389,9.094e-05,-0.0004661,0.0002225,0.0003136,-0.0001235,-0.0002506,9.702e-05,0.0002176,-0.0003991,
			-0.001081,-0.001144,0.001372,-0.009551,0.003909,0.04085,0.001256,-0.0003842,-0.0001465,-0.0005369,0.0005603,0.0006489,-0.0003811,0.0001704,6.514e-06,-0.000145,-0.0003699,0.000637,-0.0002251,-0.0001866,-0.0001185,
			1.649},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}}};//*/

			/*double awptemp[2][6][27] = {{
			{0.0004009,0.0004194,-0.002381,-0.003686,0.0002201,-0.0003876,0.0001056,-0.003751,0.005036,0.0005932,8.464e-05,-2.523e-05,0.0002135,-0.0005129,-0.0005287,0.001595,0.007715,-0.0006818,0.00162,4.149e-05,0.001213,0.0005415,-0.0001221,-0.0005406,-0.0005389,-0.0002147,2.007},   //俯仰角--航向视差决定（dr）
			{-0.0004909,3.272e-05,-0.0004567,0.001492,0.0001891,-0.0136,-0.002258,0.0008857,0.0008859,-0.0001803,0.000257,-0.0004797,0.001456,0.0007246,-0.0005309,0.0004027,-0.0001797,0.001565,-0.00104,-0.001335,0.0002126,-0.0004075,-0.00118,-0.0007308,0.0009191,0.000754,1.066},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{
			{0.00062,0.0002824,-0.0007695,0.001402,0.02229,-0.002533,-0.0005994,0.0008303,0.002671,4.099e-05,0.0005245,-0.0005703,-6.892e-05,-0.0001,-0.0002327,0.0005865,-0.001637,-0.02638,0.02441,-0.0005232,0.001181,-0.000693,0.006643,0.0004317,0.0002313,-0.0001916,1.641},     //俯仰角--航向视差决定（dr）
			{-0.001609,0.0008957,-0.00129,0.004337,-0.002953,-0.01576,0.002219,-0.0009183,-8.702e-05,-0.0005129,-0.0004834,0.0003415,8.075e-05,-0.001129,-0.001111,0.001412,-0.00963,0.003923,0.0409,0.001248,-0.0003858,-0.0001331,-0.0005332,0.0005741,0.0006108,-0.0003155,1.649},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/

			/*double awptemp[2][6][21] = {{
			{-0.0005729e-04,-0.0002961e-04,0.0004135e-04,0.0001615e-04,-0.0005218e-04,0.0004305e-04,0.0001756e-04,-0.0007241e-04,-0.0004324e-04,0.001404e-04,0.001512e-04,-0.001684e-04,-0.0007336e-04,0.0087e-04,0.0002667e-04,-0.002061e-04,-0.0006462e-04,0.00391e-04,-0.00008126e-04,-0.0003683e-04,1.16},   //俯仰角--航向视差决定（dr）
			{-0.002423e-04,0.001974e-04,-0.002341e-04,0.001052e-04,-0.001678e-04,0.00029e-04,-0.002159e-04,0.0009713e-04,-0.002054e-04,-0.00004603e-04,-0.002353e-04,0.001705e-04,-0.004257e-04,0.003604e-04,-0.01312e-04,0.007822e-04,-0.0006462e-04,0.00391e-04,-0.0000626e-04,-0.0007683e-04,0.7809},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{
			{0.0004401e-04,0.0001151e-04,0.0006552e-04,-0.001447e-04,0.002958e-04,-0.002699e-04,0.006972e-04,-0.03162e-04,-0.01788e-04,0.005648e-04,-0.0008349e-04,0.00102e-04,-0.003448e-04,0.003026e-04,0.006106e-04,0.002235e-04,-0.0006462e-04,0.00391e-04,-0.0000226e-04,-0.0002683e-04,2.074},     //俯仰角--航向视差决定（dr）
			{-0.000367e-04,-0.001179e-04,0.001732e-04,0.0006322e-04,-0.001823e-04,0.001656e-04,0.004749e-04,-0.008912e-04,-0.003174e-04,0.004537e-04,-0.0159e-04,0.04108e-04,0.002175e-04,0.001707e-04,-0.00103e-04,-0.0001295e-04,-0.0006462e-04,0.00391e-04,-0.0000726e-04,-0.0007683e-04,1.692},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/
			/*double awptemp[2][6][21] = {{
			{-9.92141173118823e-09,0.000709651920961839e-04,0.00183007492828545e-04,-0.00804855773723921e-04,-0.0005729e-04,-0.0002961e-04,0.0004135e-04,0.0001615e-04,-0.0005218e-04,0.0004305e-04,0.0001756e-04,-0.0007241e-04,-0.0004324e-04,0.001404e-04,0.001512e-04,-0.001684e-04,-0.0007336e-04,0.0087e-04,0.0002667e-04,-0.002061e-04,1.16},   //俯仰角--航向视差决定（dr）
			{-9.09719694891126e-09,0.000834810178944602e-04,-0.00254548591014061e-04,0.00438697190237473e-04,-0.001826e-04,0.00006244e-04,-0.00297e-04,-0.0004808e-04,-0.00244e-04,-0.0005562e-04,-0.002856e-04,0.0005642e-04,-0.002656e-04,-0.000175e-04,-0.002808e-04,0.001761e-04,-0.004559e-04,0.003786e-04,-0.01308e-04,0.008182e-04,0.7802},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{
			{-1.37760088694656e-09,0.000150412988031824e-04,0.00117733852487890e-04,0.00220330031020204e-04,-0.0001704e-04,0.0006463e-04,-0.0006055e-04,0.00006056e-04,-0.0007094e-04,0.0007551e-04,-0.001689e-04,0.0006904e-04,-0.0002968e-04,0.00236e-04,0.0317e-04,-0.01286e-04,0.01499e-04,0.01621e-04,-0.0005067e-04,-0.002212e-04,1.419},     //俯仰角--航向视差决定（dr）
			{-0.000320338940121263e-04,0.00433226786266934e-04,-0.0173198374939716e-04,0.0250356702980821e-04,-0.001604e-04,-0.0007734e-04,0.001022e-04,-0.001215e-04,-0.001495e-04,0.001352e-04,0.004074e-04,-0.009567e-04,-0.002866e-04,0.004073e-04,-0.0161e-04,0.04055e-04,0.002031e-04,0.001455e-04,-0.0009752e-04,-0.0004053e-04,1.693},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/

			/*double awptemp[2][6][17] = {{
			{-0.0005729e-04,-0.0002961e-04,0.0004135e-04,0.0001615e-04,-0.0005218e-04,0.0004305e-04,0.0001756e-04,-0.0007241e-04,-0.0004324e-04,0.001404e-04,0.001512e-04,-0.001684e-04,-0.0007336e-04,0.0087e-04,0.0002667e-04,-0.002061e-04,1.16},   //俯仰角--航向视差决定（dr）
			{-0.001826e-04,0.00006244e-04,-0.00297e-04,-0.0004808e-04,-0.00244e-04,-0.0005562e-04,-0.002856e-04,0.0005642e-04,-0.002656e-04,-0.000175e-04,-0.002808e-04,0.001761e-04,-0.004559e-04,0.003786e-04,-0.01308e-04,0.008182e-04,0.7802},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{
			{-0.0001704e-04,0.0006463e-04,-0.0006055e-04,0.00006056e-04,-0.0007094e-04,0.0007551e-04,-0.001689e-04,0.0006904e-04,-0.0002968e-04,0.00236e-04,0.0317e-04,-0.01286e-04,0.01499e-04,0.01621e-04,-0.0005067e-04,-0.002212e-04,1.419},     //俯仰角--航向视差决定（dr）
			{-0.001604e-04,-0.0007734e-04,0.001022e-04,-0.001215e-04,-0.001495e-04,0.001352e-04,0.004074e-04,-0.009567e-04,-0.002866e-04,0.004073e-04,-0.0161e-04,0.04055e-04,0.002031e-04,0.001455e-04,-0.0009752e-04,-0.0004053e-04,1.693},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/


			/*double awptemp[2][6][17] = {{
			{-0.0005729e-04,-0.0002961e-04,0.0004135e-04,0.0001615e-04,-0.0005218e-04,0.0004305e-04,0.0001756e-04,-0.0007241e-04,-0.0004324e-04,0.001404e-04,0.001512e-04,-0.001684e-04,-0.0007336e-04,0.0087e-04,0.0002667e-04,-0.002061e-04,1.16},   //俯仰角--航向视差决定（dr）
			{-0.002423e-04,0.001974e-04,-0.002341e-04,0.001052e-04,-0.001678e-04,0.00029e-04,-0.002159e-04,0.0009713e-04,-0.002054e-04,-0.00004603e-04,-0.002353e-04,0.001705e-04,-0.004257e-04,0.003604e-04,-0.01312e-04,0.007822e-04,0.7809},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{
			{0.0004401e-04,0.0001151e-04,0.0006552e-04,-0.001447e-04,0.002958e-04,-0.002699e-04,0.006972e-04,-0.03162e-04,-0.01788e-04,0.005648e-04,-0.0008349e-04,0.00102e-04,-0.003448e-04,0.003026e-04,0.006106e-04,0.002235e-04,2.074},     //俯仰角--航向视差决定（dr）
			{-0.000367e-04,-0.001179e-04,0.001732e-04,0.0006322e-04,-0.001823e-04,0.001656e-04,0.004749e-04,-0.008912e-04,-0.003174e-04,0.004537e-04,-0.0159e-04,0.04108e-04,0.002175e-04,0.001707e-04,-0.00103e-04,-0.0001295e-04,1.692},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/  

			/*double awptemp[2][6][17] = {{
			//{0.0003314,-0.0002566,-0.0003141,0.001853,-0.0002812,0.008543,-0.0005533,-0.0008615,-0.0009071,-0.0007003,-0.001914,0.002682,-0.0004561,0.003927,0.0001145,-0.0006252,2.698},   //俯仰角--航向视差决定（dr）
			//{0.0004339,-0.0004172,0.0005724,-0.0003306,-0.001764,0.001526,-0.002326,0.008282,-2.011e-05,-0.0005577,-0.0005646,0.001372,0.0004308,0.0004057,-0.003236,0.001489,2.044},
			//{0.0004016,-4.263e-05,-0.0002756,0.001983,0.0001579,0.008406,-0.000509,-0.0006363,-0.001025,-0.0006226,-0.001861,0.002606,-0.0006314,0.004224,0.0002532,-0.0009111,2.698},
			{0.0002116,-0.0001174,-0.001676,0.001917,-0.001145,-0.001706,-0.000537,0.001561,-0.000313,0.0005562,0.004422,0.001748,0.001011,-0.0007284,0.000561,-0.0009366,3.099},
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},

			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{-0.002423,0.001974,-0.002341,0.001052,-0.001678,0.00029,-0.002159,0.0009713,-0.002054,-4.603e-05,-0.002353,0.001705,-0.004257,0.003604,-0.01312,0.007822,0.7809},
			//{-0.0007107,-0.0003812,-0.0001276,0.0003502,0.0005939,0.0006783,-0.01389,0.0011,0.0008599,-0.001714,0.001137,-0.001118,0.0005429,-0.0004478,-0.0001939,0.0002102,1.592},
			//{-0.001826,6.244e-05,-0.00297,-0.0004808,-0.00244,-0.0005562,-0.002856,0.0005642,-0.002656,-0.000175,-0.002808,0.001761,-0.004559,0.003786,-0.01308,0.008182,0.7802},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{

			//{0.0004583,-3.776e-05,0.000316,-0.0008113,-0.0009096,0.0007942,0.001841,-0.001739,0.02259,-0.02592,-0.002932,0.02413,-3.142e-05,-0.0005255,0.0009952,0.0007662,1.685},     //俯仰角--航向视差决定（dr）
			//{-0.0008329,-0.0002501,8.352e-05,0.001747,0.0006944,-0.0006781,-0.001823,0.001966,-0.02208,0.02639,0.002659,-0.024,-0.0002287,0.0006152,-0.0009601,-0.0006891,1.642},
			{0.0008329,0.0002501,-8.352e-05,-0.001747,-0.0006944,0.0006781,0.001823,-0.001966,0.02208,-0.02639,-0.002659,0.024,0.0002287,-0.0006152,0.0009601,0.0006891,1.642},
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},

			//{0.0001,0.0001,0.0001,0.0001, 0.0001, 0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001, 0.0001,0.0001, 0.0001,0.0001, 2},
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			//{0.000367,0.001179,-0.001732,-0.0006322,0.001823,-0.001656,-0.004749,0.008912,0.003174,-0.004537,0.0159,-0.04108,-0.002175,-0.001707,0.00103,0.0001295,1.65},
			{-0.000367,-0.001179,0.001732,0.0006322,-0.001823,0.001656,0.004749,-0.008912,-0.003174,0.004537,-0.0159,0.04108,0.002175,0.001707,-0.00103,-0.0001295,1.65},
			//{-0.001604,-0.0007734,0.001022,-0.001215,-0.001495,0.001352,0.004074,-0.009567,-0.002866,0.004073,-0.0161,0.04055,0.002031,0.001455,-0.0009752,-0.0004053,1.693},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/


			/*double awptemp[2][6][17] = {{
			//{0.0003314,-0.0002566,-0.0003141,0.001853,-0.0002812,0.008543,-0.0005533,-0.0008615,-0.0009071,-0.0007003,-0.001914,0.002682,-0.0004561,0.003927,0.0001145,-0.0006252,2.698},   //俯仰角--航向视差决定（dr）
			{-0.002417,0.001044,-0.0006324,-0.001305,0.0002487,0.0005642,-0.003274,0.0009243,0.00195,0.00209,-0.02325,0.02711,-0.008494,0.02052,-0.001036,0.0009525,1.382},
			{2.634e-06,4.54e-06,-5.589e-06,1.075e-06,-1.291e-06,-6.967e-06,-7.66e-06,-6.47e-05,3.441e-05,1.638e-05,4.813e-06,7.589e-06,-5.046e-06,2.327e-06,2.542e-07,-4.347e-06,2.017},
			//{-0.001826,6.244e-05,-0.00297,-0.0004808,-0.00244,-0.0005562,-0.002856,0.0005642,-0.002656,-0.000175,-0.002808,0.001761,-0.004559,0.003786,-0.01308,0.008182,0.7802},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{
			{-0.002532,0.001178,-0.000726,-0.001145,0.0001689,0.0005594,-0.003213,0.000825,0.001861,0.002077,-0.02353,0.02696,-0.008829,0.02059,-0.001108,0.0009715,1.383},     //俯仰角--航向视差决定（dr）
			{2.787e-06,3.768e-06,-4.59e-06,1.745e-06,-2.546e-06,-7.34e-06,-6.977e-06,-6.45e-05,3.447e-05,1.651e-05,4.662e-06,7.793e-06,-5.03e-06,1.801e-06,5.456e-07,-4.018e-06,2.017},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/
			//EO拟合
			/*double awptemp[2][6][17] = {{
			{-0.0006284,3.339e-05,0.001741,-0.0005884,0.0001662,0.000309,3.077e-05,0.0008255,8.362e-05,0.0002863,1.485e-05,0.0002413,4.251e-05,0.001217,0.0003212,-0.0003228,0.8742},
			{0.001521,-2.807e-05,-0.0004808,-0.001511,0.002421,-0.001298,0.003233,-0.0007522,0.002054,0.001667,-2.174e-05,-0.001991,0.0006554,-0.002219,0.001736,-0.003536,0.7793},
			{-0.0002867,0.00051,-0.0007551,-4.933e-06,0.001046,0.0007361,-0.0002041,-0.000343,-0.0008138,-0.003286,0.0005775,0.0001688,9.4e-05,-0.0008936,-0.0001769,-0.0001349,1.628},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}},{
			{-0.001695,-0.001731,0.0007786,-0.0002191,-5.268e-05,0.0009388,0.0008114,-0.001129,0.002887,-0.006313,-0.001047,0.00396,-0.0003084,-0.0003832,7.284e-05,0.0003285,1.644 },     //俯仰角--航向视差决定（dr）
			{-0.001449,-0.001089,0.002254,-0.00118,-0.001624,-0.00114,-0.0003375,-0.001063,0.0002666,-0.0001764,-0.0009723,-0.0003633,-0.0001724,0.0004864,0.001915,-0.005161,1.236},  //翻滚角--旁向视差决定（dc）
			{0.005092,0.001468,-0.0002127,-0.001481,-0.001411,0.0003268,-0.0006226,0.001364,0.001463,0.0001797,0.001249,-0.001145,-0.003907,0.008895,-0.0003338,0.0005577,1.173},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0}}};//*/


			/*double awptemp[2][6][11] = {{
			{ 0.0001669,-0.0001961,-0.0004535,0.001631,-0.001502,0.008189,-0.0003467,-0.0009976,-0.0009441,-0.0009851,2.72},   //俯仰角--航向视差决定（dr）
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{ -0.0006215,-0.0008663,-0.0003504,0.0001461,3.314e-06,0.0007135,-0.01414,0.0008231,0.0008865,-0.001759,1.592},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}},

			{
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.001206,0.0004079,0.001131,-0.0002149,0.002653,-0.003257,0.00731,-0.03153,-0.01755,0.007015,2.021},   //俯仰角--航向视差决定（dr）
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.001272,-0.001106,0.004546,-0.009526,-0.01587,0.04045,-0.001317,-0.0002342,-0.0006213,-0.0004788,3.298},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}}};//*/
			/*double awptemp[2][6][11] = {{
			{0.0005563e-04,-0.002222e-04,0.0001566e-04,0.0008861e-04,-0.002301e-04,0.008228e-04,-0.0003971e-04,-0.001284e-04,-0.0008487e-04,-0.001322e-04,2.726},   //俯仰角--航向视差决定（dr）
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{-0.0001989e-04,0.000824e-04,0.0002909e-04,0.0004321e-04,0.000642e-04,0.0003807e-04,-0.01415e-04,0.0009293e-04,0.0004385e-04,-0.001592e-04,1.591},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}},

			{
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0004071e-04,-0.0002489e-04,0.0008866e-04,-0.001115e-04,0.00289e-04,-0.003135e-04,0.00694e-04,-0.03169e-04,-0.01772e-04,0.005625e-04,2.074},   //俯仰角--航向视差决定（dr）
			//{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.002203e-04,0.0008621e-04,0.005024e-04,-0.008662e-04,-0.01613e-04,0.04059e-04,-0.001475e-04,0.0001512e-04,-0.00001196e-04,-0.0001854e-04,3.384},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}}};//*/
			/*double awptemp[2][6][11] = {{{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},   //俯仰角--航向视差决定（dr）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}},
			{{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},   //俯仰角--航向视差决定（dr）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0},
			{0.0,0.0,0.0,0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0}}};//*/
			/*double awptemp[6][6] = {{0.000001,8,-1,     0.00000001, 100, 0},   //俯仰角--航向视差决定（dr）
			{0.0000005,6,-1,     0.00000001, 100, 0},  //翻滚角--旁向视差决定（dc）
			{0.0000001,50,0,         0.0, 0.0, 0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,           0.0, 0.0, 0.0},
			{0.0,0.0,0.0,           0.0, 0.0, 0.0},
			{0.0,0.0,0.0,           0.0, 0.0, 0.0}};//*/
			/*double awptemp[6][6] = {{0.000001,9.5,1.57,     0.0000001, 100.0, 0.0},  //俯仰角--航向视差决定（dr）
			{0.000001,7.8,1.57,     0.0000001, 100.0, 0.0},  //翻滚角--旁向视差决定（dc）
			{0.0000000,0.0,0.0,     0.0, 0.0, 0.0},  //对dr、dc影响相对不明确
			{0.0,0.0,0.0,           0.0, 0.0, 0.0},
			{0.0,0.0,0.0,           0.0, 0.0, 0.0},
			{0.0,0.0,0.0,           0.0, 0.0, 0.0}};//*/
			/*double awptemp[6][6] = {{0.0,0.0,0.0, 0.0, 0.0, 0.0},  //俯仰角--航向视差决定（dr）
			{0.0,0.0,0.0, 0.0, 0.0, 0.0},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0, 0.0, 0.0, 0.0},  //对dr、dc影响相对不明确
			{0.0,0.0,0.0,           0.0, 0.0, 0.0},
			{0.0,0.0,0.0,           0.0, 0.0, 0.0},
			{0.0,0.0,0.0,           0.0, 0.0, 0.0}};//*/


			/*double polytemp[2][6][4] = {{
			{-9.92141173118823e-09,0.000709651920961839e-04,0.00183007492828545e-04,-0.00804855773723921e-04},   //俯仰角--航向视差决定（dr）
			{-9.09719694891126e-09,0.000834810178944602e-04,-0.00254548591014061e-04,0.00438697190237473e-04},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0},
			{0.0,0.0,0.0,0.0},
			{0.0,0.0,0.0,0.0}},{
			{-1.37760088694656e-09,0.000150412988031824e-04,0.00117733852487890e-04,0.00220330031020204e-04},     //俯仰角--航向视差决定（dr）
			{-0.000320338940121263e-04,0.00433226786266934e-04,-0.0173198374939716e-04,0.0250356702980821e-04},  //翻滚角--旁向视差决定（dc）
			{0.0,0.0,0.0,0.0},   //对dr、dc影响相对不明确
			{0.0,0.0,0.0,0.0},
			{0.0,0.0,0.0,0.0},
			{0.0,0.0,0.0,0.0}}};//*/
			for(int k=0;k<4;k++){
				fscanf(fp_eo," %lf",&temp);
				poly_EO[i][jj*poly_num/6+k] = temp;//+polytemp[i][jj][4-k]
				//poly_CE[i][jj*poly_num/6+k] = temp;//+polytemp[i][jj][4-k]
				//printf("%lf ",poly_EO[i][jj*poly_num/6+k]);
			}
			for(int k=4;k<poly_num/6;k++){
				//poly_EO[i][jj*poly_num/6+k] = jj<3 ? awptemp1[k-4] : awptemp2[k-4];
				poly_EO[i][jj*poly_num/6+k] = awptemp[i][jj][k-4];
				//poly_CE[i][jj*poly_num/6+k] = poly_EO[i][jj*poly_num/6+k];
			}
			fscanf(fp_eo,"\n");
			//printf("\n");
		}
		fclose(fp_eo);
		fclose(fp_eof);


		//读取观测EO
		sprintf( EOtxt, "%s%s%s%s%s%s", EOfile, "/", xulie_ID, "/", xulie_ID, "_RED5_0.txt");
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
	cout<<poly_EO[0][poly_num/6*2-1]<<" "<<poly_EO[1][poly_num/6*2-1]<<endl;

	int ground_num = match_point[5*(int(match_point.size()/5)-1)]+1;
	double ** ground_point = new double *[ground_num];
	double ** ground_point0 = new double *[ground_num];
	for(int i=0;i<ground_num;i++){
		ground_point[i] = new double[3];
		ground_point0[i] = new double[3];
		for(int j=0;j<3;j++){
			ground_point[i][j]=GT_initial[i*3+j];
			ground_point0[i][j]=GT_initial[i*3+j];
		}
	}

	double focal=-11995.48;

	/////////////////////////////////////////////////////光束法平差////////////////////////////////////////////////////////////
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
			MakeBaCauchyLoss(),
			//NULL,
			ground_point[int(match_point[5*i])],
			poly_EO[int(match_point[5*i+4])]);
	}

	// 轨道参数方程
	for(int ii=0;ii<2;ii++){
		for(int i=0;i<CCnum[ii];i+=1){
			double * eo_temp=new double[6];
			for(int j=0;j<6;j++){
				eo_temp[j]=obs_EO[ii][i*6+j];
			}
			ceres::CostFunction* cost_function =
				Orbitalcost2::Create(
				obs_et[ii][i]-et0[ii],
				eo_temp);
			problem1.AddResidualBlock(cost_function,
				MakeBaCauchyLoss(),
				//NULL,
				poly_EO[ii]);
		}
	}

	// ground point control
	double H0 = -2200;
	for(int i=0;i<ground_num;i++){
		//ceres::CostFunction* cost_function =
		//	PointControlError_H::Create(H0);
		//problem1.AddResidualBlock(cost_function,
		//	MakeBaCauchyLoss(),
		//	//NULL,
		//	ground_point[int(match_point[5*i])]);
		ceres::CostFunction* cost_function =
			PointControlError::Create(ground_point0[int(match_point[5*i])]);
		problem1.AddResidualBlock(cost_function,
			MakeBaCauchyLoss(),
			//NULL,
			ground_point[int(match_point[5*i])]);
	}

	ceres::Solver::Options options1;
	ConfigureCeresThreads(options1);
	options1.linear_solver_type = ceres::DENSE_SCHUR;
	options1.minimizer_progress_to_stdout = true;
	options1.max_num_iterations=ba_cfg().jitter_max_iterations;
	options1.function_tolerance=ba_cfg().function_tolerance;
	ceres::Solver::Summary summary1;
	ceres::Solve(options1, &problem1, &summary1);

	std::cout << summary1.FullReport() << "\n";/**/


	//计算残差
	double *resdual = new double[ground_num*2];
	int count=0;
	double temp;
	double res[2];
	res[0]=0;res[1]=0;
	int js=0;
	//for(int i=0;i<match_point.size()/5;i++){
	// if(match_point[5*i]==count){
	// Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
	// res+=temp*temp;
	// //res+=temp;
	// js+=1;
	// if(i==match_point.size()/5-1){
	// resdual[count]=sqrt(res/js);
	// //resdual[count]=res/js;
	// }
	// }
	// else{
	// resdual[count]=sqrt(res/js);
	// //resdual[count]=res/js;
	// res=0;
	// js=0;
	// count++;
	// Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
	// res+=temp*temp;
	// //res+=temp;
	//
	// }
	//}
	//for(int i=0;i<match_point.size()/5;i++){
	// if(match_point[5*i]==count){
	// Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,0);
	// res[0]=temp;
	// Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,1);
	// res[1]=temp;
	// js+=1;
	// if(i==match_point.size()/5-1){
	// resdual[2*count]=res[0];
	// resdual[2*count+1]=res[1];
	// }
	// }
	// else{
	// resdual[2*count]=res[0];
	// resdual[2*count+1]=res[1];
	// res[0]=0;res[1]=0;
	// js=0;
	// count++;
	// Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,0);
	// res[0]=temp;
	// Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,1);
	// res[1]=temp;
	// }
	//}
	FILE *fp_fi=fopen(GTtxt,"w");
	fprintf(fp_fi,"%d\n",ground_num);
	for(int i=0;i<match_point.size()/5;i++){
		if(match_point[5*i]==count){
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,0);
			res[0]=temp;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,1);
			res[1]=temp;
			fprintf(fp_fi,"%lf %lf %lf %lf %lf\n",ground_point[count][0],ground_point[count][1],ground_point[count][2],res[0],res[1]);
			resdual[2*count]=res[0];
			resdual[2*count+1]=res[1];
			res[0]=0;res[1]=0;
			js=0;
			count++;
			if(i==match_point.size()/5-1){
				resdual[2*count]=res[0];
				resdual[2*count+1]=res[1];
			}
		}
		else{
			js+=1;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,0);
			res[0]=temp;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,1);
			res[1]=temp;
			//fprintf(fp_fi,"%lf %lf %lf %lf %lf\n",ground_point[count-1][0],ground_point[count-1][1],ground_point[count-1][2],res[0],res[1]);
		}
	}

	////输出地面点坐标
	//FILE *fp_fi=fopen(GTtxt,"w");
	//fprintf(fp_fi,"%d\n",ground_num);
	//for(int i=0;i<ground_num;i++){
	// fprintf(fp_fi,"%lf %lf %lf %lf %lf\n",ground_point[i][0],ground_point[i][1],ground_point[i][2],resdual[2*i],resdual[2*i+1]);
	//}
	fclose(fp_fi);

	//输出外方位元素
	for(int i=0;i<image_num;i++){
		if(i==0){
			xulie_ID=xulie_ID1;
		}
		else if(i==1){
			xulie_ID=xulie_ID2;
		}
		char* EOretxt = new char[80];
		sprintf(EOretxt, "%s%s%s%s", EOreFile, "/", xulie_ID, "_EOre.txt" );
		FILE *fp_eo=fopen(EOretxt,"w");
		for(int j=0;j<6;j++){
			for(int k=0;k<poly_num/6+1;k++){
				if(k==0){
					fprintf(fp_eo,"%.12lf",et0[i]);
				}
				else{
					fprintf(fp_eo," %.12lf",poly_EO[i][j*poly_num/6+k-1]);
				}
			}
			fprintf(fp_eo,"\n");
			//delete[] (poly_EO[i]);
			//delete[] (poly_CE[i]);
			//fprintf(fp_eo,"%.12lf %.12lf %.12lf %.12lf %.12lf\n",et0[i],poly_EO[i][j*4+0],poly_EO[i][j*4+1],poly_EO[i][j*4+2],poly_EO[i][j*4+3]);
		}
		fclose(fp_eo);
	}

	std::vector<double>().swap(match_point);
	std::vector<double>().swap(GT_initial);

	delete[] resdual;
	delete[] ground_point;

	return 0;
}


//前方交会（格网点）
int Forward_intersection(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* FItxt,char* outfilepath=nullptr){
	int CCD_num=10;
	printf("Begin Forward_intersection!\n");

	//数据准备
	std::vector<double> match_point;
	std::vector<double> GT_initial;
	FILE *fp=fopen(observetxt,"r");
	double temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;
	int markC=-1;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf\n",&temp1,&temp2,&temp3,&temp4,&temp5,&temp6,&temp7,&temp8);
		match_point.push_back(temp1);
		match_point.push_back(temp2);
		match_point.push_back(temp3);
		match_point.push_back(temp4);
		match_point.push_back(temp5);
		if(int(temp1)!=markC){
			GT_initial.push_back(temp6);
			GT_initial.push_back(temp7);
			GT_initial.push_back(temp8);
			markC=temp1;
		}
	}
	fclose(fp);


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
		sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID, "_EOre.txt" );
		//sprintf( EOtxt, "%s%s%s%s", "../data/EO", "/", xulie_ID, "/polyCC.txt" );
		FILE* fp_eo=fopen(EOtxt,"r");
		poly_EO[i] = new double[poly_num];
		poly_CE[i] = new double[poly_num];
		poly_CR[i] = new double[poly_num/2];
		poly_CT[i] = new double[poly_num/2];
		double temp,temp1;
		int jj;
		for(int j=0;j<6;j++){
			if(j<3){
				jj=j+3;
			}
			else{
				jj=j-3;
			}
			jj=j;
			fscanf(fp_eo,"%lf",&temp);
			et0[i]=temp;

			for(int k=0;k<poly_num/6;k++){
				fscanf(fp_eo," %lf",&temp);
				poly_EO[i][jj*poly_num/6+k] = temp;
				poly_CE[i][jj*poly_num/6+k] = temp;
				/*if(jj<3){
				poly_CR[i][jj*4+k] = temp;
				}
				else{
				poly_CT[i][(jj-3)*4+k] = temp;
				}*/
			}
			fscanf(fp_eo,"\n");
		}
		fclose(fp_eo);
	}

	int ground_num = match_point[5*(int(match_point.size()/5)-1)]+1;
	double ** ground_point = new double *[ground_num];
	for(int i=0;i<ground_num;i++){
		ground_point[i] = new double[3];
		for(int j=0;j<3;j++){
			ground_point[i][j]=GT_initial[i*3+j];
		}
	}


	double focal=-11995.48;

	//////////////////////////////////////////////////前方交会////////////////////////////////////////////////////////////
	Problem problem; 
	int index=0;
	int countFI=0;

	int FI_size=match_point.size()/5;
	//int FI_size=2;

	//CCD间点选相邻航带
	for (int i=0; i<FI_size; i++) {
		if(countFI<2 && int(match_point[5*i])==index){
			/*if(countFI==0){
			ground_point[index][0]=GT_initial[i*3+0];
			ground_point[index][1]=GT_initial[i*3+1];
			ground_point[index][2]=GT_initial[i*3+2];
			}*/
			ceres::CostFunction* cost_function =
				ForInsec_cost::Create(
				match_point[5*i+1],
				match_point[5*i+2],
				match_point[5*i+3]-et0[int(match_point[5*i+4])],
				poly_CE[int(match_point[5*i+4])],
				focal);
			problem.AddResidualBlock(cost_function,
				MakeFiCauchyLoss(),
				//NULL,
				ground_point[int(match_point[5*i])]);
			countFI++;
		}
		else if(countFI==2 && int(match_point[5*i])==index+1){
			countFI=0;
			/*if(countFI==0){
			ground_point[index+1][0]=GT_initial[i*3+0];
			ground_point[index+1][1]=GT_initial[i*3+1];
			ground_point[index+1][2]=GT_initial[i*3+2];
			}*/
			ceres::CostFunction* cost_function =
				ForInsec_cost::Create(
				match_point[5*i+1],
				match_point[5*i+2],
				match_point[5*i+3]-et0[int(match_point[5*i+4])],
				poly_CE[int(match_point[5*i+4])],
				focal);
			problem.AddResidualBlock(cost_function,
				MakeFiCauchyLoss(),
				//NULL,
				ground_point[int(match_point[5*i])]);
			countFI++;
			index++;
		}
		else if(countFI==2 && int(match_point[5*i])==index){
			index++;
			countFI=0;
		}
	}

	//求解
	ceres::Solver::Options options;
	ConfigureCeresThreads(options);
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations=ba_cfg().fi_max_iterations;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << "\n";

	//输出地面点坐标
	FILE *fp_fi=fopen(FItxt,"w");
	fprintf(fp_fi,"%d\n",ground_num);
	//计算残差
	int count=0;
	double temp;
	double res[2];
	int js=0;
	double *resdual = new double[ground_num*2];
	for(int i=0;i<match_point.size()/5;i++){
		if(match_point[5*i]==count){
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,0);
			res[0]=temp;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,1);
			res[1]=temp;
			//fprintf(fp_fi,"%lf %lf %lf %lf %lf\n",ground_point[count][0],ground_point[count][1],ground_point[count][2],res[0],res[1]);
			resdual[2*count]=res[0];
			resdual[2*count+1]=res[1];
			res[0]=0;res[1]=0;
			js=0;
			count++;
			if(i==match_point.size()/5-1){
				resdual[2*count]=res[0];
				resdual[2*count+1]=res[1];
			}
		}
		else{
			js++;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,0);
			res[0]=temp;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp,1);
			res[1]=temp;
			fprintf(fp_fi,"%lf %lf %lf %lf %lf\n",ground_point[count-1][0],ground_point[count-1][1],ground_point[count-1][2],res[0]/0.012,res[1]/0.012);
		}
	}
	fclose(fp_fi);


	// //输出Z到地面控制文件
	// if(outfilepath != nullptr){
	// 	for(int j=0;j<CCD_num;j++){
	// 		char* matchtxt_ = new char[256];
	// 		sprintf( matchtxt_, "%s/%s/downsample/0/%s_RED%d_match_.txt", outfilepath, xulie_ID1, xulie_ID1, j );
	// 		char* matchtxt__ = new char[256];
	// 		sprintf( matchtxt__, "%s/%s/downsample/0/%s_RED%d_match__.txt", outfilepath, xulie_ID1, xulie_ID1, j );
	// 		FILE *fp_=fopen(matchtxt_,"r");
	// 		FILE *fp__=fopen(matchtxt__,"w");

	// 		if(fp_ == NULL || fp__ == NULL){
	// 			if(fp_) fclose(fp_);
	// 			if(fp__) fclose(fp__);
	// 			delete[] matchtxt_; delete[] matchtxt__;
	// 			continue;
	// 		}

	// 		int bj=0,imgID=0;float row=0,col=0;
	// 		double Z=0;

	// 		while(!feof(fp_)){
	// 			int ret = fscanf(fp_,"%d %f %f %lf %d\n",&bj, &row, &col, &Z, &imgID);
	// 			if(ret != 5) break;
	// 			if(bj < 0 || bj >= ground_num) continue;
	// 			fprintf(fp__,"%d %f %f %lf %d\n",bj, row, col, ground_point[bj][2], imgID);
	// 		}
	// 		fclose(fp_);
	// 		fclose(fp__);
	// 		remove(matchtxt_);
	// 		rename(matchtxt__,matchtxt_);
	// 		delete[] matchtxt_; delete[] matchtxt__;
	// 	}
	// }

	return 0;
}

int Forward_intersection(char* observetxt,char* EOfile,double** poly_IO,char* xulie_ID1,char* xulie_ID2,char* FItxt){
	int CCD_num=10;
	printf("Begin Forward_intersection!\n");

	//数据准备
	std::vector<double> match_point;
	std::vector<double> GT_initial;
	FILE *fp=fopen(observetxt,"r");
	double temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;
	int markC=-1;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf\n",&temp1,&temp2,&temp3,&temp4,&temp5,&temp6,&temp7,&temp8);
		match_point.push_back(temp1);
		match_point.push_back(temp2);
		match_point.push_back(temp3);
		match_point.push_back(temp4);
		match_point.push_back(temp5);
		if(int(temp1)!=markC){
			GT_initial.push_back(temp6);
			GT_initial.push_back(temp7);
			GT_initial.push_back(temp8);
			markC=temp1;
		}
	}
	fclose(fp);


	int image_num=2;
	int CCnum[2];
	CCnum[0]=80;CCnum[1]=160;
	//外方位元素观测
	double LR[2];

	char* EOtxt = new char[80];
	double ** poly_EO = new double *[image_num];
	double ** poly_CE = new double *[image_num];
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
		sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID, "_EOre.txt" );
		FILE* fp_eo=fopen(EOtxt,"r");
		poly_EO[i] = new double[24];
		poly_CE[i] = new double[24];
		double temp,temp1;
		int jj;
		for(int j=0;j<6;j++){
			jj=j;
			fscanf(fp_eo,"%lf",&temp);
			et0[i]=temp;

			for(int k=0;k<24/6;k++){
				fscanf(fp_eo," %lf",&temp);
				poly_EO[i][jj*24/6+k] = temp;
				poly_CE[i][jj*24/6+k] = temp;
				/*if(jj<3){
				poly_CR[i][jj*4+k] = temp;
				}
				else{
				poly_CT[i][(jj-3)*4+k] = temp;
				}*/
			}
			fscanf(fp_eo,"\n");
		}
		fclose(fp_eo);
	}

	int ground_num = match_point[5*(int(match_point.size()/5)-1)]+1;
	double ** ground_point = new double *[ground_num];
	for(int i=0;i<ground_num;i++){
		ground_point[i] = new double[3];
		for(int j=0;j<3;j++){
			ground_point[i][j]=GT_initial[i*3+j];
		}
	}


	double focal=-11995.48;

	//////////////////////////////////////////////////前方交会////////////////////////////////////////////////////////////
	Problem problem; 
	int index=0;
	int countFI=0;

	//CCD间点选相邻航带
	for (int i=0; i<match_point.size()/5; i++) {
		if(countFI<2 && int(match_point[5*i])==index){
			/*if(countFI==0){
			ground_point[index][0]=GT_initial[i*3+0];
			ground_point[index][1]=GT_initial[i*3+1];
			ground_point[index][2]=GT_initial[i*3+2];
			}*/
			ceres::CostFunction* cost_function =
				IO_ForInseccost::Create(
				match_point[5*i+1],
				match_point[5*i+2],
				match_point[5*i+3]-et0[int(match_point[5*i+4])],
				focal,
				poly_EO[int(match_point[5*i+4])],
				poly_IO[int(match_point[5*i+4])]);
			problem.AddResidualBlock(cost_function,
				new ceres::CauchyLoss(0.5),
				//NULL,
				ground_point[int(match_point[5*i])]);
			countFI++;
		}
		else if(countFI==2 && int(match_point[5*i])==index+1){
			countFI=0;
			/*if(countFI==0){
			ground_point[index+1][0]=GT_initial[i*3+0];
			ground_point[index+1][1]=GT_initial[i*3+1];
			ground_point[index+1][2]=GT_initial[i*3+2];
			}*/
			ceres::CostFunction* cost_function =
				IO_ForInseccost::Create(
				match_point[5*i+1],
				match_point[5*i+2],
				match_point[5*i+3]-et0[int(match_point[5*i+4])],
				focal,
				poly_EO[int(match_point[5*i+4])],
				poly_IO[int(match_point[5*i+4])]);
			problem.AddResidualBlock(cost_function,
				new ceres::CauchyLoss(0.5),
				//NULL,
				ground_point[int(match_point[5*i])]);
			countFI++;
			index++;
		}
		else if(countFI==2 && int(match_point[5*i])==index){
			index++;
			countFI=0;
		}
	}

	//求解
	ceres::Solver::Options options;
	ConfigureCeresThreads(options);
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations=50;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << "\n";

	//计算残差
	double *resdual = new double[ground_num];
	int count=0;
	double res,temp;
	res=0;
	int js=0;
	for(int i=0;i<match_point.size()/5;i++){
		if(match_point[5*i]==count){
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],poly_IO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
			res=temp;
			js+=1;
			if(i==match_point.size()/5-1){
				resdual[count]=res;
			}
		}
		else{
			resdual[count]=res;
			res=0;
			js=0;
			count++;
			Compu_ProjRES(match_point[5*i+1],match_point[5*i+2], match_point[5*i+3]-et0[int(match_point[5*i+4])], poly_EO[int(match_point[5*i+4])],poly_IO[int(match_point[5*i+4])],focal,ground_point[int(match_point[5*i])],&temp);
			res=temp;
		}
	}

	//输出地面点坐标
	FILE *fp_fi=fopen(FItxt,"w");
	fprintf(fp_fi,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		fprintf(fp_fi,"%lf %lf %lf %lf\n",ground_point[i][0],ground_point[i][1],ground_point[i][2],resdual[i]);
	}
	fclose(fp_fi);

	//输出Z到地面控制文件 (skipped: outfilepath not available in poly_IO overload)

	return 0;
}

int Forward_intersection_initial(char* observetxt,char* EOfile,char* xulie_ID1,char* xulie_ID2,char* FItxt){
	int CCD_num=10;
	printf("Begin Forward_intersection!\n");

	//数据准备
	std::vector<double> match_point;
	std::vector<double> GT_initial;
	FILE *fp=fopen(observetxt,"r");
	if(fp == NULL){
		printf("[ERROR][Forward_intersection_initial] 无法打开观测文件: %s\n", observetxt);
		return -1;
	}
	double temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;
	int markC=-1;
	while(!feof(fp)){
		fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf\n",&temp1,&temp2,&temp3,&temp4,&temp5,&temp6,&temp7,&temp8);
		match_point.push_back(temp1);
		match_point.push_back(temp2);
		match_point.push_back(temp3);
		match_point.push_back(temp4);
		match_point.push_back(temp5);
		if(int(temp1)!=markC){
			GT_initial.push_back(temp6);
			GT_initial.push_back(temp7);
			GT_initial.push_back(temp8);
			markC=temp1;
		}
	}
	fclose(fp);


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
		//sprintf( EOtxt, "%s%s%s%s", EOfile, "/", xulie_ID, "_EOre.txt" );
		sprintf( EOtxt, "%s%s%s%s", "../data/EO", "/", xulie_ID, "/polyCC.txt" );
		FILE* fp_eo=fopen(EOtxt,"r");
		poly_EO[i] = new double[init_poly_num];
		poly_CE[i] = new double[init_poly_num];
		double temp,temp1;
		int jj;
		for(int j=0;j<6;j++){
			if(j<3){
				jj=j+3;
			}
			else{
				jj=j-3;
			}
			//jj=j;
			fscanf(fp_eo,"%lf",&temp);
			et0[i]=temp;

			for(int k=0;k<init_poly_num/6;k++){
				fscanf(fp_eo," %lf",&temp);
				poly_EO[i][jj*init_poly_num/6+k] = temp;
				poly_CE[i][jj*init_poly_num/6+k] = temp;
				/*if(jj<3){
				poly_CR[i][jj*4+k] = temp;
				}
				else{
				poly_CT[i][(jj-3)*4+k] = temp;
				}*/
			}
			fscanf(fp_eo,"\n");
		}
		fclose(fp_eo);
	}

	int ground_num = match_point[5*(int(match_point.size()/5)-1)]+1;
	double ** ground_point = new double *[ground_num];
	for(int i=0;i<ground_num;i++){
		ground_point[i] = new double[3];
		for(int j=0;j<3;j++){
			ground_point[i][j]=GT_initial[i*3+j];
		}
	}


	double focal=-11995.48;

	//////////////////////////////////////////////////前方交会////////////////////////////////////////////////////////////
	Problem problem; 
	int index=0;
	int countFI=0;

	int FI_size=match_point.size()/5;
	//int FI_size=2;

	//CCD间点选相邻航带
	for (int i=0; i<FI_size; i++) {
		if(countFI<2 && int(match_point[5*i])==index){
			/*if(countFI==0){
			ground_point[index][0]=GT_initial[i*3+0];
			ground_point[index][1]=GT_initial[i*3+1];
			ground_point[index][2]=GT_initial[i*3+2];
			}*/
			ceres::CostFunction* cost_function =
				ForInsec_cost_initial::Create(
				match_point[5*i+1],
				match_point[5*i+2],
				match_point[5*i+3]-et0[int(match_point[5*i+4])],
				poly_CE[int(match_point[5*i+4])],
				focal);
			problem.AddResidualBlock(cost_function,
				MakeFiCauchyLoss(),
				//NULL,
				ground_point[int(match_point[5*i])]);
			countFI++;
		}
		else if(countFI==2 && int(match_point[5*i])==index+1){
			countFI=0;
			/*if(countFI==0){
			ground_point[index+1][0]=GT_initial[i*3+0];
			ground_point[index+1][1]=GT_initial[i*3+1];
			ground_point[index+1][2]=GT_initial[i*3+2];
			}*/
			ceres::CostFunction* cost_function =
				ForInsec_cost_initial::Create(
				match_point[5*i+1],
				match_point[5*i+2],
				match_point[5*i+3]-et0[int(match_point[5*i+4])],
				poly_CE[int(match_point[5*i+4])],
				focal);
			problem.AddResidualBlock(cost_function,
				MakeFiCauchyLoss(),
				//NULL,
				ground_point[int(match_point[5*i])]);
			countFI++;
			index++;
		}
		else if(countFI==2 && int(match_point[5*i])==index){
			index++;
			countFI=0;
		}
	}

	//求解
	ceres::Solver::Options options;
	ConfigureCeresThreads(options);
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations=ba_cfg().fi_max_iterations;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << "\n";

	// 输出前方教会的地面点initail value
	FILE *fp_fi=fopen(FItxt,"w");
	for (int i=0; i<match_point.size()/5; i++) {
		fprintf(fp_fi,"%d %lf %lf %lf %d %lf %lf %lf\n",int(match_point[5*i+0]),match_point[5*i+1],match_point[5*i+2],match_point[5*i+3],int(match_point[5*i+4]),
			ground_point[int(match_point[5*i+0])][0],ground_point[int(match_point[5*i+0])][1],ground_point[int(match_point[5*i+0])][2]);
	}
	fclose(fp_fi);

	//输出地面点坐标
	fp_fi=fopen("out.txt","w");
	fprintf(fp_fi,"%d\n",ground_num);
	for(int i=0;i<ground_num;i++){
		fprintf(fp_fi,"%lf %lf %lf %lf\n",ground_point[i][0],ground_point[i][1],ground_point[i][2],0);
	}
	fclose(fp_fi);

	return 0;
}

int BA_main(const PipelineConfig& cfg){
	g_ba_params = &cfg.ba;
	char* outfilepath = const_cast<char*>(cfg.filepath);
	char* xulie_ID1 = const_cast<char*>(cfg.xulie_ID1);
	char* xulie_ID2 = const_cast<char*>(cfg.xulie_ID2);
	int rows1 = cfg.rows1;
	int rows2 = cfg.rows2;
	//像点观测文件
	char* observetxt = new char[80];
	sprintf( observetxt, "%s%s%s%s%s", "../data/observedata/", xulie_ID1,"_", xulie_ID2, ".txt");

	//外方位元素观测路径
	char* EOfile = "../data/EO";

	//地面点坐标输出文件路径（特征点BA）
	char* GTtxt = new char[80];
	sprintf( GTtxt, "%s%s%s%s%s", "../data/result/", xulie_ID1,"_", xulie_ID2, ".txt");
	//外方位元素输出路径（特征点BA）
	char* EOreFile = "../data/result/";

	Observation OB;
	OB.cfg_ = cfg;

	//计算fs_c
	float* fs_c = new float[6];
	memset(fs_c,1.0,sizeof(float)*6);
	OB.Compute_fsc(fs_c,0);

	printf("[INFO][BA_main] fi_source=%s run_feature_ba=%d use_jitter=%d\n",
		cfg.ba.fi_use_grid() ? "grid" : "feature",
		(int)cfg.ba.run_feature_ba,
		(int)cfg.ba.use_jitter);

	// ---------- 特征点 BA（更新 EO）----------
	if(cfg.ba.run_feature_ba){
		OB.Tichu_CX(0,fs_c);
		if(!OB.prepare_for_BA(0)){
			printf("[ERROR][BA_main] prepare_for_BA(feature) 失败，中止 BA\n");
			g_ba_params = nullptr;
			return -1;
		}

		cout<<"recompute the initial value of fea_points!"<<endl;
		if(Forward_intersection_initial(observetxt,EOfile,xulie_ID1,xulie_ID2,observetxt) == -1){
			printf("[ERROR][BA_main] Forward_intersection_initial 失败，中止 BA\n");
			g_ba_params = nullptr;
			return -1;
		}

		bool eo_ok = false;
		if(cfg.ba.use_jitter){
			if(Jitter_adjustment(observetxt,EOfile,xulie_ID1,xulie_ID2,GTtxt,EOreFile) == -1){
				printf("Jitter_adjustment 跳过，改用 Block_adjustment 生成 _EOre.txt\n");
				eo_ok = (Block_adjustment(observetxt,EOfile,xulie_ID1,xulie_ID2,GTtxt,EOreFile) != -1);
			}else{
				eo_ok = true;
			}
		}else{
			eo_ok = (Block_adjustment(observetxt,EOfile,xulie_ID1,xulie_ID2,GTtxt,EOreFile) != -1);
		}
		if(!eo_ok){
			printf("[ERROR][BA_main] 特征点 BA 失败，未生成 _EOre.txt\n");
			g_ba_params = nullptr;
			return -1;
		}
	}else{
		printf("[INFO][BA_main] run_feature_ba=false，跳过特征点 BA，直接用已有 _EOre.txt 做 FI\n");
	}

	// ---------- FI（前方交会）----------
	const int fi_mark = cfg.ba.fi_use_grid() ? 1 : 0;
	if(fi_mark == 1){
		OB.Tichu_CX(1,fs_c);
		if(!OB.prepare_for_BA(1)){
			printf("[ERROR][BA_main] prepare_for_BA(grid) 失败，中止 FI\n");
			g_ba_params = nullptr;
			return -1;
		}
	}else if(!cfg.ba.run_feature_ba){
		// 特征 FI 且未跑特征 BA：需要单独准备观测文件
		OB.Tichu_CX(0,fs_c);
		if(!OB.prepare_for_BA(0)){
			printf("[ERROR][BA_main] prepare_for_BA(feature) 失败，中止 FI\n");
			g_ba_params = nullptr;
			return -1;
		}
	}else{
		printf("[INFO][BA_main] fi_source=feature：复用特征 BA 阶段的 observetxt\n");
	}

	//外方位元素路径（优先 result/_EOre.txt）
	EOfile = "../data/result/";
	char* FItxt = new char[80];
	sprintf( FItxt, "%s%s%s%s%s", "../data/result/", xulie_ID1,"_", xulie_ID2, "_FI.txt");
	sprintf( observetxt, "%s%s%s%s%s", "../data/observedata/", xulie_ID1,"_", xulie_ID2, ".txt");
	printf("[INFO][BA_main] Forward_intersection using %s matches → %s\n",
		fi_mark == 1 ? "grid" : "feature", FItxt);
	Forward_intersection(observetxt,EOfile,xulie_ID1,xulie_ID2,FItxt,outfilepath);

	g_ba_params = nullptr;
	return 0;
}
