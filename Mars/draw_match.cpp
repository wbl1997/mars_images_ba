#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <time.h>

#include "BA.h"
#include "ImageProcess.h"

#include "EO.h"
#include "Observation.h"


using namespace std;

int main(int argc, const char* argv[])
{
	Observation OB;
    ImageMatch IM;

	const char* filepath="/media/wbl/Elements/paper_experiments/Mars/new";
	const char* xulie_ID1="ESP_069731_2055";
	const char* xulie_ID2="ESP_075559_2055";
    int CCD_num=9;
	int rows1=55000;
	int rows2=40000;
	int cols=2048;

    int CCD_id;

    //读取拼接系数
    std::cout<<"读取拼接系数……"<<std::endl;
    char ds_path_mosaictxt1[256];
    snprintf(ds_path_mosaictxt1, sizeof(ds_path_mosaictxt1), "%s/%s/downsample/%d/mosaic.txt", filepath, xulie_ID1, 0);
    FILE *fpm1=fopen(ds_path_mosaictxt1,"r");
    if(fpm1==NULL){
        std::cout<<"未找到拼接系数文件："<<ds_path_mosaictxt1<<std::endl;
        return 0;
    }
    std::vector<int> mosaic_c1(CCD_num*4);
    for(int ii=0;ii<CCD_num;ii++){
        if(fscanf(fpm1,"%d %d %d %d %d\n",&CCD_id,&mosaic_c1[ii*4+0],&mosaic_c1[ii*4+1],&mosaic_c1[ii*4+2],&mosaic_c1[ii*4+3]) != 5){
            std::cout<<"拼接系数文件格式错误："<<ds_path_mosaictxt1<<std::endl;
            fclose(fpm1);
            return 0;
        }
    }
    fclose(fpm1);
    char ds_path_mosaictxt2[256];
    snprintf(ds_path_mosaictxt2, sizeof(ds_path_mosaictxt2), "%s/%s/downsample/%d/mosaic.txt", filepath, xulie_ID2, 0);
    FILE *fpm2=fopen(ds_path_mosaictxt2,"r");
    if(fpm2==NULL){
        std::cout<<"未找到拼接系数文件："<<ds_path_mosaictxt2<<std::endl;
        return 0;
    }
    std::vector<int> mosaic_c2(CCD_num*4);
    for(int ii=0;ii<CCD_num;ii++){
        if(fscanf(fpm2,"%d %d %d %d %d\n",&CCD_id,&mosaic_c2[ii*4+0],&mosaic_c2[ii*4+1],&mosaic_c2[ii*4+2],&mosaic_c2[ii*4+3]) != 5){
            std::cout<<"拼接系数文件格式错误："<<ds_path_mosaictxt2<<std::endl;
            fclose(fpm2);
            return 0;
        }
    }
    fclose(fpm2);


    std::vector<int> KeyPoint_x1,KeyPoint_y1,KeyPoint_x2,KeyPoint_y2;
    std::vector<int> KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22;
    //航带间
    int count0=0;
    int count1=0;
    int mark=0; // 0: fea，1: grid
    for(int j=0;j<9;j++){
        char featurepoint1[256];
        if(mark==0){
            snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d.txt", filepath, xulie_ID1, 0, xulie_ID1, j);
        }
        else{
            snprintf(featurepoint1, sizeof(featurepoint1), "%s/%s/downsample/%d/%s_RED%d_grid.txt", filepath, xulie_ID1, 0, xulie_ID1, j);
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
        double sfr=pow(double(2),double(0))/pow(double(2),double(4));
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

    std::vector<int> match(KeyPoint_x1.size(), 1);

    std::cout<<"开始绘制匹配结果在mosaic img……"<<std::endl;
    char imgL_path1[256];
    snprintf(imgL_path1, sizeof(imgL_path1), "%s/%s/downsample/%d/mosaic_ds4.tif", filepath, xulie_ID1, 0);
    char imgR_path1[256];
    snprintf(imgR_path1, sizeof(imgR_path1), "%s/%s/downsample/%d/mosaic_ds4.tif", filepath, xulie_ID2, 0);
    IM.drawMatch3(imgL_path1,imgR_path1,const_cast<char*>("../out/match_.tif"),KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match.data(),1);
    IM.drawMatch4(imgL_path1,imgR_path1,const_cast<char*>("../out/match1_.tif"),KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,match.data(),count1,1);
    IM.drawMatch2(imgL_path1,imgR_path1,KeyPoint_x11,KeyPoint_y11,KeyPoint_x22,KeyPoint_y22,1);

    std::vector<int>().swap(KeyPoint_x1);
    std::vector<int>().swap(KeyPoint_y1);
    std::vector<int>().swap(KeyPoint_x2);
    std::vector<int>().swap(KeyPoint_y2);
    std::vector<int>().swap(KeyPoint_x11);
    std::vector<int>().swap(KeyPoint_y11);
    std::vector<int>().swap(KeyPoint_x22);
    std::vector<int>().swap(KeyPoint_y22);

    return 0;
}
