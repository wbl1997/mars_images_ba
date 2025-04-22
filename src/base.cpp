#include "base.h"

void split(char *src,char separator,char** dest,int *num) {
	/*
	src 源字符串的首地址(buf的地址) 
	separator 指定的分割字符
	dest 接收子字符串的数组
	num 分割后子字符串的个数
	*/

	char *pNext;
	int count = 0;
	if (src == NULL || strlen(src) == 0){ //如果传入的地址为空或长度为0，直接终止 
	    return;
	}

	char* p=src;
	int i = 0, j = 0;    
	char tmp[32][32] = {0};    
	char *p1 = (char *)malloc(1024); 
	while((p1 = strchr(p, separator)) != NULL)    
	{        
		strncpy(tmp[i], p, strlen(p) - strlen(p1));       
		p = p1 + 1;        
		i++;    
	}    
	strncpy(tmp[i], p, strlen(p)); 
	for(j = 0; j <= i; j++){ 
		dest[j]=tmp[j];
	    //printf("temp[%d] = %s\n", j, dest[j]);
	}
	*num = i+1;
	//printf("分割后字符串个数：%d\n",*num);
} 	

void mxm(double *m1, double *m2, double *re){
	int i__1, i__2, i__3, i__4, i__5, i__6, i__7;
	int i__, j;
	double prodm[9];
	for (i__ = 1; i__ <= 3; ++i__) { //行
		for (j = 1; j <= 3; ++j) { //列
			re[(i__1 = j + i__ * 3 - 4) ] = m1[(i__2 = i__ * 3 - 3)] * m2[(i__3 = j - 1)]
			                              + m1[(i__4 = i__ * 3 - 2)] * m2[(i__5 = j + 2)] 
			                              + m1[(i__6 = i__ * 3 - 1)] * m2[(i__7 = j + 5)];
		}
	}
}

void m2v(double **m1, int rows, int cols, double *m2){
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			m2[i*rows+j]=m1[i][j];
		}
	}
}