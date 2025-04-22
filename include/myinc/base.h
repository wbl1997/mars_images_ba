#ifndef _BASE_H_
#define _BASE_H_

#include <string.h>
#include <stdio.h>

//×Ö·û´®·Ö¸î
void split(char *src,char separator,char** dest,int *num);
void mxm(double *m1, double *m2, double *re);
void m2v(double **m1, int rows, int cols, double *m2);

#include "../src/base.cpp"
#endif 