#include <iostream>
#include <cstdio>
#include <cstdlib> 
#include <ctime>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

using namespace std;

const int N=1000050;

int n,a[N];

int simple_calc(int a[N], int n){
	int sum = 0;
	for(int i = 0; i < n; ++i){
		sum += a[i];
	}
	return sum;
}

int advanced_calc(int a[N], int n){
	int sum1 = 0, sum2 = 0;
	for(int i = 0;i < n; i += 2){
		sum1 += a[i];
		sum2 += a[i + 1];
	}
	return sum1 + sum2;
}

int unroll_calc(int a[N], int n){
	int sum = 0;
	for (int i = 0; i < n - 3; i += 4) {
        sum += a[i];
        sum += a[i+1];
        sum += a[i+2];
        sum += a[i+3];  
    }
    return sum;
}

int Rand(int val){
	int x = rand();
	return x % val;
}

void data_init(int scale){
	n = scale;
	for(int i = 0; i < n; ++i)
		a[i] = Rand(1000);
}

int scale;
int rep;

int main(){

	srand(static_cast<unsigned int>(time(0)));
	float time_use = 0;
    struct timeval start;
    struct timeval end;


    for(int i = 100000; i <= 1000000; i += 10000){

 		scale = i;  	
    	data_init(scale);
    	rep = 100000000 / scale;
 		int r = rep;

    	gettimeofday(&start, NULL);
    	while(r != 0){
    		--r;
    		simple_calc(a, n);
    	}
    	gettimeofday(&end,NULL);
    	
    	printf("scale = %d, repeat = %d \n", scale, rep);
    	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    	printf("simple calc's total time_use is %.10f μs, single time_use is %.10f μs\n", time_use, time_use / rep);

    	r = rep;

    	gettimeofday(&start, NULL);
    	while(r != 0){
    		--r;
    	    advanced_calc(a, n);
    	}
    	gettimeofday(&end,NULL);
    	
    	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    	printf("advanced calc's total time_use is %.10f μs, single time_use is %.10f μs\n ",time_use, time_use / rep);

    	r = rep;

    	gettimeofday(&start, NULL);
    	while(r != 0){
    		--r;
    		unroll_calc(a, n);
    	}
    	gettimeofday(&end,NULL);
    	
    	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    	printf("unroll calc's total time_use is %.10f μs, single time_use is %.10f μs\n ",time_use, time_use / rep);
    	printf("\n");
	}
	return 0;
}
