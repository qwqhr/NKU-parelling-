#include <iostream>
#include <cstdio>
#include <cstdlib> 
#include <ctime>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

using namespace std;
const int N=10500;

float sum[N];
float b[N][N],a[N];
int n;

void simple_calc(float b[N][N], float a[N], int n)
{
 	for (int i = 0; i < n; i++){
 		sum[i] = 0.0;
 		for (int j = 0; j < n; j++)
 			sum[i] += b[j][i] * a[j];
 		}
}

void advanced_calc(float b[N][N], float a[N], int n){
	for(int i = 0; i < n; i++)
		sum[i] = 0.0;
	for(int j = 0; j < n; j++)
		for(int i = 0; i < n; i++)
			sum[i] += b[j][i] * a[j];
}

float Rand(int val){
	int x =  rand() % (val * 10);
	return 1.0 * x / 10.0;
}

void data_init(int scale){
	n = scale;
	for(int i = 1; i <= n; ++i){
		a[i] = Rand(1000);
		for(int j = 1; j <= n; ++j){
			b[i][j] = Rand(1000);
		}
	}
}

int scale, cnt;
int rep;

int main(){
	srand(static_cast<unsigned int>(time(0)));
	float time_use = 0;
    struct timeval start;
    struct timeval end;


    for(int i = 100; i <= 1000; i += 20){

    	scale = i;
    	rep = 100000000/(scale*scale);

    	data_init(scale);
 		int r = rep;

    	gettimeofday(&start, NULL);
    	while(r != 0){
    		--r;
    		simple_calc(b, a, n);
    	}
    	gettimeofday(&end,NULL);
    	
    	printf("scale = %d, repeat = %d \n", scale, rep);
    	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    	printf("simple calc's total time_use is %.10f μs, single time_use is %.10f μs\n", time_use, time_use / rep);

    	 r = rep;

    	gettimeofday(&start, NULL);
    	while(r != 0){
    		--r;
    		advanced_calc(b, a, n);
    	}
    	gettimeofday(&end,NULL);
    	
    	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    	printf("advanced calc's total time_use is %.10f μs, single time_use is %.10f μs\n ",time_use, time_use / rep);
    	printf("\n");
	}
/*    
    data_init(1000);
    int rep=1000;
    
    while(rep--){
        simple_calc(b,a,n);
        cout<<rep<<" ";
    }
*/
	return 0;
}
