
#include <stdio.h>
#include <math.h>
#include <cuda.h>

float *y, *x; 


//GPU kernel 
__global__
void calc(float *Y, float *X){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Y[i] = sqrt(X[i])/X[i];
}

//CPU function
void calc_h(float *Y, int N){

    for (int x = 0; x < N; x++){
       Y[x] = sqrt(x)/x;
    }
}

int main(int argc,char **argv)
{

    //Iterations
    int n=400000000;
    //Size
    int nBytes = n*sizeof(float);


    //memory allocation	
    y = (float *) malloc(nBytes);
    x = (float *) malloc(nBytes);

    float *y_d;
    float *x_d;

    //Block size and number
    int block_size, block_no;

    block_size = 250; //threads per block
    block_no = n/block_size;
    
    //Work definition
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(block_no, 1, 1);

    // Data filling
    for(int i=0; i < n ;i++)
       x[i]=i;


   //GPU memory allocation
    cudaMalloc((void **) &y_d, n*sizeof(float));
    cudaMalloc((void **) &x_d, n*sizeof(float));

    //Copy fron host to device
    cudaMemcpy(y_d, y, n*sizeof(float), cudaMemcpyHostToDevice);

    clock_t start_d=clock();
    //GPU 
    calc<<<block_no,block_size>>>(y_d, x_d);

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();
    
    //CPU calc
    clock_t start_h = clock();
    calc_h(y,n);
    clock_t end_h = clock();
	
    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    //Copying data back to host, this is a blocking call and will not start until all kernels are finished
    cudaMemcpy(y, y_d, n*sizeof(float), cudaMemcpyDeviceToHost);
    printf("n = %d \t GPU time = %fs \t CPU time = %fs\n", n, time_d, time_h);

    //Free GPU memory
    cudaFree(x_d);
    cudaFree(y_d);

    return 0;
}
