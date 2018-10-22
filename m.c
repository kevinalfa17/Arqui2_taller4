/* Instituto Tecnologico de Costa Rica
 * Area academica de Ingenieria en Computadores
 * Curso de Arquitectura en Computadores II
 * Profesor: Jeferson Gonzales Gomez 
 * Taller 4: Multiplicacion de matrices con cuda
 * Kevin Alfaro Vega - 2015027603
 * II Semestre 2018
*/

#include<cuda.h>
#include<stdio.h>



int main(void) {

	float a[16];
	float b[16];
	float c[16];

	int index = 0; 

	printf("Llenar matrix 1\n");

	for(int i = 0; i < 4 ; i++){
	
		for(int j = 0; j < 4 ; j++){
			index = j+i*4;
			printf("F: %d C: %d\n",i,j);
			scanf("%f",&a[index]);	
		}
	}

	printf("Llenar matrix 2\n");

	for(int i = 0; i < 4 ; i++){

		for(int j = 0; j < 4 ; j++){
			index = j+i*4;
			printf("F: %d C: %d\n",i,j);
			scanf("%f",&b[index]);
			//Initialize c matrix	
			c[index] = 0;	
		}
	}


    	void multiplication(float *, float *, float *);
    	multiplication(a, b, c);

	//Print result
	for(int i = 0; i < 4 ; i++){

		for(int j = 0; j < 4 ; j++){
			index = j+i*4;
			printf("%f ",c[index]);
		}

		printf("\n");

	}

    return 0;
}

//Kernel to compute multiplication
__global__ void kernel(float *a_d, float *b_d, float *c_d) {
    
    //2D Threads
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float c_temp = 0;

    for(int i = 0; i < 4 ;i++) {

        float a_d_temp = a_d[ty*4 + i];
        float b_d_temp = b_d[i*4 + tx];

        c_temp += (a_d_temp * b_d_temp);
    }

    c_d[ty*4 + tx] = c_temp;

}

void multiplication(float *a, float *b, float *c) {
    int size = 16*sizeof(float);

    //memory used by GPU
    float *a_d, *b_d, *c_d;

    //copy a device memory
    cudaMalloc((void**)&a_d, size);
    cudaMemcpy(a_d,a,size,cudaMemcpyHostToDevice);

    //copy b to device memory
    cudaMalloc((void**)&b_d, size);
    cudaMemcpy(b_d,b,size,cudaMemcpyHostToDevice);

    //allocate memory for result c
    cudaMalloc((void**)&c_d,size);

    //setup the grid and blocks
    dim3 dimBlock(4,4);
    dim3 dimGrid(1,1);

    //compute multiplication
    kernel<<<dimGrid,dimBlock>>>(a_d,b_d,c_d);

    //copy c_d device memory into cpu memory
    cudaMemcpy(c,c_d,size,cudaMemcpyDeviceToHost);

    //free device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
