#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void matrixVectorMult(float *matrix, float *vector, float *result,int N,int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < N && idy<N) {
        
        result[0] = matrix[idx*k]*vector[idy];
    }
}

int main(void)
{
    float elapsed_time;
    int nstreams = 32;
    // Initialize matrix and vector
    float matrix[nstreams*nstreams];
    float vector[nstreams];
    for (int i = 0; i < nstreams * nstreams; i++) {
        matrix[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < nstreams; i++) {
        vector[i] = (float)(rand() % 10);
    }

    // Allocate memory on device
    float *d_matrix, *d_vector, *d_result;
    cudaMalloc((void **)&d_matrix, nstreams * nstreams * sizeof(float));
    cudaMalloc((void **)&d_vector, nstreams * sizeof(float));
    cudaMalloc((void **)&d_result, nstreams * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_matrix, matrix, nstreams * nstreams * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, nstreams * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate and initialize an array of stream handles
    cudaStream_t *streams =(cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++) {
    cudaStreamCreate(&(streams[i]));
    }

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    // Launch kernel
    dim3 block_size(16, 16, 1);
    dim3 grid_size((nstreams + block_size.x - 1) / block_size.x,(nstreams + block_size.y - 1) / block_size.y , 1);
    
    for(int i = 0; i < nstreams; ++i){
    matrixVectorMult<<<grid_size, block_size,0,streams[i]>>>(d_matrix, d_vector, &d_result[i],nstreams,i);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time,start_event,stop_event);
    printf("Measured time for sample = %.3fus\n", elapsed_time);
    
    // Copy data from device to host
    float result[nstreams];
    cudaMemcpy(result, d_result, nstreams * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
//    printf("Matrix:\n");
//    for (int i = 0; i < nstreams; i++) {
//        for (int j = 0; j < nstreams; j++) {
//            printf("%f ", matrix[i * nstreams + j]);
//        }
//        printf("\n");
//    }
//    printf("Vector:\n");
//    for (int i = 0; i < nstreams; i++) {
//        printf("%f ", vector[i]);
//    }
//    printf("\n");
    printf("Result:\n");
    for (int i = 0; i < nstreams; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");
    
    // Release resources
    for (int i = 0; i < nstreams; i++) {
      cudaStreamDestroy(streams[i]);
    }

    free(streams);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // Free memory on device
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return 0;
}

