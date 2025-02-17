
/*
 * CSS-535 Program 2: Matrix Multiplication on GPU
 * Author: Afrooz Rahmati
 * 02-23-2021
 *
 * special thanks to Tony Varela, I worked with him on previous lab for 
 * vector-matrix multiplication, so I used some of our previous implementations 
 * our previous work is here (private): https://github.com/valenotary/CSS-535-Lab-03-GEMV
 *
 */


 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <cublas_v2.h> // as a benchmark 
 #include <random> // for random initialization
 #include <chrono> // timing
 #include <iostream> // for output 
 using namespace std;
 using namespace std::chrono;
 
 #define BLOCK_SIZE 2
 #define TILE_WIDTH 2

 /*
 * Functionality: Initializing Matrix with float values other than 0 and 1
 * input:
         m = the input matrix
         M, N = matrix dimensions M X N
   output:
         None      
 */
 void initialize_matrix(float *m, const int M, const int N) {
     // seed RNG
     std::default_random_engine dre;
     dre.seed(3); // seeded for reproducibility
     std::uniform_real_distribution<float> uniform_dist(-10, 10); // uniform distribution [-10, 10]
 
     for (size_t i=0; i < M; i++) 
         for (size_t j=0; j < N; j++) 
             m[i * M + j] = 1 + uniform_dist(dre);  //just to ignore 0 and 1 for error handling
 }
 
 /* 
 functionality: naive Matrix Multiplication Implementation a*b=c
 input parameters: 
             a         : the input matrix
             b         : the input matrix
             c         : the result matrix
             N         : Matrices dimension( number of elements )
 consideration : the matrix size is square */
 
 __global__ void multiplication(float *a, float *b, float *c, const int n){
     
     int row = blockIdx.y * blockDim.y + threadIdx.y; 
     int col = blockIdx.x * blockDim.x + threadIdx.x;
 
 
     float sum = 0.0f;
     if( col < n && row < n) 
     {
         for(int i = 0; i < n; i++) 
             sum += a[row * n + i] * b[i * n + col];
     }
     c[row * n + col] = sum;
 }
 


 __global__ void matMulGPU(float* A, float* B, float* C, int numARows, int numACols, int numBCols) {
    // allocate shared memory
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // coordinates for C
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float cumSum = 0;
    for (int m = 0; m < ceil(numACols/(float)TILE_WIDTH); m++) {
        // load tiles
        if ((row < numARows) && ((m*TILE_WIDTH + tx) < numACols))
            sharedA[ty][tx] = A[row*numACols + m*TILE_WIDTH + tx] ;
        else
            sharedA[ty][tx] = 0;
        if ((col < numBCols) && ((m*TILE_WIDTH + ty) < numACols))
            sharedB[ty][tx] = B[(m*TILE_WIDTH + ty)*numBCols + col];
        else
            sharedB[ty][tx] = 0;
        // pause until all threads have loaded tile values
        __syncthreads();

        // compute partial dot product (for individual thread)
        for (int k = 0; k < TILE_WIDTH; k++) {
            cumSum += sharedA[ty][k] * sharedB[k][tx];
        }
        // wait until all threads have used tile values
        __syncthreads();
    }
    if((row < numACols) && (col < numBCols)) {
        C[row*numBCols + col] = cumSum;
    }
}
 
 
 //functionality: Computing the residual between two matrices 
 //pre condition : M as matrices size ,resultVector2 and resultVector are two matrixes stores in 1D dimensions
 //post condition: return the residual value as double precision
 float residual(int M, float* resultVector2, float* resultVector) {
 
     float* k = new float[M];
     float accum = 0.0;
     for (int i = 0; i < M; ++i) {
         k[i] = pow(fabs(resultVector2[i] - resultVector[i]), 2);
         accum += k[i];
     }
     float norm = sqrt(accum);
 
     return norm;
     //std::cout << "Residual : " << norm<<endl;
 
 }
 
 
 
 /*
 * Functionality: Printing Matrix values for Debugging purpose only 
 * input:
         a = the input matrix
         M, N = matrix dimensions M X N
         d = The text before each value
   output:
         None      
 */
 void print_matrix(float *a, const int N, const int M, char *d) {
     int i, j;
     for(i=0; i<N; i++)
     { 
         printf("\n%s[%d]:", d, i);
         for (j=0; j<M; j++)
             printf("\t%6.4f", a[i*M+j]);
     }
     printf("\n");
 }
 
 
 int main(int argc, char **argv) {
     // TODO: create command line arguments to configure grid/block dimensions
     // This program should only take in the M and N dimensions; within the program, we figure out the execution configurations ourselves
     
 
     const size_t N = 4 ;
     // let's create the grid / block configuration, but just really simply.
     dim3 grid = 4; // (1, 1, 1)
     dim3 block = 36;
 
     // cublas declarations
     cublasHandle_t cublas_handle;
 
     // for now, let's put the matrix/vector dimensions in here as well
     //
     // yes, I know they're always going to be square, but I like separating M and N for my own understanding.
     // TODO: consider experimenting with thrust device/host vectors as well
 
     // allocate host memory
     float *a =  (float*)malloc( N * N * sizeof(float) );
     float *b = (float*)malloc( N * N * sizeof(float) );
     float *c_out_naive= (float*)malloc( N * N * sizeof(float) );
     float *c_out_cublas= (float*)malloc( N * N * sizeof(float) );
 
     // allocate device memory
     float *d_a, *d_b, *d_c_out_naive, *d_c_out_cublas;
     cudaMalloc((void**)&d_a, N * N * sizeof(float));
     cudaMalloc((void**)&d_b, N * N * sizeof(float));
     cudaMalloc((void**)&d_c_out_naive, N * N * sizeof(float));
     cudaMalloc((void**)&d_c_out_cublas, N * N * sizeof(float));
 
     // cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * N * N);
     // cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float)  * N * N);
     // cudaMalloc(reinterpret_cast<void**>(&d_c_out_naive), sizeof(float) * N * N);
     // cudaMalloc(reinterpret_cast<void**>(&d_c_out_cublas), sizeof(float)  * N * N);
 
     //**************************These lines are for debugging purpose only************************
 
   /* The elements of the first column */
     a[0] = 1;
     a[1] = 2;
     a[2] = 3;
     a[3] = 4;
    /* The elements of the second column */	
     a[N] = 1;
     a[N + 1] = 1;
     a[N + 2] = 2;
     a[N + 3] = 1;
    /* The elements of the third column */
     a[N * 2] = 3;
     a[N * 2 + 1] = 1;
     a[N * 2 + 2] = 2;
     a[N * 2 + 3] = 1;
    /* The elements of the fourth column */
     a[N * 3] = 5;
     a[N * 3 + 1] = 4;
     a[N * 3 + 2] = 7;
     a[N * 3 + 3] = 3;
 
 
     ////////////Second input matrix
     /* The elements of the first column */
     b[0] = 1;
     b[1] = 2;
     b[2] = 3;
     b[3] = 4;
     /* The elements of the second column */	
     b[N] = 1;
     b[N + 1] = 1;
     b[N + 2] = 2;
     b[N + 3] = 1;
     /* The elements of the third column */
     b[N * 2] = 3;
     b[N * 2 + 1] = 1;
     b[N * 2 + 2] = 2;
     b[N * 2 + 3] = 1;
     /* The elements of the fourth column */
     b[N * 3] = 5;
     b[N * 3 + 1] = 4;
     b[N * 3 + 2] = 7;
     b[N * 3 + 3] = 3;
   
 
 ///////////////////////**************************************************
 
     // initialize host array with random data
 
     print_matrix(a, N, N, "input Matrix a");  
     print_matrix(b, N, N, "input Matrix b");
 
     std::cout << '\n';
     // copy m and v_in into device memory, time it as well
     auto d2h_start = std::chrono::high_resolution_clock::now();
     cudaMemcpy(d_a, a, sizeof(float) * N * N, cudaMemcpyHostToDevice);
     cudaMemcpy(d_b, b, sizeof(float) * N * N, cudaMemcpyHostToDevice);
     auto d2h_end = std::chrono::high_resolution_clock::now();
     auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(d2h_end - d2h_start).count();
 
     // TODO: there are CUBLAS operations for getting/setting matrices/vectors between host/device; consider looking/timing these as well: https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf (pg.48-49)
 
     // let's create the grid / block configuration, but just really simply.
     
     //*****************************************************************************************
     /////////////////specific to part 2////////////////////////////////////////////////////////////////
     //const unsigned int BLOCK_SIZE = 16; ///we need to change it
     //unsigned int gridrows =  (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
 
     //dim3 grid ( 1 , 1 );  
     //dim3 block(BLOCK_SIZE,BLOCK_SIZE);
     
     //dim3 threadPerBlock(32, 32);
     //dim3 blockPerGrid(4, 1);
 
     dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
     dim3 blockPerGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);

 
     std::cout << "STARTING NAIVE" << std::endl;
     auto naive_exec_start = std::chrono::high_resolution_clock::now();
     matMulGPU<<<blockPerGrid, threadPerBlock>>>( d_a, d_b, d_c_out_naive, N,N,N);
    
     //naive_gemv <<<grid, block >>>(d_m, d_v_in, d_v_out_naive, M, N);
     cudaDeviceSynchronize();
     std::cout << "FINISHED NAIVE" << std::endl;
     // since the kernels are executed asynchronously, need to sync so that we can get accurate timing
     auto naive_exec_end = std::chrono::high_resolution_clock::now();
     auto naive_exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(naive_exec_end - naive_exec_start).count();
     
 
 
     // // copy d_v_out_naive back into host
     auto h2d_start = std::chrono::high_resolution_clock::now();
     cudaMemcpy(c_out_naive, d_c_out_naive, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
     auto h2d_end = std::chrono::high_resolution_clock::now();
     auto h2d_duration = std::chrono::duration_cast<std::chrono::microseconds>(h2d_end - h2d_start).count();
 
     // // get total inclusive time 
      auto gpu_transfer_total_duration = h2d_duration + d2h_duration;
     
     // try timing cublas (not timing inclusive times, although I am copying back out to host as well)
      cublasCreate(&cublas_handle);
     // cublasSetMatrix(M, N, sizeof(float), m, M, )
 
 
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;
     int lda=N,ldb=N,ldc=N;
 
     auto cublas_exec_start = std::chrono::high_resolution_clock::now();
     
     cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha, d_a, lda, d_b, ldb, beta, d_c_out_cublas, ldc);
 
     auto cublas_exec_end = std::chrono::high_resolution_clock::now();
     auto cublas_exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(cublas_exec_end - cublas_exec_start).count();
 
     // copy the cublas device vector back out to host
     cudaMemcpy(c_out_cublas, d_c_out_cublas, sizeof(float) *N* N, cudaMemcpyDeviceToHost);
 
     std::cout << "Comparing output vectors:\n";
 
     float rse{ 0.0f };
     rse = residual(N*N,c_out_naive,c_out_cublas) ;
 
     
     // for (size_t i{ 0 }; i < N; i++) 
     //     rse += abs(c_out_naive[i] - c_out_cublas[i]);
     std::cout << "ERROR: " << rse << std::endl;
 
 /////// 
 
 //print_vector(d_v_out_cublas, M, "out vector");
 
 
     // std::cout << "Naive: ";
     // for (size_t i = 0 ; i < N; i++) 
     //     std::cout << c_out_naive[i] << ' ';
     // std::cout << '\n';
     
     // std::cout << "cuBLAS: ";
     // for (size_t i{0}; i < M; i++) std::cout << v_out_cublas[i] << ' ';
     // std::cout << '\n';
 
     print_matrix(c_out_naive, N, N, "output naive Matrix c"); 
     print_matrix(c_out_cublas, N, N, "output cublas Matrix c"); 
 
     std::cout <<
         "Total Inclusive Time, Naive Execution Time, cuBLAS Execution Time, Naive Total Time, cuBLAS Total Time\n";
     std::cout << gpu_transfer_total_duration << ", " << naive_exec_duration << ", " << cublas_exec_duration << ", " <<
         naive_exec_duration +
         gpu_transfer_total_duration << ", " << cublas_exec_duration + gpu_transfer_total_duration << '\n';
 
     //clean up
     cublasDestroy(cublas_handle);
 
     cudaFree(d_c_out_cublas);
     cudaFree(d_c_out_naive);
     cudaFree(d_b);
     cudaFree(d_a);
 
     delete[] c_out_cublas;
     delete[] c_out_naive;
     delete[] a;
     delete[] b;
 
 
     return 0;
 }
 