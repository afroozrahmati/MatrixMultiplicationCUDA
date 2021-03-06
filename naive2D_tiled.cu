
/*
 * CSS-535 Program 2: Matrix Multiplication on GPU
 * Author: Afrooz Rahmati
 * 02-23-2021
 *
 * special thanks to Tony Varela, I worked with him on previous lab for 
 * vector-matrix multiplication, so I used some of our previous implementations 
 * our previous work is here (private): https://github.com/valenotary/CSS-535-Lab-03-GEMV
 * Compile command =  nvcc -std=c++11  naive1D.cu  -lcublas -o naive1D 
 * Run command with profiling = ./naive1D matrix_size number_of_Blocks number_of_threads 
 * Run example : ./naive1D 1000 1000 1000
 */


 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <cublas_v2.h> // as a benchmark 
 #include <random> // for random initialization
 #include <chrono> // timing
 #include <iostream> // for output 
 using namespace std;
 using namespace std::chrono;
 
 
#define TILE_WIDTH 27

 /*
 * Functionality: Initializing Matrix with float values other than 0 and 1
 * input:
		 m = the input matrix
		 M, N = matrix dimensions M X N
   output:
		 None      
 */
 void initialize_matrix(float *m, const int M, const int N) {
	 //std::default_random_engine dre;
	// dre.seed(3); // seeded for reproducibility
	// const std::uniform_real_distribution<float> uniform_dist(-10, 10); // uniform distribution [-10, 10]

	 for (size_t i=0; i < M; i++) 
		//just to ignore 0 and 1 for error handling
		//This will generate a number from some 1 to some arbitrary HI=99998
		 for (size_t j=0; j < N; j++) 
			 m[i * M + j] = 1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(9)));
			 
 }
 
 /* 
 functionality: naive2D + tiled Matrix Multiplication Implementation a*b=c
 input parameters: 
			 a         : the input matrix
			 b         : the input matrix
			 c         : the result matrix
			 N         : Matrices dimension( number of elements )
 consideration : the matrix size is square */
 
 __global__ void multiplication(float *a, float *b, float *c, const int n){
	 
    unsigned int col = TILE_WIDTH * blockIdx.x + threadIdx.x ;
    unsigned int row = TILE_WIDTH * blockIdx.y + threadIdx.y ;
    
    if( col < n && row < n) 
    {
        for (int k = 0 ; k < n ; k++ )
        {
            c[row * n + col]+= a[row * n + k ] * b[ k * n + col] ;
        }
    }
 
 }
 
 
 
 //functionality: Computing the max residual between two matrices 
 //pre condition : M as matrices size ,resultVector2 and resultVector are two matrixes stores in 1D dimensions
 //post condition: return the residual value as double precision
 float residual(int M, float* resultVector2, float* resultVector) {
 
	 //float* k = new float[M];
	 float max_res =0.0;
	 float accum = 0.0;
	 for (int i = 0; i < M; ++i) {
		 accum = fabs(resultVector2[i] - resultVector[i]);
		 max_res=max(max_res,accum) ;
	 }
	
	 return max_res;
 
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
 
 
 /*
 * Functionality: transpose the matrix - for comparing the result with cublas  
 * input:
		 src = the input matrix
		 dst = expected transpose matrix
		 M, N = matrix dimensions M X N
		 
   output:
		 None      
 */
 void transpose(float *src, float *dst, const int N, const int M) {
	 for(int n = 0; n<N*M; n++) {
		 int i = n/N;
		 int j = n%N;
		 dst[n] = src[M*j + i];
	 }
 }
 
 
 
 int main(int argc, char **argv) {
 
	 // This program should only take in the M and N dimensions; within the program, we figure out the execution configurations ourselves		
	 if( argc != 2 )
	 {      
		 std::cout << "please enter matrix size " << std::endl;
		 return 0; 
	 }
	 
	 const size_t N = atoi(argv[1]) ;
	 // let's create the grid / block configuration, but just really simply.
	 dim3 blockPerGrid (ceil(N/(float)TILE_WIDTH), ceil(N/(float)TILE_WIDTH) );      
	 dim3 threadPerBlock (TILE_WIDTH, TILE_WIDTH );
 

	 // cublas declarations
	 cublasHandle_t cublas_handle;
 
	 // allocate host memory
	 float *a =  (float*)malloc( N * N * sizeof(float) );
	 float *b = (float*)malloc( N * N * sizeof(float) );
	 float *c_out_naive= (float*)malloc( N * N * sizeof(float) );
	 float *c_out_cublas= (float*)malloc( N * N * sizeof(float) );
	 float *c_transposed_cublas= (float*)malloc( N * N * sizeof(float) );

 
	 // allocate device memory
	 float *d_a, *d_b, *d_c_out_naive, *d_c_out_cublas;
	 cudaMalloc((void**)&d_a, N * N * sizeof(float));
	 cudaMalloc((void**)&d_b, N * N * sizeof(float));
	 cudaMalloc((void**)&d_c_out_naive, N * N * sizeof(float));
	 cudaMalloc((void**)&d_c_out_cublas, N * N * sizeof(float));
  
	 // initialize host array with random data
	 initialize_matrix(a, N, N);
	 initialize_matrix(b, N, N);
 
 
   /* The elements of the first column */
    // a[0] = 1;
    // a[1] = 2;
 	// a[2] = 3;
 	// a[3] = 4;
    // /* The elements of the second column */	
 	// a[N] = 1;
 	// a[N + 1] = 1;
 	// a[N + 2] = 2;
 	// a[N + 3] = 1;
    // /* The elements of the third column */
 	// a[N * 2] = 3;
 	// a[N * 2 + 1] = 1;
 	// a[N * 2 + 2] = 2;
 	// a[N * 2 + 3] = 1;
    // /* The elements of the fourth column */
 	// a[N * 3] = 5;
 	// a[N * 3 + 1] = 4;
 	// a[N * 3 + 2] = 7;
 	// a[N * 3 + 3] = 3;
 
 
    //  ////////////Second input matrix
    //  /* The elements of the first column */
    //  b[0] = 1;
    //  b[1] = 2;
    //  b[2] = 3;
    //  b[3] = 4;
    //  /* The elements of the second column */	
    //  b[N] = 5;
    //  b[N + 1] = 1;
    //  b[N + 2] = 8;
    //  b[N + 3] = 1;
    //  /* The elements of the third column */
    //  b[N * 2] = 3;
    //  b[N * 2 + 1] = 1;
    //  b[N * 2 + 2] = 2;
    //  b[N * 2 + 3] = 1;
    //  /* The elements of the fourth column */
    //  b[N * 3] = 5;
    //  b[N * 3 + 1] = 4;
    //  b[N * 3 + 2] = 7;
    //  b[N * 3 + 3] = 3;

	// print_matrix(a, N, N, "input Matrix a");  
	// print_matrix(b, N, N, "input Matrix b");

 
	 std::cout << '\n';
	 // copy m and v_in into device memory, time it as well
	 auto d2h_start = std::chrono::high_resolution_clock::now();
	 cudaMemcpy(d_a, a, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	 cudaMemcpy(d_b, b, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	 auto d2h_end = std::chrono::high_resolution_clock::now();
	 auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(d2h_end - d2h_start).count();
 

	 // the naive2D kernel execution .....timing....
	 std::cout << "STARTING NAIVE" << std::endl;
	 auto naive_exec_start = std::chrono::high_resolution_clock::now();
	 multiplication<<<blockPerGrid, threadPerBlock>>>( d_a, d_b, d_c_out_naive, N);
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
	 
	 
	  cublasCreate(&cublas_handle);
	 // cublasSetMatrix(M, N, sizeof(float), m, M, )
 
 
	 const float alf = 1;
	 const float bet = 0;
	 const float *alpha = &alf;
	 const float *beta = &bet;
	 int lda=N,ldb=N,ldc=N;

     // try timing cublas (not timing inclusive times, although I am copying back out to host as well)
	 auto cublas_exec_start = std::chrono::high_resolution_clock::now();
	 cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, alpha, d_a, lda, d_b, ldb, beta, d_c_out_cublas, ldc);
	 auto cublas_exec_end = std::chrono::high_resolution_clock::now();
	 auto cublas_exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(cublas_exec_end - cublas_exec_start).count();
 
	 // copy the cublas device vector back out to host
	 cudaMemcpy(c_out_cublas, d_c_out_cublas, sizeof(float) *N* N, cudaMemcpyDeviceToHost);
	 std::cout << "Comparing output vectors:\n";
 
	 //calculating the residuals 
	 //Cublas data need to be transposed to compare with my kernel implementation 
	 //we can use GPU transpose, but here for simplicity I didn't go through that
	 float rse{ 0.0f };
	 transpose( c_out_cublas, c_transposed_cublas , N, N);
	 rse = residual(N*N,c_out_naive,c_transposed_cublas) ;

	 std::cout << "ERROR: " << rse << std::endl;
 
 
	 //print_test(c_out_naive,N);
	// print_matrix(c_out_naive, N, N, "output naive Matrix c"); 
	 //print_matrix(c_transposed_cublas, N, N, "output cublas Matrix c"); 
 
	 std::cout <<
		 " Naive Total Time , Naive Execution Time,\n";
	 std::cout <<naive_exec_duration +
	 gpu_transfer_total_duration << ", "  << naive_exec_duration  
		  << '\n';
 
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
 