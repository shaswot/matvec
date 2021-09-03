#include <iostream>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>

#include <boost/filesystem.hpp>

#define BLOCK_HEIGHT 64

// GLOBAL VARIABLES
uint LAYER_WIDTH = 512;
uint MODEL_SEED = 52233264;


// GPC_ID to get thread ID values
struct GPC_ID 
{
    uint t_idx, t_idy, t_idz;
    uint cta_idx, cta_idy, cta_idz;
    uint warp_id, sm_id, grid_id;
};

// https://stackoverflow.com/questions/612328/difference-between-struct-and-typedef-struct-in-c
typedef struct GPC_ID gpc_id;

// https://forums.developer.nvidia.com/t/any-way-to-know-on-which-sm-a-thread-is-running/19974/15
// https://www.codeproject.com/Articles/15971/Using-Inline-Assembly-in-C-C
__device__ gpc_id get_gpcid(void) 
{
     struct GPC_ID my_id;
     asm("mov.u32 %0, %tid.x;"    : "=r"(my_id.t_idx)    );
     asm("mov.u32 %0, %tid.y;"    : "=r"(my_id.t_idy)    );
     asm("mov.u32 %0, %tid.z;"    : "=r"(my_id.t_idz)    );

     asm("mov.u32 %0, %warpid;" : "=r"(my_id.warp_id) );
     asm("mov.u32 %0, %smid;"   : "=r"(my_id.sm_id)   );
     asm("mov.u32 %0, %gridid;"   : "=r"(my_id.grid_id)   );
     
     asm("mov.u32 %0, %ctaid.x;"  : "=r"(my_id.cta_idx)  );
     asm("mov.u32 %0, %ctaid.y;"  : "=r"(my_id.cta_idy)  );
     asm("mov.u32 %0, %ctaid.z;"  : "=r"(my_id.cta_idz)  );
     
     return my_id;
}

// Matrix-vector multiplication using CUDA
// Once CUDA Core is responsible for one element of the output matrix

template<typename T>
__global__ void MatMulKernel_naive(T *out, T *in, T *a, 
                             const int matrixHeight, 
                             const int matrixWidth,
                             gpc_id* myid) 
{
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;   
  
  // check boundry conditions
  if( row < matrixHeight)
  {
    T value = 0;
    for(int k = 0; k < matrixWidth; k++)
      value += a[row * matrixWidth + k] * in[k];
    // store results
    out[row] = value;
  }
}

template <class _Tp>
xt::xarray<_Tp> matvec_naive (xt::xarray<_Tp> matrix_A, 
                           xt::xarray<_Tp> vector_B)
{
  unsigned int n_rows = matrix_A.shape()[0];
  unsigned int n_cols = matrix_A.shape()[1];
  
  unsigned int size_A = n_rows * n_cols;
  unsigned int size_B = n_cols;
  assert (vector_B.shape()[0] == size_B && "matrix A and vector B shape mismatch.");
  assert (vector_B.shape()[1] == 1 && "vector B no. of columns != 1");
  unsigned int size_C = n_rows;
  
  // declare matrices for GPU and allocate memory
  
  // host copies of A,B,C
  _Tp *A = new _Tp[size_A];
  _Tp *B = new _Tp[size_B]; 
  _Tp *C = new _Tp[size_C];
  gpc_id *myid = new gpc_id[size_C];
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&A, size_A*sizeof(_Tp));
  cudaMallocManaged(&B, size_B*sizeof(_Tp));
  cudaMallocManaged(&C, size_C*sizeof(_Tp));
  cudaMallocManaged(&myid, size_C*sizeof(gpc_id));
  
  // Fill the matrix values from xtensor to C++ array
  for (int i = 0; i < size_A; i++)
  A[i] = matrix_A.flat(i);
   
  for (int i = 0; i < size_B; i++)
  B[i] = vector_B.flat(i);
  

  //run mat-vec multiplication
  // set up threading and blocking variables
  // Block Grid for MatMulKernel_naive<<< >>>
  // Each thread calculates on value of output vector by taking an entire row of matrix and the entire col. of vector
  // Threads are grouped into a 1-D Block Array
  dim3 dimBlock(BLOCK_HEIGHT, 1);
  int no_of_blocks = (int) ceil(n_rows / (double) BLOCK_HEIGHT);
  dim3 dimGrid(no_of_blocks,1);

  // time the matvel multiplication operation
  // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // execute kernel
  MatMulKernel_naive<float><<<dimGrid, dimBlock>>>(C, B, A, n_rows, n_cols, myid);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
   
  // Convert product vector to xtensor
  xt::xarray<double>::shape_type C_shape = {size_C, 1};
  xt::xarray<_Tp> vec_C = xt::adapt(C, size_C, xt::no_ownership(), C_shape);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return vec_C;
}

int main()
{
  // load weights from npy files
  boost::filesystem::path weight_folder("../weights");
  const std::string dense_weights_folder = "../weights/mnist_dense-w" + 
                                          std::to_string(LAYER_WIDTH) + 
                                          "x" + 
                                          std::to_string(LAYER_WIDTH) + 
                                          "-" + 
                                          std::to_string(MODEL_SEED);
  
  const std::string dense_weights_file = dense_weights_folder + "/mnist_dense-w" +
                                          std::to_string(LAYER_WIDTH) + 
                                          "x" + 
                                          std::to_string(LAYER_WIDTH) + 
                                          "-" + 
                                          std::to_string(MODEL_SEED) + "_dense_weights.npy";
  
//   std::cout << "******************************" << std::endl;
//   std::cout << "Weights: " << dense_weights_file << std::endl;
  xt::xarray<float> dense_weights = xt::load_npy<float>(dense_weights_file);
  xt::xarray<float> tr_dense_weights = xt::transpose(dense_weights);
  // load input vector from npy file
  uint image_no = 69999;
  const std::string input_vector_file = "../data/vector_" + std::to_string(image_no) + ".npy";
//   std::cout << "Input: " << input_vector_file << std::endl;
  xt::xarray<float> input_vector = xt::load_npy<float>(input_vector_file);
//   std::cout << "******************************" << std::endl;
  
//   std::cout << "Transposed Weight Matrix Shape: "<< xt::adapt(tr_dense_weights.shape()) << std::endl;
//   std::cout << "Input Vector Shape: "<< xt::adapt(input_vector.shape()) << std::endl;
//   std::cout << "******************************" << std::endl;
  
//   for (int i = 0; i < 10; ++i)
//   {
//     matvec_naive(tr_dense_weights, input_vector);
//   }
//   std::cout << "******************************" << std::endl;
  
  // Display Output
  auto matvecproduct = matvec_naive(tr_dense_weights, input_vector);
//   std::cout << "Matrix-Vector Product Shape: " << xt::adapt(matvecproduct.shape()) << std::endl;
//   std::cout << "Matrix-Vector Product" << std::endl;
//   std::cout << matvecproduct << std::endl;
//   std::cout << "******************************" << std::endl;
  return 0;
}

