#include <iostream>
#include <string>
#include <cublas_v2.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>

#include <boost/filesystem.hpp>

// GLOBAL VARIABLES
uint LAYER_WIDTH = 512;
uint MODEL_SEED = 52233264;

template <class _Tp>
xt::xarray<_Tp> matVec_cublas (xt::xarray<_Tp> matrix_A, 
                           xt::xarray<_Tp> vector_B)
{
  unsigned int n_rows = matrix_A.shape()[0];
  unsigned int n_cols = matrix_A.shape()[1];
  
  unsigned int size_A = n_rows * n_cols;
  unsigned int size_B = n_cols;
  assert (vector_B.shape()[0] == size_B && "matrix A and vector B shape mismatch.");
  assert (vector_B.shape()[1] == 1 && "vector B no. of columns != 1");
  unsigned int size_C = n_rows;
  
  //cublas handle
  cublasHandle_t handle;
  cublasCreate(&handle);
  
  // declare matrices for GPU and allocate memory
  
  // host copies of A,B,C
  _Tp *A = new _Tp[size_A];
  _Tp *B = new _Tp[size_B]; 
  _Tp *C = new _Tp[size_C];
//   gpc_id *myid = new gpc_id[size_C];
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&A, size_A*sizeof(_Tp));
  cudaMallocManaged(&B, size_B*sizeof(_Tp));
  cudaMallocManaged(&C, size_C*sizeof(_Tp));
//   cudaMallocManaged(&myid, size_C*sizeof(gpc_id));
  
  // Fill the matrix values from xtensor to C++ array
  for (int i = 0; i < size_A; i++)
  A[i] = matrix_A.flat(i);
   
  for (int i = 0; i < size_B; i++)
  B[i] = vector_B.flat(i);
  
  //run mat-vec multiplication
  float alpha = 1.0f, beta = 0.0f;
  cudaDeviceSynchronize();
  
  // time the matvel multiplication operation
  
  // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);

  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
  // https://stackoverflow.com/questions/16376804/clarification-of-the-leading-dimension-in-cublas-when-transposing
  // A (stored in row-major) is read as A_T when read in column major
  // So instead of A.B (in row-major), we do B_T.A_T
  // B_T = 1 x n_cols 
  // A_T = n_cols x n_rows
  cublasSgemm(handle,
              CUBLAS_OP_N, CUBLAS_OP_N, // B is read as B_T and A is read as A_T
              1, // rows of matrix B_T
              n_rows, // cols of A_T
              n_cols, // cols of matrix B_T
              &alpha,
              B, 1,
              A, n_cols,
              &beta,
              C, 1);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
  
  // Convert product vector to xtensor
  xt::xarray<double>::shape_type C_shape = {size_C, 1};
  xt::xarray<_Tp> vec_C = xt::adapt(C, size_C, xt::no_ownership(), C_shape);
  
  // free memory
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
  
  std::cout << "******************************" << std::endl;
  
  std::cout << "Weights: " << dense_weights_file << std::endl;
  xt::xarray<float> dense_weights = xt::load_npy<float>(dense_weights_file);
  xt::xarray<float> tr_dense_weights = xt::transpose(dense_weights);
  // load input vector from npy file
  uint image_no = 69999;
  const std::string input_vector_file = "../data/vector_" + std::to_string(image_no) + ".npy";
  std::cout << "Input: " << input_vector_file << std::endl;
  xt::xarray<float> input_vector = xt::load_npy<float>(input_vector_file);
  std::cout << "******************************" << std::endl;
  
  std::cout << "Transposed Weight Matrix Shape: "<< xt::adapt(tr_dense_weights.shape()) << std::endl;
  std::cout << "Input Vector Shape: "<< xt::adapt(input_vector.shape()) << std::endl;
  std::cout << "******************************" << std::endl;
  
  for (int i = 0; i < 10; ++i)
  {
    matVec_cublas(tr_dense_weights, input_vector);
  }
  std::cout << "******************************" << std::endl;
  
//   // Display Output
//   auto matvecproduct = matVec_cublas(tr_dense_weights, input_vector);
//   std::cout << "Matrix-Vector Product Shape: " << xt::adapt(matvecproduct.shape()) << std::endl;
//   std::cout << "Matrix-Vector Product" << std::endl;
//   std::cout << matvecproduct << std::endl;
  
//   std::cout << "******************************" << std::endl;
  return 0;
}

