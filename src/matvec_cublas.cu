#include <iostream>
#include <string>


#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <boost/filesystem.hpp>

#include <time.h>
#include <cblas.h>
#include <cublas_v2.h>



// GLOBAL VARIABLES
uint LAYER_WIDTH = 512;
uint MODEL_SEED = 52233264;

template <class _Tp>
xt::xarray<_Tp> matVecMul (xt::xarray<_Tp> matrix_A, 
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
  
  
  xt::xarray<float> matvecproduct = xt::linalg::dot(tr_dense_weights, input_vector);
  
  
  std::cout << "Matrix-Vector Product Shape: " << xt::adapt(matvecproduct.shape()) << std::endl;
  std::cout << "Matrix-Vector Product" << std::endl;
  std::cout << matvecproduct << std::endl;
  
  
    

  std::cout << "******************************" << std::endl;
  return 0;
}

