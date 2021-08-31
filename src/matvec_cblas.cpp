#include <iostream>
#include <string>
#include <chrono>
#include <cblas.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>
// #include <xtensor-blas/xlinalg.hpp>

#include <boost/filesystem.hpp>
// GLOBAL VARIABLES
uint LAYER_WIDTH = 512;
uint MODEL_SEED = 52233264;

template <class _Tp>
xt::xarray<_Tp> matVec_cblas (xt::xarray<_Tp> matrix_A,
                              xt::xarray<_Tp> vector_B)
{
  unsigned int n_rows = matrix_A.shape()[0];
  unsigned int n_cols = matrix_A.shape()[1];
  
  unsigned int size_A = n_rows * n_cols;
  unsigned int size_B = n_cols;
  assert (vector_B.shape()[0] == size_B && "matrix A and vector B shape mismatch.");
  assert (vector_B.shape()[1] == 1 && "vector B no. of columns != 1");
  unsigned int size_C = n_rows;
  
  // allocate memory for A,B,C
  _Tp *A = new _Tp[size_A];
  _Tp *B = new _Tp[size_B]; 
  _Tp *C = new _Tp[size_C];
  
  // initialize input matrix
  for (int i = 0; i < size_A; i++)
  A[i] = matrix_A.flat(i);
   
  //initialize input vector
  for (int i = 0; i < size_B; i++)
  B[i] = vector_B.flat(i);
  
  float alpha = 1.0f, beta = 0.0f;
  // Time the actual operation
  // https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
  auto t1 = std::chrono::high_resolution_clock::now();
  //https://www.ibm.com/docs/en/essl/6.2?topic=mvs-sgemv-dgemv-cgemv-zgemv-sgemx-dgemx-sgemtx-dgemtx-matrix-vector-product-general-matrix-its-transpose-its-conjugate-transpose
  //  Y <- alpha*Ax + beta*y
  cblas_sgemv(CblasRowMajor, 
              CblasNoTrans, 
              n_rows, // number of rows in matrix A
              n_cols, // number of cols in matrix A
              alpha, // alpha: scaling constant
              A, // m by n matrix A
              n_cols, // leading dimension of the array specified for a
              B, // the vector x
              1, // stride for vector x
              beta, // beta: scaling constant
              C, // vector y
              1 // the stride for vector y
             );
  auto t2 = std::chrono::high_resolution_clock::now();
  //Getting number of milliseconds as a double
  std::chrono::duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "Execution Time: " << ms_double.count() << " ms" << std::endl;

  // Convert product vector to xtensor
  xt::xarray<double>::shape_type C_shape = {size_C, 1};
  xt::xarray<_Tp> vec_C = xt::adapt(C, size_C, xt::no_ownership(), C_shape);
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
  
  // Matrix-vector multiplication
  for (int i = 0; i < 10; ++i)
  {
    matVec_cblas(tr_dense_weights, input_vector);
  }
  std::cout << "******************************" << std::endl;

//   // Display Output
//   auto matvecproduct = matVec_cblas(tr_dense_weights, input_vector);
//   std::cout << "Matrix-Vector Product Shape: " << xt::adapt(matvecproduct.shape()) << std::endl;
//   std::cout << "Matrix-Vector Product" << std::endl;
//   std::cout << matvecproduct << std::endl;
//   std::cout << "******************************" << std::endl;
  return 0;
}

