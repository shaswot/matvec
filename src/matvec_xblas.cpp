#include <iostream>
#include <string>
#include <chrono>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <boost/filesystem.hpp>
// GLOBAL VARIABLES
uint LAYER_WIDTH = 512;
uint MODEL_SEED = 52233264;


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
  
  // Time the actual operation
  // https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
  for (int i = 0; i < 10; ++i)
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    xt::linalg::dot(tr_dense_weights, input_vector);
    auto t2 = std::chrono::high_resolution_clock::now();
    //Getting number of milliseconds as a double
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Execution Time: " << ms_double.count() << "ms\n";  
  }
    std::cout << "******************************" << std::endl;

//   // Display Output
//   auto matvecproduct = xt::linalg::dot(tr_dense_weights, input_vector);
//   std::cout << "Matrix-Vector Product Shape: " << xt::adapt(matvecproduct.shape()) << std::endl;
//   std::cout << "Matrix-Vector Product" << std::endl;
//   std::cout << matvecproduct << std::endl;
//   std::cout << "******************************" << std::endl;
  return 0;
}
