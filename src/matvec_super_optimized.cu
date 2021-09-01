#include <iostream>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>

#include <boost/filesystem.hpp>

#define BLOCK_HEIGHT 64
#define BLOCK_WIDTH 98

// GLOBAL VARIABLES
uint LAYER_WIDTH = 512;
uint MODEL_SEED = 52233264;


// // Matrix-vector multiplication using CUDA
// // Using shared memory and avoiding banking conflicts
template<typename T>
__global__ void MatMulKernel(T *out, T *in, T *a, 
                             const int matrixHeight, 
                             const int matrixWidth)
{
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // looping is happening horizontally on the matrix
  // BLOCK_WIDTH is again horizontal
  // BLOCK_HEIGHT is going vertical
  // n_cols / BLOCK_WIDTH blocks horizontally
  // n_rows / BLOCK_HEIGHT block vertically

  // get variables for loop
  // variable for loop length: blockEltHeight
  __shared__ int blockElt;
  __shared__ int blockxInd;
  __shared__ int blockyInd;
  if (threadIdx.x == 0) // only the first thread of the entire block initializes the shared variables blockElt, blockxInd, blockyInd.
  {  
    if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
      blockElt = BLOCK_WIDTH; // NOT the rightmost block so width of block = BLOCK_WIDTH
    else blockElt = matrixWidth % BLOCK_WIDTH; // rightmost block so width of block = matrixWidth % BLOCK_WIDTH
    blockxInd = blockIdx.x * BLOCK_WIDTH; // top left thread x-index of the block
    blockyInd = blockIdx.y * BLOCK_HEIGHT; // top left thread y-index of the block
  }
  
  __syncthreads(); //all threads have value of blockElt, blockxInd, blockyInd
  
  // copy section of b into shared mem
  // https://stackoverflow.com/questions/24419822/efficiently-initializing-shared-memory-array-in-cuda/24419969#24419969
  // use threads to write into independent locations of b[] from in []
  __shared__ T b[BLOCK_WIDTH];
  __shared__ T in_sub[BLOCK_HEIGHT][BLOCK_WIDTH + 31];
//   __shared__ T in_sub[BLOCK_HEIGHT][BLOCK_WIDTH];

  
  int threads_per_block = BLOCK_HEIGHT;
  int lidx = threadIdx.x;
  while (lidx < BLOCK_WIDTH)
  {
    b[lidx] = in[lidx + blockIdx.x * BLOCK_WIDTH];
    lidx += threads_per_block;
  }  
  __syncthreads();
  
  for (int i=0; i<blockElt; i++) //each thread loads one sub-row of matrix a[].
  {
    in_sub[threadIdx.x][i] = a[(blockyInd + threadIdx.x) * matrixWidth + blockxInd + i];
  }
  __syncthreads();
  
   
  // summing variable
  T cSum = (T) 0.0;
  int threadyInd = blockyInd + threadIdx.x;

  // make sure we are inside the matrix verticallly
  if (threadyInd < matrixHeight) 
  {
    // each thread computes one element of a block segment of the output vector
    
    for (int i=0; i<blockElt; i++)
    {
      // row R of matrix a[] --> (blockIdx.y * BLOCK_HEIGHT + threadIdx.x) * matrixWidth = (blockyInd + threadIdx.x) * matrixWidth
      // col C of row R of matrix a[] --> blockIdx.x * BLOCK_WIDTH = blockxInd
      // element E of col C of row R of matrix a[] --> i
      // b[i] is accessed by all threads and therefore it is broadcast without any banking conflicts.
//       cSum += b[i] * a[(blockyInd + threadIdx.x) * matrixWidth + blockxInd + i]; //working version
      cSum += in_sub[threadIdx.x][i] * b[i];

//       if (i==blockElt-1)
//       printf("blockxInd = %d, blockyInd = %d, threadIdx.x = %d, csum = %f\n", blockxInd, blockyInd, threadIdx.x, cSum);
    }
    // atomic add these variables to the corresponding c index
    atomicAdd(out + threadyInd, cSum);

  }
  
}

template <class _Tp>
xt::xarray<_Tp> matvec_banking (xt::xarray<_Tp> matrix_A, 
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
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&A, size_A*sizeof(_Tp));
  cudaMallocManaged(&B, size_B*sizeof(_Tp));
  cudaMallocManaged(&C, size_C*sizeof(_Tp));
  
  // Fill the matrix values from xtensor to C++ array
  for (int i = 0; i < size_A; i++)
  A[i] = matrix_A.flat(i);
   
  for (int i = 0; i < size_B; i++)
  B[i] = vector_B.flat(i);
  

  //run mat-vec multiplication
  // set up threading and blocking variables
  // Block Grid for MatMulKernel<<< >>>
  int blockCols = (int) ceil(n_cols / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(n_rows / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT); // BLOCK_HEIGHT directly corresponds to no. of threads per block i.e., one thread per row of the block.
  dim3 dimGrid(blockCols, blockRows);
  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof(_Tp) + BLOCK_HEIGHT*(BLOCK_WIDTH + 31) * sizeof(_Tp);
  // 31 is for padding s.t. (98+31) mod 32 = 1
  // 3 * sizeof (int) -> to store blockElt, blockxInd, blockyInd;

  // initialize vector C to zero
  cudaMemset(C, 0, n_rows*sizeof(_Tp));
  // execute kernels
  MatMulKernel<float><<<dimGrid, dimBlock, sharedMem>>>(C, B, A, n_rows, n_cols);
  cudaDeviceSynchronize();
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
//     matvec_banking(tr_dense_weights, input_vector);
//   }
//   std::cout << "******************************" << std::endl;
  
  // Display Output
  auto matvecproduct = matvec_banking(tr_dense_weights, input_vector);
//   std::cout << "Matrix-Vector Product Shape: " << xt::adapt(matvecproduct.shape()) << std::endl;
//   std::cout << "Matrix-Vector Product" << std::endl;
//   std::cout << matvecproduct << std::endl;
//   std::cout << "******************************" << std::endl;
  return 0;
}

