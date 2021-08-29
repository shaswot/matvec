cmake_minimum_required(VERSION 3.10)

# specify CUDA architecture
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html

# Volta, RTX 2080, 2080Ti, 3080
set(CMAKE_CUDA_ARCHITECTURES "70;75;80")

# set the project name
project(matvec
    VERSION 1.0
    DESCRIPTION "GEMM CUDA"
    LANGUAGES CXX CUDA)
    
# specify the C++ standard
# C17 is not compatible because xtl/xtensor were installed using cxx_std_14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER nvc++)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(xtensor-blas REQUIRED)

# add Boost Library
# https://stackoverflow.com/questions/6646405/how-do-you-add-boost-libraries-in-cmakelists-txt
set(BOOST_ROOT "/usr/boost_1_77_0")
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.77.0 COMPONENTS program_options REQUIRED)
# https://stackoverflow.com/questions/3897839/how-to-link-c-program-with-boost-using-cmake/43885372#43885372

if(Boost_FOUND)
    add_executable(matvec src/matvec.cu) 
    target_link_libraries(matvec Boost::program_options ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()


## target_link_libraries(matvec -lcurand)
