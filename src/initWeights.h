#include <algorithm>
#include <vector>
#include <string>
#include <array>
#include <iostream>
#include <complex>
#include "ConvStruct.h"
#include "npy.hpp"


template <typename Scalar> struct npy_data {
  std::vector<unsigned long> shape;
  bool fortran_order;
  std::vector<Scalar> data;
};

template <typename Scalar> npy_data<Scalar> test_load(const char * path); 

void readTwoKernel(ConvStruct::ArrOfVols const &target1, ConvStruct::ArrOfVols const &target2, const char * path, int dim[]);
void readKernel(ConvStruct::ArrOfVols const &target, const char * path, int dim[]);
void init_weights(ConvStruct ** layers);
void init_kernels(ConvStruct ** layers);
