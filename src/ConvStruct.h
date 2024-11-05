#include "Volume.h"

#pragma once

struct ConvStruct{
	ConvStruct()=default;
	ConvStruct(int num_of_inputs, int num_of_outputs, int batchsize, int padded_imgsize, int kernel_size);
	typedef std::shared_ptr<Volume[]> ArrOfVols;
	
	int in, out, batchsize, padded_imgsize, kernel_size;
	ArrOfVols Aof, Aoe, Aok, Aok_gradient, Aok_back, Aom_adam, Aov_adam;
};