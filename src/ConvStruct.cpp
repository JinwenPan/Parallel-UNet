#include "ConvStruct.h"

ConvStruct::ConvStruct(int num_of_inputs, int num_of_outputs, int batchsize, int padded_imgsize, int kernel_size) : 
	in(num_of_inputs), 
	out(num_of_outputs),
	batchsize(batchsize),
	padded_imgsize(padded_imgsize),
	kernel_size(kernel_size),
	Aof(new Volume[batchsize]),		// array of feature channels
	Aoe(new Volume[batchsize]),		// array of error tensors
	Aok(new Volume[out]),			// array of kernels
	Aom_adam(new Volume[out]),		// for adam optimiser
	Aov_adam(new Volume[out]),		// for adam optimiser
	Aok_gradient(new Volume[out]),	// array of kernel gradients 
	Aok_back(new Volume[in])		// array of backward kernels
	{
		for (int i=0; i<batchsize; ++i){
			Aof[i]=Volume(in, padded_imgsize);
			Aoe[i]=Volume(in, padded_imgsize);
		}
		for (int i=0; i<out; ++i){
			Aok[i]=Volume(in, kernel_size);
			Aok_gradient[i]=Volume(in, kernel_size);
			Aom_adam[i]=Volume(in, kernel_size);
			Aov_adam[i]=Volume(in, kernel_size);
		}
		for (int i=0; i<in; ++i){
			Aok_back[i]=Volume(out, kernel_size);
		}
	}