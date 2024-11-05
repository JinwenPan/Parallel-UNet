#include "unetFuncs.h"

int batchsize;
int input_imgsize;
double learning_rate;
bool verbose;
bool debug;

void reluBackwards(ArrOfVols &image, ArrOfVols &error){
	int num_of_features = image[0].d;
	int imgsize = image[0].w; // includes padding (doesnt matter but just in case)
	for (int b=0; b<batchsize; ++b){
	    #pragma omp parallel for
		for(int i = 0; i < num_of_features; ++i){
			for(int x = 1; x < imgsize-1; ++x){
				for(int y = 1; y < imgsize-1; ++y){
					if(image[b](i,x,y) <= 0) error[b](i,x,y) = 0;
				}
			}
		}
	}
}
void relu(ArrOfVols &input){
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding (doesnt matter but just in case)
	for (int b=0; b<batchsize; ++b){
	    #pragma omp parallel for
		for(int i = 0; i < num_of_features; ++i){
			for(int x = 1; x < imgsize-1; ++x){
				for(int y = 1; y < imgsize-1; ++y){
					if(input[b](i,x,y) < 0) input[b](i,x,y) = 0;
				}
			}
		}
	}
}
void conv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output, bool dont_wipe_before_adding){
	int num_of_kernels = output[0].d;
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // imgsize includes padding (both input and output are padded)
	int width_of_kernels = kernel[0].w;

	if (dont_wipe_before_adding){
		for (int b=0; b<batchsize; ++b){
			for(int i = 0; i < num_of_kernels; ++i){	// just add to old output
				#pragma omp parallel for collapse(2)
				for(int x = 0; x < imgsize-2; ++x){
					for(int y = 0; y < imgsize-2; ++y){
						for(int j = 0; j < depth_of_kernels; ++j){							
							for(int n = 0; n < width_of_kernels; ++n){
								for(int m = 0; m < width_of_kernels; ++m){
									output[b](i,x+1,y+1) += input[b](j,x+n,y+m) * kernel[i](j,n,m); // i-th kernel applied to j-th input is added to i-th output
								}
							}
						}
					}
				}
			}
		}
	}
	else{
		for (int b=0; b<batchsize; ++b){
			for(int i = 0; i < num_of_kernels; ++i){	// zero out old output before adding
				#pragma omp parallel for collapse(2)
				for(int x = 0; x < imgsize-2; ++x){
					for(int y = 0; y < imgsize-2; ++y){
						double tmp = 0.0;
						for(int j = 0; j < depth_of_kernels; ++j){							
							for(int n = 0; n < width_of_kernels; ++n){
								for(int m = 0; m < width_of_kernels; ++m){
									tmp += input[b](j,x+n,y+m) * kernel[i](j,n,m); // i-th kernel applied to j-th input is added to i-th output							
								}
							}
						}
						output[b](i,x+1,y+1) = tmp;
					}
				}
			}
		}
	}
	if(debug) std::cout<< vol_avg(input) << "\t" << kernel_avg(kernel, output[0].d) << "\t" << vol_avg(output)<< std::endl;
}

void fullconv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output){
	int num_of_kernels = output[0].d;
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // imgsize is padded

	for (int b=0; b<batchsize; ++b){
	    #pragma omp parallel for
		for(int i = 0; i < num_of_kernels; ++i){
			for(int x = 1; x < imgsize-1; ++x){
				for(int y = 1; y < imgsize-1; ++y){
					double tmp = 0.0;
					for(int j = 0; j < depth_of_kernels; ++j){
						tmp += input[b](j,x,y) * kernel[i](j,0,0); // i-th kernel applied to j-th input is added to i-th output	
					}
					output[b](i,x,y) = tmp;
				}
			}
		}
	}
	if(debug) std::cout<< vol_avg(input) << "\t" << kernel_avg(kernel,output[0].d) << "\t" << vol_avg(output)<< std::endl;
}

void avgpool(ArrOfVols const &input, ArrOfVols &output){ 
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding

	for (int b=0; b<batchsize; ++b){
	    #pragma omp parallel for
		for (int i=0; i<num_of_features; ++i){
			for (int x=1; x<imgsize-1; x+=2){
				for(int y=1; y<imgsize-1; y+=2){
					output[b](i,x/2+1,y/2+1) = 0.25*(input[b](i,x,y) + input[b](i,x+1,y) + input[b](i,x,y+1)+ input[b](i,x+1,y+1));      
				}
			}
		}
	}
	if(debug) std::cout<< vol_avg(input) << "\t" << vol_avg(output)<< std::endl; // for debug
}

void avgpool_backward(ArrOfVols const &input, ArrOfVols &output){ // adds new error to the one found from concat
	int num_of_features = input[0].d;
	int imgsize = input[0].w; // includes padding

	for (int b=0; b<batchsize; ++b){
	    #pragma omp parallel for
		for (int i=0; i<num_of_features; ++i){
			for (int x=1; x<imgsize-1; ++x){
				for(int y=1; y<imgsize-1; ++y){
					output[b](i,2*x-1,2*y-1)+= 0.25*(input[b](i,x,y));
					output[b](i,2*x,2*y-1) 	+= 0.25*(input[b](i,x+1,y));
					output[b](i,2*x-1,2*y) 	+= 0.25*(input[b](i,x,y+1));
					output[b](i,2*x,2*y) 	+= 0.25*(input[b](i,x+1,y+1));      
				}
			}
		}
	}
	if(debug) std::cout<< vol_avg(input) << "\t" << vol_avg(output)<< std::endl;
}

void upconv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output){
	int num_of_kernels = output[0].d; //depth of output or number of kernels should be half of depth of input
	int depth_of_kernels = input[0].d;
	int imgsize = input[0].w; // includes padding
	
	for (int b=0; b<batchsize; ++b){
	    #pragma omp parallel for
		for (int i=0; i<num_of_kernels; ++i){
			for (int x=1; x<output[0].w-1; ++x){
				for (int y=1; y<output[0].w-1; ++y){
					output[b](i,x,y)		= 0.0;
				}
			}
		}
	}
	
	for (int b=0; b<batchsize; ++b){
	    #pragma omp parallel for
		for (int i=0; i<num_of_kernels; ++i){
			for (int x=1; x<imgsize-1; ++x){
				for (int y=1; y<imgsize-1; ++y){
					for (int j=0; j<depth_of_kernels; ++j){
						output[b](i,2*x-1,2*y-1)	+= kernel[i](j,0,0) * input[b](j,x,y);
						output[b](i,2*x,2*y-1)		+= kernel[i](j,1,0) * input[b](j,x,y);
						output[b](i,2*x-1,2*y)		+= kernel[i](j,0,1) * input[b](j,x,y);
						output[b](i,2*x,2*y)		+= kernel[i](j,1,1) * input[b](j,x,y);
					}
				}
			}
		}
	}
	if(debug) std::cout<< vol_avg(input) << "\t" << kernel_avg(kernel,output[0].d) << "\t" << vol_avg(output)<< std::endl;
}

void upconv_backward(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output){
	for(int i=0; i<output[0].d; ++i){
        for(int x=1; x<output[0].w-1; ++x){
            for(int y=1; y<output[0].w-1; ++y){
                for(int b=0; b<batchsize; ++b){
                    for(int j=0; j<input[0].d; ++j){
                        output[b](i,x,y) = 0.0;
                    }
                }
            }
        }
    }
	
    for(int i=0; i<output[0].d; ++i){
        for(int x=1; x<output[0].w-1; ++x){
            for(int y=1; y<output[0].w-1; ++y){
                for(int b=0; b<batchsize; ++b){
                    for(int j=0; j<input[0].d; ++j){
                        output[b](i,x,y) += kernel[i](j,0,0) * input[b](j, 2*x-1, 2*y-1) +
                                           	kernel[i](j,1,0) * input[b](j, 2*x, 2*y-1) +
                                           	kernel[i](j,0,1) * input[b](j, 2*x-1, 2*y) +
                                          	kernel[i](j,1,1) * input[b](j, 2*x, 2*y);
                    }
                }
            }
        }
    }
    if(debug) std::cout<< vol_avg(input) << "\t" << kernel_avg(kernel, output[0].d) << "\t" << vol_avg(output)<< std::endl;
}

void create_all_Aok_backward(ConvStruct *conv_struct, int num_of_convstructs){
	if(verbose) std::cout<<"create_all_Aok_backward"<<std::endl;
	#pragma omp parallel for
	for (int k=0; k<num_of_convstructs; ++k){
		for (int j=0; j<conv_struct[k].in; ++j){ 	// depth of forward kernels aok[0].d
			for (int i=0; i<conv_struct[k].out; ++i){ 	// num of forward kernels 	aok_back[0].d
				for (int x=0; x<conv_struct[k].kernel_size; ++x){
					for (int y=0; y<conv_struct[k].kernel_size; ++y){
						conv_struct[k].Aok_back[j](i, x, y) = conv_struct[k].Aok[i](j, conv_struct[k].kernel_size-1-x, conv_struct[k].kernel_size-1-y);
					}
				}
			}
		}
	}

}

void compute_Aoloss(ArrOfVols &Aoloss, ArrOfVols const &Aof_final, ArrOfVols const &Ao_annots){
	if(verbose) std::cout<<"compute_Aoloss"<<std::endl;
	#pragma omp parallel for
	for (int b=0; b<batchsize; ++b){
		for(int x = 1; x < Aof_final[0].w-1; ++x){
			for(int y = 1; y < Aof_final[0].w-1; ++y){
				double sum = 0.0;
				for(int i = 0; i < 3; ++i){ // (class indices - 1)
					sum += exp(Aof_final[b](i,x,y));
				}
				Aoloss[b](0,x-1,y-1) = -log(exp(Aof_final[b](Ao_annots[b](0,x-1,y-1)-1, x, y))/sum);
			}
		}
	}
	if(debug) std::cout<< vol_avg(Aoloss) << "\t" << vol_avg(Aof_final) << "\t" << vol_avg(Ao_annots)<< std::endl;

}

double avg_batch_loss(const ArrOfVols &Aoloss){
	double sum = 0.0;

	for (int b=0; b<batchsize; ++b){
		for(int x = 0; x < input_imgsize; ++x){
			for(int y = 0; y < input_imgsize; ++y){
				sum += Aoloss[b](0,x,y);
			}
		}
	}
	return sum/(input_imgsize*input_imgsize*batchsize);
}


ArrOfVols create_ArrOfVols(int num_of_arrs, int depth, int width){
	ArrOfVols output(new Volume[num_of_arrs]);
	for (int i=0; i<num_of_arrs; ++i){
		output[i]=Volume(depth, width);
	}
	return output;
}

//each input image with a error tensor will generate several gradient kernels. (the number depends on the depth of the error tensor)
//so for each batch, there will be (batchsize * error tensor depth) gradient kernels. 

void compute_Aok_gradient(ArrOfVols const &input, ArrOfVols const &old_error_tensor, ArrOfVols &Aok_gradient){
    
    for (int n = 0; n < old_error_tensor[0].d; ++n){ // the number of gradient kernels for each input = old_error_tensor[0].d
        for (int d = 0; d < input[0].d; ++d){ // the depth of each gradient kernel = input[0].d
            for (int w = 0; w < Aok_gradient[0].w; ++w){
               	for (int h = 0; h < Aok_gradient[0].w; ++h){ 
                    double tmp = 0;
                    for (int b = 0; b < batchsize; ++b){
                    	#pragma omp parallel for
	                    for (int x = 1; x < old_error_tensor[0].w-1; ++x){
	                        for (int y = 1; y < old_error_tensor[0].w-1; ++y){
	                        	tmp += old_error_tensor[b](n,x,y) * input[b](d,x+w-1,y+h-1); //apply interior of error_tensor to padded input
	                        }
	                    }
	                }
                    Aok_gradient[n](d,w,h) = tmp/batchsize; // this is the "average" Aok gradient for the whole batch
                }
            }
        }
    }
    if(debug) std::cout<< vol_avg(input) << "\t" << vol_avg(old_error_tensor) << "\t" << kernel_avg(Aok_gradient,old_error_tensor[0].d)<< std::endl;

}

void compute_Aok_uc_gradient(ArrOfVols const &input, ArrOfVols const &old_error_tensor, ArrOfVols &Aok_gradient){
    for (int n = 0; n < old_error_tensor[0].d; ++n){ // the number of kernels
        for (int d = 0; d < input[0].d; ++d){ // the depth of each kernel = input[0].d
            double tmp1 = 0;
            double tmp2 = 0;
            double tmp3 = 0;
            double tmp4 = 0;
            for (int b = 0; b < batchsize; ++b){
	            #pragma omp parallel for
                for (int x = 1; x < input[0].w-1; x+=2){
                    for (int y = 1; y < input[0].w-1; y+=2){
                    	tmp1 += old_error_tensor[b](n,2*x-1,2*y-1) * input[b](d,x,y); 
                    	tmp2 += old_error_tensor[b](n,2*x,2*y-1) * input[b](d,x+1,y);
                    	tmp3 += old_error_tensor[b](n,2*x-1,2*y) * input[b](d,x,y+1);
                    	tmp4 += old_error_tensor[b](n,2*x,2*y) * input[b](d,x+1,y+1);
                    }
                }
            }
            Aok_gradient[n](d,0,0) = tmp1/batchsize; // this is the "average" Aok gradient for the whole batch
			Aok_gradient[n](d,1,0) = tmp2/batchsize;
			Aok_gradient[n](d,0,1) = tmp3/batchsize;
			Aok_gradient[n](d,1,1) = tmp4/batchsize;        
        }
    }
    if(debug) std::cout<< vol_avg(input) << "\t" << vol_avg(old_error_tensor) << "\t" << kernel_avg(Aok_gradient,old_error_tensor[0].d)<< std::endl;
}

void create_all_Aok_gradient(ConvStruct **layers, int num_of_layers){
	if(verbose) std::cout<<"create_all_Aok_gradient"<< std::endl;
	compute_Aok_gradient(layers[0][5].Aof, layers[0][6].Aoe, layers[0][5].Aok_gradient);	// first layer has an additional kernel
	for (int i=0; i<=num_of_layers-2; ++i){
		for (int k=0; k<5; ++k){
			compute_Aok_gradient(layers[i][k].Aof, layers[i][k+1].Aoe, layers[i][k].Aok_gradient);
		}
		if(i==num_of_layers-2){break;}
		compute_Aok_uc_gradient(layers[i+1][5].Aof, layers[i][3].Aoe, layers[i+1][5].Aok_gradient);
	}
	// last layer
	compute_Aok_gradient(layers[num_of_layers-1][0].Aof, layers[num_of_layers-1][1].Aoe, layers[num_of_layers-1][0].Aok_gradient);
	compute_Aok_gradient(layers[num_of_layers-1][1].Aof, layers[num_of_layers-1][2].Aoe, layers[num_of_layers-1][1].Aok_gradient);
	compute_Aok_uc_gradient(layers[num_of_layers-1][2].Aof, layers[num_of_layers-2][3].Aoe, layers[num_of_layers-1][2].Aok_gradient);

}

void compute_Aoe_final(ArrOfVols const &Aof_final, ArrOfVols &Aoe_final, const ArrOfVols &Ao_annots){
    for (int b = 0; b < batchsize; ++b){
        #pragma omp parallel for
    	for (int x = 1; x < Aof_final[0].w-1; ++x){
        	for (int y = 1; y < Aof_final[0].w-1; ++y){
                double sum = 0;
                for (int c = 0; c < Aof_final[0].d; ++c){
                    sum += exp(Aof_final[b](c,x,y));
                }
                for (int i=0; i<Aof_final[0].d; ++i){	// i = 0,1,2
                	if (i==Ao_annots[0](0,x-1,y-1)-1){ 	// Ao_annots = 1,2,3
                		Aoe_final[b](i, x, y) = exp(Aof_final[b](i, x, y)) / sum -1;
                	}
                	else{
                		Aoe_final[b](i, x, y) = exp(Aof_final[b](i, x, y)) / sum;
                	}
                }
            }
        }
    }
    if(debug) std::cout<< vol_avg(Aof_final) << "\t" << vol_avg(Aoe_final) << "\t" << vol_avg(Ao_annots)<< std::endl;
}

void update_all_Aok(ConvStruct *conv_struct, int num_of_convstructs){ // update all kernels from their gradient
	if(verbose) std::cout<<"update_all_Aok"<< std::endl;
	double beta1 = 0.9;
	double beta2 = 0.999;
	double alpha = learning_rate;
	double epsilon = 1e-8; //1e-8
	static int timestep=1;

	#pragma omp parallel for
	for (int k=0; k<num_of_convstructs; ++k){
		double before;
		double after;
		if(debug) before=kernel_avg(conv_struct[k].Aok, conv_struct[k].out);
		for (int i=0; i<conv_struct[k].out; ++i){	//num of kernels
			for (int d=0; d<conv_struct[k].in; ++d){		// depth of kernels
				for (int x=0; x<conv_struct[k].kernel_size; ++x){
					for (int y=0; y<conv_struct[k].kernel_size; ++y){
						double grad = conv_struct[k].Aok_gradient[i](d,x,y);
						// SGD =================
						// if (grad > 1){ grad = 1.0;}
						// else if (grad < -1){ grad = -1.0;}
						// conv_struct[k].Aok[i](d,x,y) = conv_struct[k].Aok[i](d,x,y) - learning_rate * grad;
						//=======================

						// Adam ======================
						conv_struct[k].Aom_adam[i](d,x,y) = beta1*conv_struct[k].Aom_adam[i](d,x,y) + (1-beta1)*grad;
						conv_struct[k].Aov_adam[i](d,x,y) = beta2*conv_struct[k].Aov_adam[i](d,x,y) + (1-beta2)*grad*grad;
						double m_hat = conv_struct[k].Aom_adam[i](d,x,y) / (1-pow(beta1, timestep));
						double v_hat = conv_struct[k].Aov_adam[i](d,x,y) / (1-pow(beta2, timestep));
						conv_struct[k].Aok[i](d,x,y) = conv_struct[k].Aok[i](d,x,y) - alpha*m_hat/(sqrt(v_hat)+epsilon);
						// ======================
					}

				}
			}
		}
		if(debug) after=kernel_avg(conv_struct[k].Aok, conv_struct[k].out);
		if(debug) std::cout<<"("<<k<<") "<< before<< " -> "<< after <<std::endl;
	}
	timestep++;
}

void forward_pass(int i, ConvStruct **layers, int num_of_layers){
	
	conv(layers[i][0].Aof, layers[i][0].Aok, layers[i][1].Aof);
	relu(layers[i][1].Aof);

	conv(layers[i][1].Aof, layers[i][1].Aok, layers[i][2].Aof);
	relu(layers[i][2].Aof);

	if(i==num_of_layers-1){
		upconv(layers[i][2].Aof, layers[i][2].Aok, layers[i-1][3].Aof);
		if(verbose) std::cout<<"layer "<< i << " -> " << i-1 <<std::endl;
		return;
	}
	avgpool(layers[i][2].Aof, layers[i+1][0].Aof);
	if(verbose) std::cout<<"layer "<< i << " -> " << i+1 <<std::endl;
	forward_pass(i+1, layers, num_of_layers);

	conv(layers[i][3].Aof, layers[i][3].Aok, layers[i][4].Aof);
	conv(layers[i][2].Aof, layers[i][2].Aok, layers[i][4].Aof, true); // 'true' means just add results, dont wipe previous values
	relu(layers[i][4].Aof);

	conv(layers[i][4].Aof, layers[i][4].Aok, layers[i][5].Aof);
	relu(layers[i][5].Aof);

	if(i==0){
		fullconv(layers[0][5].Aof, layers[0][5].Aok, layers[0][6].Aof); // no relu
	}
	else{
		upconv(layers[i][5].Aof, layers[i][5].Aok, layers[i-1][3].Aof);
		if(verbose) std::cout<<"layer "<< i << " -> " << i-1 <<std::endl;
	}

}

void backward_pass(int i, ConvStruct **layers, int num_of_layers){
	
	if(i==0){
		fullconv(layers[0][6].Aoe, layers[0][5].Aok_back, layers[0][5].Aoe);
    	reluBackwards(layers[0][5].Aof, layers[0][5].Aoe);
	}
    else if(i==num_of_layers-1){
    	upconv_backward(layers[i-1][3].Aoe, layers[i][2].Aok_back, layers[i][2].Aoe);
    	if(verbose) std::cout<<"layer "<< i-1 << " -> " << i <<std::endl;
		conv(layers[i][2].Aoe, layers[i][1].Aok_back, layers[i][1].Aoe);
	    reluBackwards(layers[i][1].Aof, layers[i][1].Aoe);
		conv(layers[i][1].Aoe, layers[i][0].Aok_back, layers[i][0].Aoe);
	    reluBackwards(layers[i][0].Aof, layers[i][0].Aoe);
    	return;
    }
    else{
    	upconv_backward(layers[i-1][3].Aoe, layers[i][5].Aok_back, layers[i][5].Aoe); 
    	if(verbose) std::cout<<"layer "<< i-1 << " -> " << i <<std::endl;
    }

	conv(layers[i][5].Aoe, layers[i][4].Aok_back, layers[i][4].Aoe);
    reluBackwards(layers[i][4].Aof, layers[i][4].Aoe);

	conv(layers[i][4].Aoe, layers[i][3].Aok_back, layers[i][3].Aoe);
    reluBackwards(layers[i][3].Aof, layers[i][3].Aoe);
	conv(layers[i][4].Aoe, layers[i][2].Aok_back, layers[i][2].Aoe);
    reluBackwards(layers[i][2].Aof, layers[i][2].Aoe);

    backward_pass(i+1, layers, num_of_layers);
	avgpool_backward(layers[i+1][0].Aoe, layers[i][2].Aoe); // adds error to the error found from concat (doesn't overwrite)
	if(verbose) std::cout<<"layer "<< i+1 << " -> " << i <<std::endl;

	conv(layers[i][2].Aoe, layers[i][1].Aok_back, layers[i][1].Aoe);
    reluBackwards(layers[i][1].Aof, layers[i][1].Aoe);
	conv(layers[i][1].Aoe, layers[i][0].Aok_back, layers[i][0].Aoe);
    reluBackwards(layers[i][0].Aof, layers[i][0].Aoe);

}

void create_architecture(ConvStruct **layers, ConvStruct *conv_struct, int num_of_layers, int channel_size){

    for (int layer_index=0; layer_index<num_of_layers; ++layer_index){
        int num_of_features = pow(2, channel_size + layer_index);
        int padded_imgsize = input_imgsize/pow(2,layer_index) + 2;

        if (layer_index==0) {
            layers[layer_index] = &conv_struct[0];
            layers[layer_index][0] = ConvStruct(3,                 num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][1] = ConvStruct(num_of_features,   num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][2] = ConvStruct(num_of_features,   num_of_features*2,    batchsize, padded_imgsize, 3);
            layers[layer_index][3] = ConvStruct(num_of_features,   num_of_features*2,    batchsize, padded_imgsize, 3);
            layers[layer_index][4] = ConvStruct(num_of_features*2, num_of_features*2,    batchsize, padded_imgsize, 3);
            layers[layer_index][5] = ConvStruct(num_of_features*2, 3,                    batchsize, padded_imgsize, 1);
            layers[layer_index][6] = ConvStruct(3,                 0,                    batchsize, padded_imgsize, 0);

        }
        else if (layer_index==num_of_layers-1){
            layers[layer_index] = &conv_struct[layer_index*6 + 1];
            layers[layer_index][0] = ConvStruct(num_of_features/2, num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][1] = ConvStruct(num_of_features,   num_of_features/2,    batchsize, padded_imgsize, 3);
            layers[layer_index][2] = ConvStruct(num_of_features/2, num_of_features/2,    batchsize, padded_imgsize, 2);

        }
        else {
            layers[layer_index] = &conv_struct[layer_index*6 + 1];
            layers[layer_index][0] = ConvStruct(num_of_features/2, num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][1] = ConvStruct(num_of_features,   num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][2] = ConvStruct(num_of_features,   num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][3] = ConvStruct(num_of_features,   num_of_features,      batchsize, padded_imgsize, 3);
            layers[layer_index][4] = ConvStruct(num_of_features,   num_of_features/2,    batchsize, padded_imgsize, 3);
            layers[layer_index][5] = ConvStruct(num_of_features/2, num_of_features/2,    batchsize, padded_imgsize, 2);

        }
    }
}

//=====================================================================================
//=====================================================================================
// The rest of functions below are only for debugging
//=====================================================================================
//=====================================================================================

void minmax_batch_loss(const ArrOfVols &Aoloss, double &min, double &max){ // for debug
	max = min = Aoloss[0](0,0,0);

	for (int b=0; b<batchsize; ++b){
        #pragma omp parallel for
		for(int x = 0; x < input_imgsize; ++x){
			for(int y = 0; y < input_imgsize; ++y){
				if (Aoloss[b](0,x,y)>max){max = Aoloss[b](0,x,y);}
				if (Aoloss[b](0,x,y)<min){min = Aoloss[b](0,x,y);}
			}
		}
	}
}

void compute_segmap(ArrOfVols const &Aof_final, ArrOfVols &Ao_segmap){ // segmentation map not padded (for debug)
	for (int b=0; b<batchsize; ++b){
		for (int x=1; x<Aof_final[0].w-1; ++x){
			for (int y=1; y<Aof_final[0].w-1; ++y){
				if(Aof_final[b](0,x,y)>Aof_final[b](1,x,y) && Aof_final[b](0,x,y)>Aof_final[b](2,x,y)){
					Ao_segmap[b](0,x-1,y-1)=80;
				}
				else if(Aof_final[b](1,x,y)>Aof_final[b](0,x,y) && Aof_final[b](1,x,y)>Aof_final[b](2,x,y)){
					Ao_segmap[b](0,x-1,y-1)=160;
				}
				else if(Aof_final[b](2,x,y)>Aof_final[b](0,x,y) && Aof_final[b](2,x,y)>Aof_final[b](1,x,y)){
					Ao_segmap[b](0,x-1,y-1)=240;
				}
				else {
					Ao_segmap[b](0,x-1,y-1)=0;
				}
			}
		}
	}
}

void init_kernel_guess(ConvStruct *conv_struct, int num_of_convstructs, double value){ 	// for debug (init kernels with random numbers)
	const int range_from  = -1;
    const int range_to    = 1;
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_int_distribution<int>  distr(range_from, range_to);
	for (int k=0; k<num_of_convstructs; ++k){
		for (int i=0; i<conv_struct[k].out; ++i){	//num of kernels
			for (int d=0; d<conv_struct[k].in; ++d){		// depth of kernels
				for (int x=0; x<conv_struct[k].kernel_size; ++x){
					for (int y=0; y<conv_struct[k].kernel_size; ++y){
						// conv_struct[k].Aok[i](d,x,y)=distr(generator)/6.0f;
						conv_struct[k].Aok[i](d,x,y)=value*(k+1)*(x+ y + 1/(i+1) + 1/(d+1));
						if(conv_struct[k].Aok[i](d,x,y)>1 || conv_struct[k].Aok[i](d,x,y)<-1){
							std::cout<< "bad weight"<<std::endl;
							exit(1);}
					}

				}
			}
		}
	}
}

void print_arr(const ArrOfVols &arr, int index, int depth, int *range_x, int *range_y, std::string filename){ // for debug (prints any ArrOfVols)
	std::ofstream outfile;
	outfile.open(filename);
	for (int y=range_y[0]; y<range_y[1] ; ++y){
		for (int x=range_x[0]; x<range_x[1] ; ++x){
			outfile<<arr[index](depth,x,y)<< "\t";
		}
		outfile<<std::endl;
	}
	outfile.close();
}

double vol_avg(const ArrOfVols &arr){		// for debug
	double avg;
    for (int b = 0; b < batchsize; ++b){
    	for (int i=0; i<arr[0].d; ++i){
	    	for (int x = 0; x < arr[0].w; ++x){
	        	for (int y = 0; y < arr[0].w; ++y){
        			avg += arr[b](i, x, y);
        		}
        	}
        }
    }
    return avg/(batchsize*(arr[0].w)*(arr[0].w)*(arr[0].d));
}

double kernel_avg(const ArrOfVols &arr, int num_of_kernels){		// for debug
	double avg;
    for (int n = 0; n < num_of_kernels; ++n){
    	for (int i=0; i<arr[0].d; ++i){
	    	for (int x = 0; x < arr[0].w; ++x){
	        	for (int y = 0; y < arr[0].w; ++y){
        			avg += arr[n](i, x, y);
        		}
        	}
        }
    }
    return avg/(num_of_kernels*(arr[0].w)*(arr[0].w)*(arr[0].d));
}

void read_img_text(ArrOfVols &arr, int batchNr){	// read imges from .csv (if CImg does not work)
	for(int b=0; b<batchsize; ++b){
		int img_num = batchsize*batchNr + b;
		std::ifstream ch0("training_data/img_csv/img_"+std::to_string(img_num)+"_ch0.csv");
		std::ifstream ch1("training_data/img_csv/img_"+std::to_string(img_num)+"_ch1.csv");
		std::ifstream ch2("training_data/img_csv/img_"+std::to_string(img_num)+"_ch2.csv");
		if(!ch0.is_open() || !ch1.is_open() || !ch2.is_open() ){
			std::cout<<"invalid img"<<std::endl;
			exit(1);
		}
		for (int y=0; y<input_imgsize; ++y){
			std::string l0;
			std::string l1;
			std::string l2;
			getline(ch0, l0);
			getline(ch1, l1);
			getline(ch2, l2);
			std::istringstream line0(l0);
			std::istringstream line1(l1);
			std::istringstream line2(l2);
			for (int x=0; x<input_imgsize; ++x){
				line0 >> arr[b](0,input_imgsize-y,x+1);
				line1 >> arr[b](1,input_imgsize-y,x+1);
				line2 >> arr[b](2,input_imgsize-y,x+1);
			}
		}
	}
}

void read_annot_text(ArrOfVols &arr, int batchNr){ // for debug (if CImg does not work)
	for (int b=0; b<batchsize; ++b){
		int img_num = batchsize*batchNr +b;
		std::ifstream ch0("training_data/annot_csv/annot_"+std::to_string(img_num)+".csv");
		if(!ch0.is_open()){
			std::cout<<"invalid annot"<<std::endl;
			exit(1);
		}
		for (int y=0; y<input_imgsize; ++y){
			std::string l0;
			getline(ch0, l0);
			std::istringstream line0(l0);
			for (int x=0; x<input_imgsize; ++x){
				line0 >> arr[b](0,input_imgsize-y-1,x);
			}
		}
	}
}

void backup_kernels(ConvStruct *conv_struct, int num_of_convstructs){ // for debug
	for (int k=0; k<num_of_convstructs; ++k){
		for (int i=0; i<conv_struct[k].out; ++i){
			for (int j=0; j<conv_struct[k].in; ++j){
				std::ofstream outfile("backup_kernel_csv/Convstruct_"+std::to_string(k)+"_"+std::to_string(i)+"_"+std::to_string(j)+".csv");
				if(!outfile.is_open()){std::cout<<"couldnt open backup kernel outfile"<<std::endl; exit(1);}
				for (int y=0; y<conv_struct[k].kernel_size; ++y){
					for (int x=0; x<conv_struct[k].kernel_size; ++x){
						outfile << conv_struct[k].Aok[i](j,x,y) << "\t";
					}
					outfile<<std::endl;
				}
			}
		}
	}
}

void read_backup_kernels(ConvStruct *conv_struct, int num_of_convstructs){ // for debug
	for (int k=0; k<num_of_convstructs; ++k){
		for (int i=0; i<conv_struct[k].out; ++i){
			for (int j=0; j<conv_struct[k].in; ++j){
				std::ifstream infile("backup_kernel_csv/Convstruct_"+std::to_string(k)+"_"+std::to_string(i)+"_"+std::to_string(j)+".csv");
				if(!infile.is_open()){std::cout<<"couldnt open backup kernel infile"<<std::endl; exit(1);}
				std::string line;
				for (int y=0; y<conv_struct[k].kernel_size; ++y){
					getline(infile, line);
					std::istringstream inputline(line);
					for (int x=0; x<conv_struct[k].kernel_size; ++x){
						inputline >> conv_struct[k].Aok[i](j,x,y) ;
					}
				}
			}
		}
	}	
}
