#ifndef UNETFUNCS_H
#define UNETFUNCS_H
#include <cmath>
#include <iostream>
#include <omp.h>
#include "ConvStruct.h"
#include "Volume.h"
#include <random>
#include <fstream>
#include <sstream>
#include <string>

extern int batchsize;
extern int input_imgsize;
extern double learning_rate;
extern bool verbose;
extern bool debug;

using ArrOfVols = ConvStruct::ArrOfVols;

void reluBackwards(ArrOfVols &image, ArrOfVols &error);
void relu(ArrOfVols &input);
void conv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output, bool dont_wipe_before_adding=false);
void fullconv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output);
void avgpool(ArrOfVols const &input, ArrOfVols &output);
void avgpool_backward(ArrOfVols const &input, ArrOfVols &output);
void upconv(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output);
void upconv_backward(ArrOfVols const &input, ArrOfVols const &kernel, ArrOfVols &output);
void create_all_Aok_backward(ConvStruct *conv_struct, int num_of_convstructs);
void compute_Aoloss(ArrOfVols &Aoloss, ArrOfVols const &Aof_final, ArrOfVols const &Ao_annots);
double avg_batch_loss(const ArrOfVols &Aoloss);
void minmax_batch_loss(const ArrOfVols &Aoloss, double &min, double &max);
ArrOfVols create_ArrOfVols(int num_of_arrs, int depth, int width);
void compute_Aok_gradient(ArrOfVols const &input, ArrOfVols const &old_error_tensor, ArrOfVols &Aok_gradient);
void compute_Aok_uc_gradient(ArrOfVols const &input, ArrOfVols const &old_error_tensor, ArrOfVols &Aok_gradient);
void create_all_Aok_gradient(ConvStruct **layers, int num_of_layers);
void compute_Aoe_final(ArrOfVols const &Aof_final, ArrOfVols &Aoe_final, const ArrOfVols &Ao_annots);
void update_all_Aok(ConvStruct *conv_struct, int num_of_convstructs);
void forward_pass(int i, ConvStruct **layers, int num_of_layers);
void backward_pass(int i, ConvStruct **layers, int num_of_layers);
void create_architecture(ConvStruct **layers, ConvStruct *conv_struct, int num_of_layers, int channel_size);
void compute_segmap(ArrOfVols const &Aof_final, ArrOfVols &Ao_segmap);
void init_kernel_guess(ConvStruct *conv_struct, int num_of_convstructs, double value);
void print_arr(const ArrOfVols &arr, int index, int depth, int *range_x, int *range_y, std::string filename);
double vol_avg(const ArrOfVols &arr);
double kernel_avg(const ArrOfVols &arr, int num_of_kernels);
void read_img_text(ArrOfVols &arr, int batchNr);
void read_annot_text(ArrOfVols &arr, int batchNr);
void backup_kernels(ConvStruct *conv_struct, int num_of_convstructs);
void read_backup_kernels(ConvStruct *conv_struct, int num_of_convstructs);

#endif
