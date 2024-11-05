#include "CImg.h"
#include "ConvStruct.h"
#include "unetFuncs.h"
#include <iostream>
#include <sstream>
#include <cmath>

void ReadImages(ConvStruct::ArrOfVols &output, int batchNr, int batchSize, int input_imgsize);
void ReadAnnot(ConvStruct::ArrOfVols &output, int batchNr, int batchSize, int input_imgsize);
void displayImage(ConvStruct::ArrOfVols &output, int b, int c, int input_imgsize);
void convert_all_imgs_to_csv();