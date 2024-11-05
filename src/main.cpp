#include <iostream>
#include "unetFuncs.h"
#include "processImages.h"
#include "initWeights.h"
#include "ConvStruct.h"


int main(int argc, char *argv[]){
    verbose=false;
    debug=false;
    learning_rate = 0.001; // this is alpha in the adam optimiser

    if(argc<4){std::cout<<"Usage: [imgsize] [batchsize] [num_of_batches]"<<std::endl; return 0;}
    input_imgsize = std::stoi(argv[1]);      // should be 512
    batchsize  = std::stoi(argv[2]);         // should be 8
    int num_of_batches = std::stoi(argv[3]); // should be 128/batchsize
    int epochs = 100;      
    int num_of_layers = 5;  // At least must be 2
    int channel_size = 2;
    bool write_segmap = false;
    std::cout<< "\nimgsize: "<< input_imgsize<< "\t batchsize: "<< batchsize<< "\t num_of_batches: "<< num_of_batches<< std::endl;
    std::cout<< "epochs: "<< epochs<< "\t num_of_layers: "<< num_of_layers<<"\t channel_size: "<< channel_size<< "\t learning_rate: "<< learning_rate<< std::endl;

    int num_of_convstructs = 7 + (num_of_layers-2)*6 + 3; // 7 for first layer, 6 for other layers, 3 for last layer
    ConvStruct *conv_struct = new ConvStruct[num_of_convstructs];
    ConvStruct *layers[num_of_layers];

    create_architecture(layers, conv_struct, num_of_layers, channel_size);
    ConvStruct::ArrOfVols Ao_annots(create_ArrOfVols(batchsize, 1, input_imgsize)); // array of annotations. only 1 feature channel, so depth is 1
    ConvStruct::ArrOfVols Aoloss(create_ArrOfVols(batchsize, 1, input_imgsize)); // array of loss values
    ConvStruct::ArrOfVols Ao_segmap(create_ArrOfVols(batchsize, 1, input_imgsize)); // array of segmentation maps
  
    if(verbose) std::cout<<"init kernels"<<std::endl;
    init_kernels(layers);

    for (int i=0; i<epochs; ++i){ 
        if(verbose) std::cout<<"\n========================================================="<< std::endl;
        if(verbose) std::cout<<"{{{{{{{{{{{{{{{{{{{{{{ epoch ("<< i << ") }}}}}}}}}}}}}}}}}}}}}}" << std::endl;
        if(verbose) std::cout<<"=========================================================\n"<< std::endl;
        double loss_sum=0.0;
        for (int batchNr=0; batchNr<num_of_batches; ++batchNr){
            if(verbose) std::cout<<"\n---------------------------------------------------------------"<< std::endl;
            if(verbose) std::cout<< "{{ batchNr = " << batchNr << " }}"<< std::endl;
            if(verbose) std::cout<<"---------------------------------------------------------------\n"<< std::endl;

            ReadImages(layers[0][0].Aof, batchNr, batchsize, input_imgsize);
            ReadAnnot(Ao_annots, batchNr, batchsize, input_imgsize);
            // read_img_text(layers[0][0].Aof, batchNr);    // for use on clusters (if images cannot be read with CImg)
            // read_annot_text(Ao_annots, batchNr);         // for use on clusters (if images cannot be read with CImg)
            if(verbose) std::cout<<"Forward Pass"<< std::endl;
            forward_pass(0, layers, num_of_layers);
            if(write_segmap) compute_segmap(layers[0][6].Aof, Ao_segmap);
            compute_Aoloss(Aoloss, layers[0][6].Aof, Ao_annots); // compute loss from annots
            loss_sum += avg_batch_loss(Aoloss);
            create_all_Aok_backward(conv_struct, num_of_convstructs);	//create all conv kernles for finding error tensors in the backward pass
            compute_Aoe_final(layers[0][6].Aof, layers[0][6].Aoe, Ao_annots); // find gradient of loss wrt final image to get the final error tensor
            if(verbose) std::cout<<"Backward Pass"<< std::endl;
            backward_pass(0, layers, num_of_layers);
            create_all_Aok_gradient(layers, num_of_layers);
            update_all_Aok(conv_struct, num_of_convstructs); // update all kernels from thier gradients
            if(verbose) std::cout<< "loss: " << avg_batch_loss(Aoloss) << std::endl;
            if(verbose) std::cout<< "{ batch ("<< batchNr<< ") avg_loss = "<< loss_sum<< " }" <<std::endl;
            
        }
        std::cout<<"---\n{ Loss (" << i<< ") = "<< loss_sum/num_of_batches <<" }\n---"<< std::endl;
        if(write_segmap){
            int range_x[2]={0, Ao_segmap[0].w};
            int range_y[2]={0, Ao_segmap[0].w};
            print_arr(Ao_segmap, 0, 0, range_x, range_y, "Ao_segmap_ep"+std::to_string(i)+".csv");
        }
            
    }


    return 0;
}
