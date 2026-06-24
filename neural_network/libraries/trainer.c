#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "dataset.h"
#include "tensor.h"
#include "trainer.h"
#include "layer.h"


void Trainer_train(Network* net, const char* dataset_path, Hyperparameters* params, Progress* progress, int last_dataset_point) {
    printf("\n==================================================\n");
    printf("[TRAINER DEBUG] Entering Trainer_train\n");
    printf("[TRAINER DEBUG] Dataset path: %s\n", dataset_path);
    printf("[TRAINER DEBUG] Batch size configured: %d\n", params->batch_size);
    printf("[TRAINER DEBUG] Network Layers Count: %d\n", net ? net->num_layers : -1);
    printf("==================================================\n");

    int save_count = 0;

    FILE* header_file = fopen(dataset_path, "rb");
    if (!header_file) {
        printf("ERROR: Could not open dataset file for header parsing: %s\n", dataset_path);
        return;
    }
    uint32_t magic, num_samples, channels, height, width;
    (void)fread(&magic, sizeof(uint32_t), 1, header_file);
    (void)fread(&num_samples, sizeof(uint32_t), 1, header_file);
    (void)fread(&channels, sizeof(uint32_t), 1, header_file);
    (void)fread(&height, sizeof(uint32_t), 1, header_file);
    (void)fread(&width, sizeof(uint32_t), 1, header_file);
    fclose(header_file);

    num_samples = flip_endian(num_samples);
    channels = flip_endian(channels);
    height = flip_endian(height);
    width = flip_endian(width);

    
    if (num_samples > 400) {
        printf("[TRAINER OVERRIDE] Capping total samples from %u down to 400 for testing.\n", num_samples);
        num_samples = 400;
    }


    printf("[TRAINER DEBUG] File Headers -> Samples: %u, Channels: %u, HxW: %ux%u\n",
           num_samples, channels, height, width);


    int train_samples = (int)(num_samples * params->training_ratio);
    int val_samples = num_samples - train_samples;
    
    // Determine the sample index offsets for training and validation segments
    int train_start_sample = 0;
    int val_start_sample = train_samples;

    printf("[TRAINER SPLIT] Total Samples: %u | Train Samples: %d (Starts at %d) | Val Samples: %d (Starts at %d)\n",
           num_samples, train_samples, train_start_sample, val_samples, val_start_sample);





    int total_features = channels * height * width;

    if (!net || net->num_layers == 0) {
        printf("[TRAINER ERROR] Network is NULL or empty!\n");
        return;
    }

    NetworkLayer* final_net_layer = &(net->layers[net->num_layers - 1]);
    Layer* final_layer = (Layer*)final_net_layer->layer_data;
    if (!final_layer) {
        printf("[TRAINER ERROR] Final layer data payload is missing!\n");
        return;
    }
    int output_dim = final_layer->output_dim;
    printf("[TRAINER DEBUG] Calculated Output Dim: %d\n", output_dim);

    printf("[TRAINER DEBUG] Allocating tensor workspaces...\n");
    int* x_batch_shape = malloc(4 * sizeof(int));
    x_batch_shape[0] = params->batch_size;
    x_batch_shape[1] = channels;
    x_batch_shape[2] = height;
    x_batch_shape[3] = width;

    int* y_batch_shape = malloc(2 * sizeof(int));
    y_batch_shape[0] = params->batch_size;
    y_batch_shape[1] = output_dim;

    printf("[TRAINER DEBUG] Creating x_work (4D)...\n");
    Tensor* x_work = Tensor_make(4, x_batch_shape);
    printf("[TRAINER DEBUG] Creating y_work (2D)...\n");
    Tensor* y_work = Tensor_make(2, y_batch_shape);
    printf("[TRAINER DEBUG] Wrapping into Dataset structure...\n");
    Dataset* batch = Dataset_make(x_work, y_work);

    int grad_shape[2] = {params->batch_size, output_dim};
    printf("[TRAINER DEBUG] Creating loss_grad_work...\n");
    Tensor* loss_grad_work = Tensor_make(2, grad_shape);

    printf("[TRAINER DEBUG] Creating dummy_input_grad (4D)...\n");
    Tensor* dummy_input_grad = Tensor_make(4, x_batch_shape);
    Tensor* predictions = NULL;
    float avg_loss;

    int total_batches = (train_samples + params->batch_size - 1) / params->batch_size;
    printf("[TRAINER DEBUG] Total calculation runs planned: %d batches per epoch\n", total_batches);

    for(int i = progress->current_epoch; i < params->max_epochs; i++){
        progress->current_epoch = i;
        float epoch_loss_sum = 0.0f;
        int batch_count = 0;

        printf("\n--- Starting Epoch %d ---\n", i + 1);

        for(int b = 0; b < total_batches; b++){
            printf("[BATCH %d/%d] Reading binary raw batch memory data stream...\n", b, total_batches);
            int samples_read = Dataset_read_batch(dataset_path, b, params->batch_size, batch);
            printf("[BATCH %d/%d] Raw bytes translated. Samples returned: %d\n", b, total_batches, samples_read);

            if (samples_read <= 0) {
                printf("[BATCH %d/%d] Empty execution space block. Skipping iteration step.\n", b);
                continue;
            }

            // Adjust dynamically tracked dimension layouts
            batch->x->shape[0] = samples_read;
            batch->y->shape[0] = samples_read;
            loss_grad_work->shape[0] = samples_read;
            dummy_input_grad->shape[0] = samples_read;

            //printf("[BATCH %d/%d] Executing graph Network_forward pass...\n", b);
            predictions = Network_forward(net, batch->x);
            //printf("[BATCH %d/%d] Forward pass completed. Predictions pointer validation: %p\n", b, (void*)predictions);

            //printf("[BATCH %d/%d] Computing categorical/loss properties error margins...\n", b);
            float current_loss = Network_compute_loss(net, predictions, batch->y, params->losstype, loss_grad_work);
            epoch_loss_sum += current_loss;
            printf("[BATCH %d/%d] Segment evaluation complete. Segment loss value: %.4f\n", b, current_loss);

            //printf("[BATCH %d/%d] Executing backward graph validation auto-diff matrix trace...\n", b);
            Network_backward(net, loss_grad_work, dummy_input_grad);
            //printf("[BATCH %d/%d] Error derivatives propagated cleanly.\n", b);

            //printf("[BATCH %d/%d] Updating stochastic optimization structural weights...\n", b);
            Network_update(net, params->learning_rate);
            //printf("[BATCH %d/%d] Optimization cycle iteration completed successfully.\n", b);

            batch_count++;
        }

        printf("[TRAINER DEBUG] Finalizing batch sequence steps. Calling validation tracking modules...\n");
        avg_loss = Trainer_evaluate(net, dataset_path, val_start_sample, val_samples, *params);
        printf("[TRAINER DEBUG] Validation tracking completed. Validation score returned: %.4f\n", avg_loss);

        // Early stopping metrics evaluation
        float min_range = progress->best_validation_loss - params->early_stop_delta;
        float max_range = progress->best_validation_loss + params->early_stop_delta;

        if(avg_loss >= min_range && avg_loss <= max_range){
            progress->unchanged_counter++;
        }
        if(avg_loss < progress->best_validation_loss){
            progress->best_validation_loss = avg_loss;
            progress->unchanged_counter = 0;
        }

        if(progress->unchanged_counter == params->early_stop){
            printf("early stop due to little change in validation\n");
            break;
        }

        // Periodic saves
        if(save_count == params->save_frequency){
            printf("[TRAINER DEBUG] Reached file checkpoint flag limits. Serializing structure to storage disk...\n");
            Trainer_save_model(net, params, progress);
            save_count = 0;
        }else{
            save_count++;
        }

        float display_loss = 0.0f;
        if (batch_count > 0) {
            display_loss = epoch_loss_sum / batch_count;
        } else {
            printf("[WARNING] batch_count is 0! Check dataset file layout.\n");
        }

        printf("Epoch %d | Train Loss: %.4f | Val Loss: %.4f\n", i + 1, display_loss, avg_loss);
    }

    printf("finished training best validation was: %f", progress->best_validation_loss);
    Trainer_save_model(net, params, progress);

    printf("[TRAINER DEBUG] Freeing memory allocations...\n");
    Tensor_free(dummy_input_grad);
    Tensor_free(loss_grad_work);
    Dataset_free(batch);

    free(x_batch_shape);
    free(y_batch_shape);
    printf("[TRAINER DEBUG] Exiting Trainer_train cleanly.\n");
}




/*
void Trainer_save_model(Network* net, Hyperparameters* params, Progress* progress){

    char temp_location[512];
    snprintf(temp_location, sizeof(temp_location), "%s.tmp", params->save_location);

    FILE* file = fopen(temp_location, "wb");

    if(!file){
        printf("failed to open file in trainer save model\n");
        return;
    }
    
    char id[] = {"NN"};
 
    fwrite(id, sizeof(char), strlen(id), file);

    //progess tabs
    fwrite(progress, sizeof(Progress), 1, file);
    
    //num layers in the network
    fwrite(&(net->num_layers), sizeof(int), 1, file);
    
    for(int i = 0; i < net->num_layers; i++){

        NetworkLayer* net_layer = &(net->layers[i]);
        Layer* layer = (Layer*)net_layer->layer_data;


        //layer type is in network->layers emnum with LayerType and data
        fwrite(&(net_layer->type), sizeof(LayerType), 1, file);
        
        //activation type
        fwrite(&(layer->activation), sizeof(ActivationType), 1, file);
              
        int has_weights = (layer->weights != NULL) ? 1 : 0;
        fwrite(&has_weights, sizeof(int), 1, file); // Write a flag to the file for loader safety

        if (has_weights) {
            // weights
            fwrite(&(layer->weights->ndim), sizeof(int), 1, file);
            fwrite(layer->weights->shape, sizeof(int), (layer->weights->ndim), file);
            fwrite(layer->weights->data, sizeof(float), layer->weights->total_elements, file);

            // biases
            fwrite(&(layer->biases->ndim), sizeof(int), 1, file);
            fwrite(layer->biases->shape, sizeof(int), (layer->biases->ndim), file);        
            fwrite(layer->biases->data, sizeof(float), layer->biases->total_elements, file);
        }
   
    }

    if(ferror(file)){
        printf("ERROR: OS error occured in saving model to file\n");
        fclose(file);
        remove(temp_location);
        return;
    }else{
         printf("model succesfully saved to temp\n");
    }


    fclose(file);

    if(rename(temp_location, params->save_location) != 0){
        printf("ERROR: Failed to make the temp location the final location\n");
        remove(temp_location);
        return;
    }

    printf("Model fully saved to %s\n", params->save_location);
}
*/
  

void Trainer_save_model(Network* net, Hyperparameters* params, Progress* progress){
    printf("\n==================================================\n");
    printf("[SERIALIZATION DEBUG] Entering Trainer_save_model\n");
    printf("[SERIALIZATION DEBUG] Target Save Path: %s\n", params->save_location);
    printf("==================================================\n");

    char temp_location[512];
    snprintf(temp_location, sizeof(temp_location), "%s.tmp", params->save_location);

    printf("[SERIALIZATION DEBUG] Opening temporary file: %s\n", temp_location);
    FILE* file = fopen(temp_location, "wb");

    if(!file){
        printf("[SERIALIZATION CRITICAL] Failed to open file path: %s\n", temp_location);
        return;
    }

    char id[] = {"NN"};
    printf("[SERIALIZATION DEBUG] Writing network identifier cookie: '%s'\n", id);
    fwrite(id, sizeof(char), strlen(id), file);

    printf("[SERIALIZATION DEBUG] Archiving progress state (Epoch: %d)\n", progress->current_epoch);
    fwrite(progress, sizeof(Progress), 1, file);

    printf("[SERIALIZATION DEBUG] Archiving total network layer count: %d\n", net->num_layers);
    fwrite(&(net->num_layers), sizeof(int), 1, file);

    for(int i = 0; i < net->num_layers; i++){
        NetworkLayer* net_layer = &(net->layers[i]);
        Layer* layer = (Layer*)net_layer->layer_data;

        printf("\n  --> [LAYER %d/%d] Processing structural block...\n", i, net->num_layers - 1);
        printf("      [LAYER %d] Internal Type Token: %d | Activation Token: %d\n", i, net_layer->type, layer->activation);

        // Save metadata headers
        fwrite(&(net_layer->type), sizeof(LayerType), 1, file);
        fwrite(&(layer->activation), sizeof(ActivationType), 1, file);

        // Evaluate learnable parameters state
        int has_weights = (layer->weights != NULL) ? 1 : 0;
        printf("      [LAYER %d] Learnable parameters flag evaluation: %s\n", i, has_weights ? "TRUE (Has Weights/Biases)" : "FALSE (No Weights/Biases)");
        fwrite(&has_weights, sizeof(int), 1, file);

        if (has_weights) {
            // --- WEIGHTS ---
            printf("      [LAYER %d] Serializing weights tensor properties...\n", i);
            printf("          | Dimensions (ndim): %d\n", layer->weights->ndim);
            printf("          | Total elements stream: %d\n", layer->weights->total_elements);

            fwrite(&(layer->weights->ndim), sizeof(int), 1, file);
            fwrite(layer->weights->shape, sizeof(int), (layer->weights->ndim), file);

            printf("          | Flushing weights data payload array to disk memory block...\n");
            fwrite(layer->weights->data, sizeof(float), layer->weights->total_elements, file);

            // --- BIASES ---
            printf("      [LAYER %d] Serializing biases tensor properties...\n", i);
            printf("          | Dimensions (ndim): %d\n", layer->biases->ndim);
            printf("          | Total elements stream: %d\n", layer->biases->total_elements);

            fwrite(&(layer->biases->ndim), sizeof(int), 1, file);
            fwrite(layer->biases->shape, sizeof(int), (layer->biases->ndim), file);

            printf("          | Flushing biases data payload array to disk memory block...\n");
            fwrite(layer->biases->data, sizeof(float), layer->biases->total_elements, file);
        } else {
            printf("      [LAYER %d] Skipping tensor payload step (Structural/Pooling layer design)\n", i);
        }
    }

    printf("\n[SERIALIZATION DEBUG] Checking file stream integrity flags...\n");
    if(ferror(file)){
        printf("[SERIALIZATION CRITICAL] OS I/O error occurred during serialization layout pass!\n");
        fclose(file);
        remove(temp_location);
        return;
    } else {
         printf("[SERIALIZATION DEBUG] Layout pass completed smoothly. Closing temporary file context.\n");
    }
    fclose(file);

    printf("[SERIALIZATION DEBUG] Swapping temporary file into final location placement...\n");
    if(rename(temp_location, params->save_location) != 0){
        printf("[SERIALIZATION CRITICAL] File rename step failed! check disk write permissions.\n");
        remove(temp_location);
        return;
    }

    printf("[SERIALIZATION DEBUG] Model fully serialized and saved to: %s\n", params->save_location);
    printf("==================================================\n\n");
}




void Trainer_load_model(Network* net, Hyperparameters* params, Progress* progress){
    FILE* file = fopen(params->save_location, "rb");
    if (!file) {
        printf("ERROR: Could not open file for loading: %s\n", params->save_location);
        return;
    }

    char header[2];
    (void)fread(header, sizeof(char), 2, file);
    if(strncmp(header, "NN", 2) != 0){
        printf("ERROR: header not found at the top of file\n");
        fclose(file);
        return;
    }

    (void)fread(progress, sizeof(Progress), 1, file);
    
    int num_layers;
    (void)fread(&num_layers, sizeof(int), 1, file);

    // OPTION A: Network structure already made 
    if(net && net->num_layers > 0){
        if(net->num_layers != num_layers){
            printf("ERROR: inputted layers count does not match loaded layers\n");
            fclose(file);
            return;
        }

        for(int i = 0; i < num_layers; i++){
            NetworkLayer* net_layer = &(net->layers[i]);
            Layer* layer = (Layer*)net_layer->layer_data;

            LayerType saved_type;
            (void)fread(&saved_type, sizeof(LayerType), 1, file);
            
            if (saved_type == LAYER_ADD || saved_type == LAYER_UPSAMPLE) {
                continue; 
            }

            (void)fread(&(layer->activation), sizeof(ActivationType), 1, file);
            
            // Validate and sync Weights
            int w_ndim;
            (void)fread(&w_ndim, sizeof(int), 1, file);
            if(layer->weights->ndim != w_ndim){
                printf("ERROR: Layer %d weights dimensions mismatch\n", i);
                fclose(file);
                return;
            }

            int* temp_w = malloc(w_ndim * sizeof(int));
            (void)fread(temp_w, sizeof(int), w_ndim, file);
            for(int j = 0; j < w_ndim; j++){
                 if(layer->weights->shape[j] != temp_w[j]){
                    printf("ERROR: Layer %d weights shape mismatch\n", i);
                    free(temp_w);
                    fclose(file);
                    return;
                 }
            }
            free(temp_w);
            (void)fread(layer->weights->data, sizeof(float), layer->weights->total_elements, file);

            // Validate and sync Biases
            int b_ndim;
            (void)fread(&b_ndim, sizeof(int), 1, file);
            if(layer->biases->ndim != b_ndim){
                printf("ERROR: Layer %d biases dimensions mismatch\n", i);
                fclose(file);
                return;
            }

            int* temp_b = malloc(b_ndim * sizeof(int));
            (void)fread(temp_b, sizeof(int), b_ndim, file); 
            for(int j = 0; j < b_ndim; j++){
                 if(layer->biases->shape[j] != temp_b[j]){
                    printf("ERROR: Layer %d biases shape mismatch\n", i);
                    free(temp_b);
                    fclose(file);
                    return;
                 }
            }
            free(temp_b);
            (void)fread(layer->biases->data, sizeof(float), layer->biases->total_elements, file);
        }
    }
    

    // OPTION B: Building network using pass-in hparams
    else
    {
        if(!net){
            printf("ERROR: must have a valid pointer to network\n");
            fclose(file);
            return;
        }

        net->num_layers = 0;
        net->layers = NULL;

        int current_height = params->input_height;
        int current_width = params->input_width;

        for(int i = 0; i < num_layers; i++){
            LayerType type;
            ActivationType activation;
            
            (void)fread(&type, sizeof(LayerType), 1, file);

            if (type == LAYER_ADD) {
                Network_add_addition(net, 0); 
                continue;
            }
            if (type == LAYER_UPSAMPLE) {
                Network_add_upsample(net, 2); 
                continue;
            }

           (void)fread(&activation, sizeof(ActivationType), 1, file);

            int w_dim = 0;
            (void)fread(&w_dim, sizeof(int), 1, file);
            int* w_shape = malloc(w_dim * sizeof(int));
            (void)fread(w_shape, sizeof(int), w_dim, file);
            
            Layer* new_layer = NULL;

            if (type == LAYER_DENSE) {
                int input_dim = w_shape[0];
                int output_dim = w_shape[1];
                new_layer = Layer_make_dense(input_dim, output_dim, activation);
                
                (void)fread(new_layer->weights->data, sizeof(float), new_layer->weights->total_elements, file);
                free(w_shape);

                int b_dim = 0;
                (void)fread(&b_dim, sizeof(int), 1, file);
                int* b_shape = malloc(b_dim * sizeof(int));
                (void)fread(b_shape, sizeof(int), b_dim, file);
                
                (void)fread(new_layer->biases->data, sizeof(float), new_layer->biases->total_elements, file);
                free(b_shape);

                Network_add_dense(net, new_layer);
            } 
            else if (type == LAYER_CONVOLUTION) {
                int in_channels = w_shape[0];
                int out_channels = w_shape[1];
                int kernel_size = w_shape[2]; 
                
                new_layer = Layer_make_convolution(
                    in_channels, 
                    out_channels, 
                    current_height, 
                    current_width, 
                    kernel_size, 
                    params->stride, 
                    params->padding, 
                    activation
                );
                
                (void)fread(new_layer->weights->data, sizeof(float), new_layer->weights->total_elements, file);
                free(w_shape);

                int b_dim = 0;
                (void)fread(&b_dim, sizeof(int), 1, file);
                int* b_shape = malloc(b_dim * sizeof(int));
                (void)fread(b_shape, sizeof(int), b_dim, file);
                
                (void)fread(new_layer->biases->data, sizeof(float), new_layer->biases->total_elements, file);
                free(b_shape);

                int next_height = (current_height - kernel_size + 2 * params->padding) / params->stride + 1;
                int next_width  = (current_width  - kernel_size + 2 * params->padding) / params->stride + 1;

                Network_add_convolution(net, new_layer, next_height, next_width);

                current_height = next_height;
                current_width  = next_width;

            }

            else if (type == LAYER_POOLING) {
                int in_channels = w_shape[0];
                free(w_shape);

                // Dynamically route sizes and structural stride loops via params configuration
                new_layer = Layer_make_pooling(
                    in_channels, 
                    params->input_height, 
                    params->input_width, 
                    params->pool_size, 
                    params->stride
                );
                Network_add_pooling(net, new_layer);
            }
        }
    }

    if(ferror(file)){
        printf("ERROR: OS error occurred while parsing saved file\n");
    } else {
         printf("Model state successfully loaded!\n");
    }

    fclose(file);
}







/*
 * old one without convolution and pool layers [DELETE LATER AFTER IT WORKS]
void Trainer_load_model(Network* net, Hyperparameters* params, Progress* progress){
    FILE* file = fopen(params->save_location, "rb");
    if (!file) {
        printf("ERROR: Could not open file for loading: %s\n", params->save_location);
        return;
    }

    char header[2];
    fread(header, sizeof(char), 2, file);
    if(strncmp(header, "NN", 2) != 0){
        printf("ERROR: header not found at the top of file\n");
        fclose(file);
        return;
    }

    fread(progress, sizeof(Progress), 1, file);
    
    int num_layers;
    fread(&num_layers, sizeof(int), 1, file);


    //OPTION A with already allocated space
    if(net && net->num_layers > 0){
        if(net->num_layers != num_layers){
            printf("ERROR: inputted layers count does not match loaded layers\n");
            fclose(file);
            return;
        }

        for(int i = 0; i < num_layers; i++){
            NetworkLayer* net_layer = &(net->layers[i]);
            Layer* layer = (Layer*)net_layer->layer_data;

            fread(&(net_layer->type), sizeof(LayerType), 1, file);
            fread(&(layer->activation), sizeof(ActivationType), 1, file);
            
            // 1. Synchronize Weights Layout Read
            int w_ndim;
            fread(&w_ndim, sizeof(int), 1, file);
            if(layer->weights->ndim != w_ndim){
                printf("ERROR: Layer %d weights dimensions mismatch\n", i);
                fclose(file);
                return;
            }

            int* temp_w = malloc(w_ndim * sizeof(int));
            fread(temp_w, sizeof(int), w_ndim, file);
            for(int j = 0; j < w_ndim; j++){
                 if(layer->weights->shape[j] != temp_w[j]){
                    printf("ERROR: Layer %d weights shape mismatch\n", i);
                    free(temp_w);
                    fclose(file);
                    return;
                 }
            }
            free(temp_w);

            fread(layer->weights->data, sizeof(float), layer->weights->total_elements, file);

           
            int b_ndim;
            fread(&b_ndim, sizeof(int), 1, file);
            if(layer->biases->ndim != b_ndim){
                printf("ERROR: Layer %d biases dimensions mismatch\n", i);
                fclose(file);
                return;
            }

            int* temp_b = malloc(b_ndim * sizeof(int));
            fread(temp_b, sizeof(int), b_ndim, file); 
            for(int j = 0; j < b_ndim; j++){
                 if(layer->biases->shape[j] != temp_b[j]){
                    printf("ERROR: Layer %d biases shape mismatch\n", i);
                    free(temp_b);
                    fclose(file);
                    return;
                 }
            }
            free(temp_b);
            fread(layer->biases->data, sizeof(float), layer->biases->total_elements, file);
        }
    }
    // OPTION B building from empty
    else
    {
        if(!net){
            printf("ERROR: must have a valid pointer to network\n");
            fclose(file);
            return;
        }

        net->num_layers = 0;
        net->layers = NULL;

        for(int i = 0; i < num_layers; i++){
            LayerType type;
            ActivationType activation;
            
            fread(&type, sizeof(LayerType), 1, file);
            fread(&activation, sizeof(ActivationType), 1, file);

            int w_dim = 0;
            fread(&w_dim, sizeof(int), 1, file);
            if (w_dim <= 0 || w_dim > 10) {
                printf("ERROR Loaded weights dimension (%d) is corrupted!\n", w_dim);
                fclose(file);
                return;
            }

            int* w_shape = malloc(w_dim * sizeof(int));
            fread(w_shape, sizeof(int), w_dim, file);
            
            int input_dim = w_shape[0];
            int output_dim = w_shape[1];


            // NEED TO UPDATE WITH CURRENT API'S'

            Layer* layer = Layer_make(input_dim, output_dim, activation);
            
            fread(layer->weights->data, sizeof(float), layer->weights->total_elements, file);
            free(w_shape);

            int b_dim = 0;
            fread(&b_dim, sizeof(int), 1, file);
            if (b_dim <= 0 || b_dim > 10) {
                printf("ERROR Loaded biases dimension (%d) is corrupted!\n", b_dim);
                fclose(file);
                return;
            }

            int* b_shape = malloc(b_dim * sizeof(int));
            fread(b_shape, sizeof(int), b_dim, file);
            
            fread(layer->biases->data, sizeof(float), layer->biases->total_elements, file);
            free(b_shape);
            
            Network_add_dense(net, layer);
            net->layers[i].type = type;
        }
    }

    if(ferror(file)){
        printf("ERROR: OS error occurred while parsing saved file\n");
    } else {
         printf("Model state successfully loaded!\n");
    }

    fclose(file);
}
*/





float Trainer_evaluate(Network* net, char* dataset_path, int val_start_sample, int val_samples, Hyperparameters params){
 
    float total_loss = 0.0f;
    int correct_predictions = 0;
    int total_samples = 0;

    if (params.batch_size == 0) {
        return 0.0f;
    }
    
    int num_batches = (val_samples + params.batch_size -1) / params.batch_size;

    int in_channels = net->layers[0].type == 1 ? ((Layer*)net->layers[0].layer_data)->x_cache->shape[1] : 1;

    int in_height = params.input_height;
    int in_width  = params.input_width;

    NetworkLayer* final_net_layer = &(net->layers[net->num_layers - 1]);
    Layer* final_layer = (Layer*)final_net_layer->layer_data;
    int output_dim = final_layer->output_dim;

    // Explicit shape tracking replacements (no longer reading from data->x or data->y layouts)
    int x_shape[4] = {params.batch_size, in_channels, in_height, in_width};
    Tensor* x_work = Tensor_make(4, x_shape);

    int y_shape[2] = {params.batch_size, output_dim};
    Tensor* y_work = Tensor_make(2, y_shape);

    Dataset* batch = Dataset_make(x_work, y_work);

    int argsmax_shape[2] = {params.batch_size, 1};
    Tensor* pred_classes = Tensor_make(2, argsmax_shape);
    Tensor* true_classes = Tensor_make(2, argsmax_shape);

    int grad_shape[2] = {params.batch_size, output_dim};
    Tensor* evaluation_grad_work = Tensor_make(2, grad_shape);
   
    for(int b = 0; b < num_batches; b++){
        int validation_file_batch_idx = (val_start_sample / params.batch_size) + b;
        
        int samples_read = Dataset_read_batch(dataset_path, validation_file_batch_idx, params.batch_size, batch);
        
        if (samples_read <= 0) {
            continue;
        }

        batch->x->shape[0] = samples_read;
        batch->y->shape[0] = samples_read;
        evaluation_grad_work->shape[0] = samples_read;

        Tensor* predictions = Network_forward(net, batch->x);

        total_loss += Network_compute_loss(net, predictions, batch->y, params.losstype, evaluation_grad_work);

        pred_classes->shape[0] = samples_read;
        true_classes->shape[0] = samples_read;
        
        Tensor_argsmax(predictions, pred_classes, params.classification_axis);
        Tensor_argsmax(batch->y, true_classes, params.classification_axis);

        for(int i = 0; i < samples_read; i++){
            if(pred_classes->data[i] == true_classes->data[i]){
                correct_predictions++;
            }
            total_samples++;
        }
    }

    float avg_loss = 0.0f;
    if (num_batches > 0) {
        avg_loss = total_loss / num_batches;
    }

    float accuracy = 0.0f;
    if (total_samples > 0) {
        accuracy = ((float)correct_predictions / total_samples) * 100.0f;
    }

    printf("Evaluation: |Avg_loss: %.2f| |Accuracy: %.2f|\n", avg_loss, accuracy);


    Tensor_free(pred_classes);
    Tensor_free(true_classes);
    Tensor_free(evaluation_grad_work);
    Dataset_free(batch);


    return avg_loss;
}

