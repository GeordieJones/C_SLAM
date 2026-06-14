#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "dataset.h"
#include "tensor.h"
#include "trainer.h"
#include "layer.h"

void Trainer_train(Network* net, Dataset* data, Hyperparameters* params, Progress* progress){

    int save_count = 0;


    Dataset* training_data = malloc(sizeof(Dataset));
    Dataset* testing_data = malloc(sizeof(Dataset));

    Dataset_split(data, params->training_ratio, training_data, testing_data); 
    
    int* x_batch_shape = malloc(training_data->x->ndim * sizeof(int));
    memcpy(x_batch_shape, training_data->x->shape, training_data->x->ndim * sizeof(int));
    x_batch_shape[0] = params->batch_size;

    int* y_batch_shape = malloc(training_data->y->ndim * sizeof(int));
    memcpy(y_batch_shape, training_data->y->shape, training_data->y->ndim * sizeof(int));
    y_batch_shape[0] = params->batch_size;

    Tensor* x_work = Tensor_make(training_data->x->ndim, x_batch_shape);
    Tensor* y_work = Tensor_make(training_data->y->ndim, y_batch_shape);
    Dataset* batch = Dataset_make(x_work, y_work);
   
    int grad_shape[2] = {params->batch_size, training_data->y->shape[1]};
    Tensor* loss_grad_work = Tensor_make(2, grad_shape);


   // Allocate a workspace to catch the discarded input gradients in the backward call;
    Tensor* dummy_input_grad = Tensor_make(batch->x->ndim, x_batch_shape);


    Tensor* predictions = NULL;
    
   
   
    float avg_loss;

    for(int i = progress->current_epoch; i < params->max_epochs; i++){

        progress->current_epoch = i;
        Dataset_shuffle(training_data);
        float epoch_loss_sum = 0.0f;
        int batch_count = 0;

        for(int b = 0; b < training_data->num_samples; b += params->batch_size){
            Dataset_get_batch(training_data, batch, batch_count, params->batch_size);

            batch->x->shape[0] = params->batch_size;
            batch->y->shape[0] = params->batch_size;

            predictions = Network_forward(net, batch->x);

            epoch_loss_sum += Network_compute_loss(net,predictions, batch->y, params->losstype, loss_grad_work);

            Network_backward(net, loss_grad_work, dummy_input_grad);
 
            Network_update(net, params->learning_rate);

            batch_count++;
        }        

        avg_loss = Trainer_evaluate(net, testing_data, *params);


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

        if(save_count == params->save_frequency){
            Trainer_save_model(net, params, progress);
            save_count = 0;
        }else{
            save_count++;
        }



        float display_loss = 0.0f;
        if (batch_count > 0) {
            display_loss = epoch_loss_sum / batch_count;
        } else {
            printf("[WARNING] batch_count is 0! training_data->num_samples is %d\n", training_data->num_samples);
        }

        printf("Epoch %d | Train Loss: %.4f | Val Loss: %.4f\n", i + 1, display_loss, avg_loss);

    }

    
    printf("finished training best validation was: %f", progress->best_validation_loss);
    Trainer_save_model(net, params, progress);

    Tensor_free(dummy_input_grad);
    Tensor_free(loss_grad_work);
    Dataset_free(batch);
    
    free(training_data->row_indexs);
    free(testing_data->row_indexs);
    free(training_data);
    free(testing_data);
    free(x_batch_shape);
    free(y_batch_shape);
}




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
              
        //weights
        fwrite(&(layer->weights->ndim), sizeof(int), 1, file);
        fwrite(layer->weights->shape, sizeof(int), (layer->weights->ndim), file);
        fwrite(layer->weights->data, sizeof(float), layer->weights->total_elements, file);

        //biases

        fwrite(&(layer->biases->ndim), sizeof(int), 1, file);
        fwrite(layer->biases->shape, sizeof(int), (layer->biases->ndim), file);        
        fwrite(layer->biases->data, sizeof(float), layer->biases->total_elements, file);
   
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






float Trainer_evaluate(Network* net, Dataset* data, Hyperparameters params){
 
    float total_loss = 0.0f;
    int correct_predictions = 0;
    int total_samples = 0;

    if (params.batch_size == 0) {
        return 0.0f;
    }
    
    int num_batches = (data->num_samples + params.batch_size -1) / params.batch_size;


    int* x_shape = malloc(data->x->ndim * sizeof(int));
    memcpy(x_shape, data->x->shape, data->x->ndim * sizeof(int));
    x_shape[0] = params.batch_size;
    Tensor* x_work = Tensor_make(data->x->ndim, x_shape);


    int* y_shape = malloc(data->y->ndim * sizeof(int));
    memcpy(y_shape, data->y->shape, data->y->ndim * sizeof(int));
    y_shape[0] = params.batch_size;
    Tensor* y_work = Tensor_make(data->y->ndim, y_shape);


    Dataset* batch = Dataset_make(x_work, y_work);

    int argsmax_shape[2] = {params.batch_size, 1};
    Tensor* pred_classes = Tensor_make(2, argsmax_shape);
    Tensor* true_classes = Tensor_make(2, argsmax_shape);

    int grad_shape[2] = {params.batch_size, data->y->shape[1]};
    Tensor* evaluation_grad_work = Tensor_make(2, grad_shape);

    for(int b = 0; b < num_batches; b++){
        Dataset_get_batch(data, batch, b, params.batch_size);

        int current_batch_real_samples = params.batch_size;
        if(b == num_batches -1){
            if(params.batch_size >0){
                int rem = data->num_samples % params.batch_size;
                if(rem != 0 ){
                    current_batch_real_samples = rem;
                }
            }else{
                printf("ERROR params.batch_size is 0 in trainer_evel\n");
                current_batch_real_samples = 0;
            }
        }

        batch->x->shape[0] = params.batch_size;
        batch->y->shape[0] = params.batch_size;

        Tensor* predictions = Network_forward(net, batch->x);


        total_loss += Network_compute_loss(net, predictions, batch->y, params.losstype, evaluation_grad_work);

        pred_classes->shape[0] = params.batch_size;
        true_classes->shape[0] = params.batch_size;
        
        Tensor_argsmax(predictions, pred_classes, params.classification_axis);
        Tensor_argsmax(batch->y, true_classes, params.classification_axis);

        for(int i = 0; i < current_batch_real_samples; i++){
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
    free(x_shape);
    free(y_shape);

    return avg_loss;
}

