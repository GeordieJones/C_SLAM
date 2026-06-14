#ifndef TRAINER_H
#define TRAINER_H

#include "network.h"
#include "dataset.h"

typedef struct{
    float learning_rate;
    int batch_size;
    int max_epochs;
    int early_stop;
    float early_stop_delta;
    float training_ratio;
    LossType losstype;
    int classification_axis;
    const char* save_location;
    int save_frequency;
}Hyperparameters;

typedef struct{
    int current_epoch;
    float best_validation_loss;
    int unchanged_counter;
}Progress;


void Trainer_train(Network* net, Dataset* data, Hyperparameters* params, Progress* state);

void Trainer_save_model(Network* net, Hyperparameters* params, Progress* progress);

void Trainer_load_model(Network* net, Hyperparameters* params, Progress* progress);

//assumes data is already split from the training data
float Trainer_evaluate(Network* net, Dataset* data, Hyperparameters params);

#endif
