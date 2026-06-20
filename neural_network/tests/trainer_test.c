#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Direct local inclusion matching your pathing structures
#include "../libaries/tensor.h"
#include "../libaries/layer.h"
#include "../libaries/dataset.h"
#include "../libaries/network.h"
#include "../libaries/trainer.h"

// Helper function to build automated mock datasets using your framework's API
Dataset* create_test_dataset(int num_samples, int input_dim, int output_dim) {
    int x_shape[MAX_DIMS] = {num_samples, input_dim, 1, 1};
    int y_shape[MAX_DIMS] = {num_samples, output_dim, 1, 1};

    Tensor* x = Tensor_make(2, x_shape);
    Tensor* y = Tensor_make(2, y_shape);

    // Populate data with dummy values using your library calls
    Tensor_rand(x);
    Tensor_rand(y);

    Dataset* ds = Dataset_make(x, y);
    ds->num_samples = num_samples;
    return ds;
}

int main(void) {
    printf("==================================================\n");
    printf("       RUNNING TRAINER INTEGRATION TEST SUITE     \n");
    printf("==================================================\n");

    // 1. Initialize Datasets (24 samples to evenly split batches of 6)
    int samples = 24;
    int inputs = 4;
    int outputs = 2;
    Dataset* train_dataset = create_test_dataset(samples, inputs, outputs);
    Dataset* val_dataset = create_test_dataset(6, inputs, outputs);
    printf("[INFO] Datasets successfully initialized via Tensor/Dataset APIs.\n");

    // 2. Build Your Network
    Network* net = Network_make();
    if (!net) {
        fprintf(stderr, "[ERROR] Network_make returned NULL. Verify core allocation.\n");
        return 1;
    }

    // Creating a mock layer matching layer.h assumptions
    Layer* dense_layer = Layer_make(inputs, outputs, ACTIVATION_SOFTMAX);
    Network_add_dense(net, dense_layer);
    printf("[INFO] Test Network constructed with 1 Dense Layer.\n");

    // 3. Define Stable Hyperparameters
    Hyperparameters params;
    params.learning_rate = 0.005f;
    params.batch_size = 6;
    params.max_epochs = 3;
    params.early_stop = 5;
    params.training_ratio = 0.8f;
    params.classification_axis = 1;
    params.early_stop_delta = 0.0001f;
    params.losstype = LOSS_CROSS_ENTROPY;
    params.save_location = "test_checkpoint.bin";
    params.save_frequency = 1;
    // Note: If you face problems with `params.classification_axis` in Trainer_evaluate, 
    // make sure it's explicitly defined inside your trainer.h Hyperparameters struct.

    // 4. Initialize Tracker Progress States
    Progress progress;
    progress.current_epoch = 0;
    progress.best_validation_loss = 999.0f;
    progress.unchanged_counter = 0;

    // 5. Execute Core Training Function
    printf("\n--- Invoking Trainer_train ---\n");
    Trainer_train(net, train_dataset, &params, &progress);
    printf("--- Trainer_train finished cycle ---\n\n");

    // 6. Verify Serialization (Save / Load Check)
    printf("--- Testing State Serialization Pipeline ---\n");
    Trainer_save_model(net, &params, &progress);

    Progress reloaded_progress;
    Network* reloaded_net = Network_make();
    
    printf("[INFO] Loading state back from disk: %s\n", params.save_location);
    Trainer_load_model(reloaded_net, &params, &reloaded_progress);

    // Validate that metadata matches perfectly across the streaming pipe
    assert(reloaded_progress.current_epoch == progress.current_epoch);
    printf("[SUCCESS] Verified tracking state. Saved Epoch %d == Loaded Epoch %d\n", 
            progress.current_epoch, reloaded_progress.current_epoch);

    // 7. Cleanup Test Sandbox Environments
    printf("\n--- Cleaning Up Allocations ---\n");
    remove(params.save_location);

    Dataset_free(train_dataset);

    Dataset_free(val_dataset);

    Network_free(net);
    Network_free(reloaded_net);

    printf("==================================================\n");
    printf("         ALL TRAINER TESTS PASSED CLEANLY         \n");
    printf("==================================================\n");

    return 0;
}
