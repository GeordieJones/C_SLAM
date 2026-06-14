#include "dataset.h"
#include "tensor.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>

#define EPSILON 0.0001f
#define ASSERT_CLOSE(a,b) assert(fabsf((a) - (b)) < EPSILON)

static Tensor* create_mock_tensor(int rows, int cols, float initial_val){
    int dims[2] = {rows, cols};
    Tensor* m = Tensor_make(2, dims);
    if(!m) return NULL;
    for(int i = 0; i < rows*cols; i++){
        m->data[i] = initial_val;
    }
    return m;
}

static Tensor* create_sequential_tensor(int rows, int cols){
    int dims[2] = {rows, cols};
    Tensor* m = Tensor_make(2, dims);
    if(!m) return NULL;
    for(int i = 0; i < rows * cols; i++){
        m->data[i] = (float)i;
    }
    return m;
}

void test_dataset_make_and_free(){
    printf("running test_dataset_make_and_free \n");
    Tensor* x = create_mock_tensor(10, 5, 1.0f);
    Tensor* y = create_mock_tensor(10, 1, 0.0f);

    Dataset* ds = Dataset_make(x, y);
    assert(ds != NULL);
    assert(ds->x == x);
    assert(ds->y == y);
    assert(ds->num_samples == 10);
    assert(ds->row_indexs != NULL);

    for(int i = 0; i < ds->num_samples; i++){
        assert(ds->row_indexs[i] == i);
    }

    Dataset_free(ds);

    Dataset* ds_null = Dataset_make(NULL, NULL);
    assert(ds_null == NULL);

    printf("-> test dataset_make_and_free passed\n");
}

void test_dataset_scaling_and_normalization(){
    printf("Running test_dataset_scaling_and_normalization\n");
    
    int dims[2] = {2, 2};
    Tensor* x = Tensor_make(2, dims);
    x->data[0] = 0.0f; x->data[1] = 5.0f;
    x->data[2] = 2.0f; x->data[3] = 10.0f;

    Tensor* y = create_mock_tensor(2, 1, 1.0f);

    Dataset* ds = Dataset_make(x, y);

    Dataset_min_max_scale(ds, 0.0f, 1.0f, 1);

    ASSERT_CLOSE(ds->x->data[0], 0.0f);
    ASSERT_CLOSE(ds->x->data[1], 0.5f);
    ASSERT_CLOSE(ds->x->data[3], 1.0f);

    Tensor* x_flat = create_mock_tensor(2, 2, 5.0f);
    Dataset* ds_flat = Dataset_make(x_flat, create_mock_tensor(2, 1, 1.0f));
    
    Dataset_min_max_scale(ds_flat, 0.0f, 1.0f, 1);
    assert(!isnan(ds_flat->x->data[0]));

    Dataset_free(ds);
    Dataset_free(ds_flat);

    printf("-> test_dataset_scaling passed \n");
}

void test_dataset_shuffle(){
    printf("testing dataset_shuffle \n");

    Tensor* x = create_sequential_tensor(100, 2);
    Tensor* y = create_sequential_tensor(100, 1);
    Dataset* ds = Dataset_make(x, y);

    int identity_match_count = 0; 
    Dataset_shuffle(ds);

    for(int i = 0; i < ds->num_samples; i++){
        if(ds->row_indexs[i] == i){
            identity_match_count++;
        }
        assert(ds->row_indexs[i] >= 0 && ds->row_indexs[i] < 100);
    }
    Dataset_free(ds);
    printf("-> test_dataset_shuffle passed\n");
}

void test_dataset_split(){
    printf("running test_dataset_split\n");

    Tensor* x = create_sequential_tensor(10, 2);
    Tensor* y = create_sequential_tensor(10, 1);
    Dataset* ds = Dataset_make(x, y);

    Dataset train_ds;
    Dataset val_ds;

    Dataset_split(ds, 0.8f, &train_ds, &val_ds);

    printf("train count: %d, Val count = %d \n\n", train_ds.num_samples, val_ds.num_samples);
    assert(train_ds.num_samples == 8);
    assert(val_ds.num_samples == 2);
    assert(train_ds.row_indexs != NULL);
    assert(val_ds.row_indexs != NULL);

    free(train_ds.row_indexs);
    free(val_ds.row_indexs);

    Dataset_split(ds, 1.0f, &train_ds, &val_ds);
    printf("train count: %d, Val count = %d \n\n", train_ds.num_samples, val_ds.num_samples);
    assert(train_ds.num_samples == 10);
    assert(val_ds.num_samples == 0);
    free(train_ds.row_indexs);
    if(val_ds.row_indexs) free(val_ds.row_indexs);
    
    Dataset_split(ds, 0.0f, &train_ds, &val_ds);
    printf("train count: %d, Val count = %d \n\n", train_ds.num_samples, val_ds.num_samples);
    assert(train_ds.num_samples == 0);
    assert(val_ds.num_samples == 10);
    free(val_ds.row_indexs);
    if(train_ds.row_indexs) free(train_ds.row_indexs);

    Dataset_free(ds);
    printf("->passed test_dataset_split\n");
}

void test_dataset_get_batch(){
    printf("Running test_dataset_get_batch \n");

    Tensor* x = create_sequential_tensor(10, 2);
    Tensor* y = create_sequential_tensor(10, 1);
    Dataset* ds = Dataset_make(x, y);

    Dataset batch;
    int b_x_dims[2] = {3, 2};
    int b_y_dims[2] = {3, 1};
    batch.x = Tensor_make(2, b_x_dims);
    batch.y = Tensor_make(2, b_y_dims);
    batch.row_indexs = NULL;

    Dataset_get_batch(ds, &batch, 0, 3);
    assert(batch.num_samples == 3);

    Dataset batch_last;
    int bl_x_dims[2] = {1, 2};
    int bl_y_dims[2] = {1, 1};
    batch_last.x = Tensor_make(2, bl_x_dims);
    batch_last.y = Tensor_make(2, bl_y_dims);
    batch_last.row_indexs = NULL;

    Dataset_get_batch(ds, &batch_last, 3, 3);
    assert(batch_last.num_samples == 1);

    Tensor_free(batch.x);
    Tensor_free(batch.y);
    Tensor_free(batch_last.x);
    Tensor_free(batch_last.y);
    Dataset_free(ds);

    printf("-> test_dataset_get_batch PASSED \n");
}

void test_io_error_handling(){
    printf("running io edge cases\n");

    int label_cols[1] = {0};
    Dataset* fake_csv = Dataset_read_csv("doesnt_exist.csv", label_cols, 1, 1);
    assert(fake_csv == NULL);

    Dataset* fake_mnist = Dataset_read_mnist("fake_img.idx", "fake_lbl.idx");
    assert(fake_mnist == NULL);

    printf("-> image and csv fake files passed\n");
}

void test_img_processing(){
    printf("running image processing tests\n");
    int dims[2] = {1, 5};
    Tensor* img = Tensor_make(2, dims);

    img->data[0] = 0.1f; img->data[1] = 0.4f; img->data[2] = 0.5f; 
    img->data[3] = 0.6f; img->data[4] = 0.9f;

    img_threshold(img, 0.5f);
    assert(img->data[2] == 0.0f);
    assert(img->data[3] == 1.0f);

    int inv_dims[2] = {1, 2};
    Tensor* img_inv = Tensor_make(2, inv_dims);
    img_inv->data[0] = 0.0f;
    img_inv->data[1] = 1.0f;

    img_invert(img_inv);
    ASSERT_CLOSE(img_inv->data[0], 1.0f);
    ASSERT_CLOSE(img_inv->data[1], 0.0f);

    Tensor_free(img);
    Tensor_free(img_inv);

    printf("-> img processing passed \n");
}

int main(){
    printf("==========Starting Dataset tests==============\n\n");
    test_dataset_make_and_free();
    test_dataset_scaling_and_normalization();
    test_dataset_shuffle();
    test_dataset_split();
    test_dataset_get_batch();
    test_img_processing();
    test_io_error_handling();
    printf("\n\n==========all tests passed=====================\n\n");
    return 0;
}

























