#include "tensor.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>




Tensor* Tensor_make(int ndim, int* shape){
    Tensor* self = (Tensor*)malloc(sizeof(Tensor));
    self->ndim = ndim;
    self->total_elements = 1;
    for(int i = 0; i < ndim; i++){
        self->shape[i] = shape[i];
        self->total_elements *= shape[i];
    }

    self->data = (float*)calloc(self->total_elements, sizeof(float));

    int current_stride = 1;
    for(int i = ndim - 1; i >= 0; i--){
        self->strides[i] = current_stride;
        current_stride *= self->shape[i];
    }

    return self;
}


float Tensor_at(Tensor* self, int* coords){
    int loc = 0;
    for(int i = 0; i < self->ndim; i++){
        loc += self->strides[i] * (coords[i]); 
    }
    return self->data[loc];
}



void Tensor_print(Tensor* self){
    printf("\nTensor number of dimesions: %d\n", self->ndim);
    printf("Tensor total_elements: %d\n", self->total_elements);

    printf("Tensor shape [");
    for(int i = 0; i < self->ndim; i++){
        printf("%d, ", self->shape[i]);
    }
    printf("]\n");

    printf("Tensor strides [");
    for(int i = 0; i < self->ndim; i++){
        printf("%d, ", self->strides[i]);
    }
    printf("]\n");
}

void Tensor_free(Tensor* self){
    if(!self) return;
    if(self->data) free(self->data);
    free(self);
}


void Tensor_zero(Tensor* self){
    if (!self || !self->data) return;
    for(int i = 0; i < self->total_elements; i++){
        self->data[i] = 0.0f;
    }
}


void Tensor_rand(Tensor* self){
    float random = 0.0;
    for(int i = 0; i < self->total_elements; i++){
        random = ((rand()) % 1000)*0.001;
        self->data[i] = random;
    }
}


void Tensor_scale(Tensor* self, float multiplier){
    for(int i = 0; i < self->total_elements; i++){
        self->data[i] *= multiplier;
    }
}


float Tensor_sum(Tensor* self){
    float sum = 0.0f;
    for(int i = 0; i < self->total_elements; i++){
        sum += self->data[i];
    }
    return sum;
}

void Tensor_sum_axis(const Tensor* src, Tensor* dest, int axis){

    int coords[src->ndim];
    int axis_size = src->shape[axis];
    int axis_stride = src->strides[axis];

    for(int i = 0; i < dest->total_elements; i++){
        int rem_idx = i;
        for(int j = 0; j < dest->ndim; j++){
            coords[j] = rem_idx / dest->strides[j];
            rem_idx %= dest->strides[j];
        }
        
        
        if(dest->ndim < src->ndim){
            for(int j = src->ndim -1; j>axis; j--){
                coords[j] = coords[j-1];
            }
            coords[axis] = 0;
        }
        
        int src_start_idx = 0;
        for(int j = 0; j < src->ndim; j++){
            src_start_idx +=(coords[j] * src->strides[j]);
        }

        float sum = 0.0f;
        for(int k = 0; k< axis_size; k++){
            int src_idx = src_start_idx + (k * axis_stride);
            sum += src->data[src_idx];
        }
        dest->data[i] = sum;
    }
    
}


void Tensor_copy(const Tensor* src, Tensor* dest){
    if(src->ndim != dest->ndim || src->total_elements > dest->total_elements){
        printf("ERROR: src_ndim(%d), dest_ndim(%d), src_total(%d), dest_total(%d)", src->ndim, dest->ndim, src->total_elements, dest->total_elements);
        return;
    }
     
    memcpy(dest->data, src->data, src->total_elements * sizeof(float));
}



Tensor* Tensor_clone(const Tensor* self){
    Tensor* new_tensor = Tensor_make(self->ndim, (int*)self->shape);
    Tensor_copy(self, new_tensor);
    return new_tensor;
}


int Tensor_matches_shape(const Tensor* t, int ndim, const int* shape) {
    if (!t || t->ndim != ndim) return 0;
    for (int i = 0; i < ndim; i++) {
        if (t->shape[i] != shape[i]) return 0;
    }
    return 1;
}




MinMax Tensor_min_max(Tensor* self){
    float min = FLT_MAX;
    float max = -FLT_MAX;
    int min_loc = 0;
    int max_loc = 0;
    for(int i = 0; i < self->total_elements; i++){
        if(self->data[i] < min) {
            min = self->data[i];
            min_loc = i;
        }
        if(self->data[i] > max){
            max = self->data[i];
            max_loc = i;
        }
    }

    MinMax results;
    results.min = min;
    results.min_loc = min_loc;
    results.max = max;
    results.max_loc = max_loc;

    return results;
}



void Tensor_argsmax(const Tensor* src, Tensor* dest, int axis){ 
    int max_ndim = (src->ndim > dest->ndim) ? src->ndim : dest->ndim;
    int* coords = malloc(max_ndim * sizeof(int));

    int axis_size = src->shape[axis];
    int axis_stride = src->strides[axis];

    for(int i = 0; i < dest->total_elements; i++){
        int rem_idx = i;
        
        for(int j = 0; j < max_ndim; j++) {
            coords[j] = 0;
        }
 
        for(int j = 0; j < dest->ndim; j++){
            if (dest->strides[j] > 0) {
                coords[j] = rem_idx / dest->strides[j]; 
                rem_idx %= dest->strides[j];
            } else {
                coords[j] = 0;
            }
        }

        if(dest->ndim < src->ndim){
            for(int j = src->ndim - 1; j > axis; j--){
                coords[j] = coords[j - 1];
            }
            coords[axis] = 0;
        }

        int src_start_idx = 0;
        for(int j = 0; j < src->ndim; j++){
            src_start_idx += (coords[j] * src->strides[j]);
        }

        float max_val = -FLT_MAX;
        int best_idx = 0;
        for(int k = 0; k < axis_size; k++){
            int src_idx = src_start_idx + (k * axis_stride);
            float current_val = src->data[src_idx];

            if(current_val > max_val){
                max_val = current_val;
                best_idx = k;
            }
        }
        dest->data[i] = (float)best_idx;
    }

    free(coords);
}





void Tensor_argsmin(const Tensor* src, Tensor* dest, int axis){
    int coords[src->ndim];
    int axis_size = src->shape[axis];
    int axis_stride = src->strides[axis];

    for(int i = 0; i < dest->total_elements; i++){
        int rem_idx = i;
        for(int j = 0; j < dest->ndim; j++){
            coords[j] = rem_idx / dest->strides[i];
            rem_idx %= dest->strides[j];
        }
        
        //ask about this
        if(dest->ndim < src->ndim){
            for(int j = src->ndim -1; j>axis; j--){
                coords[j] = coords[j-1];
            }
            coords[axis] = 0;
        }
        
        int src_start_idx = 0;
        for(int j = 0; j < src->ndim; j++){
            src_start_idx +=(coords[j] * src->strides[j]);
        }

        float min = FLT_MAX;
        int best_idx = 0;
        for(int k = 0; k< axis_size; k++){
            int src_idx = src_start_idx + (k * axis_stride);
            float current_val = src->data[src_idx];
            
            if(current_val < min){
                min = current_val;
                best_idx = k;
            }
        }
        dest->data[i] = (float)best_idx;
    }
}



void Tensor_map(Tensor* self, float(*operation)(float)){
    for(int i = 0; i < (self->total_elements); i++){
        self->data[i] = operation(self->data[i]);    
    }
}


void Tensor_map_axis(Tensor* self, float(*operation)(float, int), int axis){
    int axis_size = self->shape[axis];
    int axis_stride = self->strides[axis];
    int num_lines = self->total_elements / axis_size;

    for(int i = 0; i < num_lines; i++){
        int coords[self->ndim];
        int rem_idx = i;
        for(int j = 0; j < self->ndim; j++){
            if(j == axis) continue;
            coords[j] = rem_idx / self->strides[i];
            rem_idx %= self->strides[j];
        }
        coords[axis] = 0;
         
        int line_start_idx = 0;
        for(int j = 0; j < self->ndim; j++){
            line_start_idx += (coords[j] * self->strides[j]);
        }

        for(int k = 0; k< axis_size; k++){
            int src_idx = line_start_idx + (k * axis_stride);
            self->data[src_idx] = operation(self->data[src_idx], k);
        }
    }
}

void Tensor_reshape(Tensor* self, int* new_dims, int new_size){
    int new_dims_sum = 1;
    for(int i = 0; i < new_size; i++){
        new_dims_sum *= new_dims[i];
    }
    if(new_dims_sum != self->total_elements){
        printf("ERROR: number of elements do not match new dimision");
        return;
    }

    self->ndim = new_size;
    
    for(int i = 0; i < new_size; i++){
        self->shape[i] = new_dims[i];
    }
    
    int current_stride = 1;
    for(int i = new_size - 1; i >= 0; i--){
        self->strides[i] = current_stride;
        current_stride *= new_dims[i];
    }

}


//Start of unknown

void Tensor_hadamard(Tensor* self, Tensor* new){
    for(int i = 0; i < self->total_elements; i++){
        self->data[i] *= new->data[i];
    }
}


void Tensor_lazy_transpose(Tensor* self, int idx_from, int idx_to){
    int temp;
    temp = self->shape[idx_from];
    self->shape[idx_from] = self->shape[idx_to];
    self->shape[idx_to] = temp;


    temp = self->strides[idx_from];
    self->strides[idx_from] = self->strides[idx_to];
    self->strides[idx_to] = temp;
}


Tensor* Tensor_eager_transpose(Tensor* self, int idx_from, int idx_to){
    int new_shape[self->ndim];
    for(int i = 0; i < self->ndim; i++){
        if(i == idx_from){
            new_shape[i] = self->shape[idx_to];
            continue;
        }
        if(i == idx_to){
            new_shape[i] = self->shape[idx_from];
            continue;
        }
        new_shape[i] = self->shape[i];
    }
    //should make a new tensor with the new dimesions
    Tensor* dest = Tensor_make(self->ndim, new_shape);
    int coords[self->ndim];

    for(int i = 0; i < dest->total_elements; i++){
        int rem_idx = i;
        
        for(int j = 0; j < dest->ndim; j++){
            coords[j] = rem_idx / dest->strides[j];
            rem_idx %= dest->strides[j];
        }

        int src_coords[self->ndim];
        for(int j = 0; j < dest->ndim; j++){
            if(j == idx_from){
                src_coords[j] = coords[idx_to];
            }else if(j == idx_from){
                src_coords[j] = coords[idx_from];
            }else{
                src_coords[j] = coords[j];
            }

        }


        int src_idx = 0;
        for(int j = 0; j < self->ndim; j++){
            src_idx += (src_coords[j] * self->strides[j]);
        }
        dest->data[i] = self->data[src_idx];

    }

    return dest;
}



void Tensor_matmul(Tensor* a, Tensor* b, Tensor* dest){
    if(a->shape[a->ndim - 1] != b->shape[b->ndim - 2]){
        fprintf(stderr, "CRITICAL ERROR: Matrix Multiplication Shape Mismatch!\n");
        fprintf(stderr, "Matrix A inner dimension: %d | Matrix B inner dimension: %d\n", a->shape[a->ndim - 1], b->shape[b->ndim - 2]);
    exit(EXIT_FAILURE);
    }
    
    //int out_shape[a->ndim];
    int total_matrices = 1;

    for(int i = 0; i < a->ndim -2; i ++){
        if(a->shape[i] != b->shape[i]){
            printf("shape at index %d doesnt match \n", i);
            return;
        }else{
            //out_shape[i] = a->shape[i];
            total_matrices *= a->shape[i];    
        }
    }
    
    int M = a->shape[a->ndim - 2];
    int N = a->shape[a->ndim - 1];
    int P = b->shape[b->ndim - 1];
    
    //out_shape[a->ndim-2] = M;
    //out_shape[a->ndim-1] = P;

    //Tensor* dest = Tensor_make(a->ndim, out_shape);

    int a_mat_stride = (a->ndim > 2) ? a->strides[a->ndim - 3] : 0;
    int b_mat_stride = (b->ndim > 2) ? b->strides[b->ndim - 3] : 0;
    int dest_mat_stride = (dest->ndim > 2) ? dest->strides[dest->ndim - 3] : 0;


    for(int m = 0; m < total_matrices; m++){
        
        int a_offset = m * a_mat_stride;        
        int b_offset = m * b_mat_stride;        
        int dest_offset = m * dest_mat_stride;

        for(int i = 0; i < M; i++){
            for(int j = 0; j < P; j++){
                float sum = 0.0f;
                for(int k = 0; k < N; k++){
                    int a_idx = a_offset + (i * a->strides[a->ndim - 2]) + (k * a->strides[a->ndim -1]);

                    int b_idx = b_offset + (k * b->strides[b->ndim - 2]) + (j * b->strides[b->ndim -1]);

                    sum += a->data[a_idx] * b->data[b_idx];
                }
                int dest_idx = dest_offset + (i* dest->strides[dest->ndim -2]) + (j * dest->strides[dest->ndim - 1]);
                dest->data[dest_idx] = sum;
            }
        }


    }

   // return dest;
}






float Tensor_vector_dot(Tensor* a, Tensor* b){
    if(a->total_elements == b->total_elements){
        float sum = 0.0f;
        for(int i = 0; i < a->total_elements; i++){
            sum += a->data[i] * b->data[i];
        }
        return sum;
    }else{
        printf("ERROR: total elements dont match\n");
        return 0.0f;
    }
}



Tensor* Tensor_contract(const Tensor* a, const Tensor* b, int axis_a, int axis_b){
    
    Tensor* a_work = Tensor_clone(a);
    Tensor* b_work = Tensor_clone(b);

    if(axis_a != a_work->ndim -1 ){
        Tensor* temp = Tensor_eager_transpose(a_work, axis_a, a_work->ndim -1);
        a_work = temp;
    }
    if(axis_b != 0){
        Tensor* temp = Tensor_eager_transpose(b_work, axis_b, 0);
        b_work = temp;
    }

    int contracting_size = a_work->shape[a_work->ndim -1];

    int a_rows = 1;
    for(int i = 0; i < a_work->ndim; i++){
        a_rows *= a_work->shape[i];
    }
    a_rows /= contracting_size;
    
    int a_2d_dims[2] = {a_rows, contracting_size};
    Tensor_reshape(a_work, a_2d_dims, 2);

    int b_cols = 1;
    for(int i = 0; i < b_work->ndim; i++){
        b_cols *= b_work->shape[i];
    }
    b_cols /= contracting_size;
    
    int b_2d_dims[2] = {contracting_size, b_cols};
    Tensor_reshape(b_work, b_2d_dims, 2);

    int result_2d_dims[2] = {a_rows, b_cols};
    Tensor* result_2d = Tensor_make(2, result_2d_dims);

   
    if(result_2d == NULL){
        return NULL;
        Tensor_free(a_work);
        Tensor_free(b_work);
    }

    Tensor_matmul(a_work, b_work, result_2d);
    Tensor_free(a_work);
    Tensor_free(b_work);


    int final_ndim = (a->ndim -1) + (b->ndim -1);
    int final_shape[a->ndim];
    int idx = 0;

    for(int i = 0; i < a->ndim; i++){
        if(i != axis_a) final_shape[idx++] = a->shape[i];
    }
    for(int i = 0; i < b->ndim; i++){
        if(i != axis_b) final_shape[idx++] = b->shape[i];
    }

    Tensor_reshape(result_2d, final_shape, final_ndim);

    return result_2d;
}










Tensor* Tensor_append(Tensor* self, Tensor* addon, int axis){
    int new_shape[self->ndim];
    for(int i = 0; i < self->ndim; i++){
        if(i == axis){
            new_shape[i] = self->shape[axis] + addon->shape[axis] ;
        }
        else if(self->shape[i] != addon->shape[i]){
            printf("ERROR: shape at %d is not the same", i);
            return NULL;
        }
        else{
            new_shape[i] = self->shape[i];
        }
    }
    //should make a new tensor with the new dimesions
    Tensor* dest = Tensor_make(self->ndim, new_shape);
    
    int outer_loops = 1;
    for(int i = 0; i < axis; i++){
        outer_loops *= self->shape[i];
    }

    
    int self_block_size = self->strides[axis] * self->shape[axis];
    int addon_block_size = addon->strides[axis] * addon->shape[axis];
    int dest_block_size = dest->strides[axis] * dest->shape[axis];

    size_t self_bytes = self_block_size * sizeof(float);
    size_t addon_bytes = addon_block_size * sizeof(float);

    for(int m = 0; m < outer_loops; m++){
        float* dest_ptr = dest->data + (m * dest_block_size);
        float* self_ptr = self->data + (m * self_block_size);
        float* addon_ptr = addon->data + (m * addon_block_size);


        memcpy(dest_ptr, self_ptr, self_bytes);
        memcpy(dest_ptr + self_block_size, addon_ptr, addon_bytes);
    }

    return dest;
}



void Tensor_vector_addition(Tensor* a, Tensor* b){

    if(b->ndim != 1){
        printf("ERROR: not a 1D vector\n");
        return;
    }   
    int vector_len = b->shape[0];
    int tensor_last_dim = a->shape[a->ndim -1];
   
    if(tensor_last_dim != vector_len){
        printf("vector size(%d) doesnt match a's last dim(%d)", vector_len, tensor_last_dim);
        return;
    }

    int total_rows = a->total_elements / vector_len;
    int index =0;
    for(int r = 0; r < total_rows; r++){
        for(int c = 0; c < vector_len; c++){
            a->data[index] += b->data[c];
            index++;
        }
    }
}

