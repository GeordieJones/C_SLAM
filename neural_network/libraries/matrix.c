#include "matrix.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>



Matrix* Matrix_make(int rows, int cols){
    Matrix* self = malloc(sizeof(Matrix));
    self->rows = rows;
    self->cols = cols;
    self->data = calloc(rows * cols, sizeof(float));

    return self;
}

void Matrix_print(Matrix* self){
    printf("\n[ ");
    for(int i = 0; i < self->rows; i++){
        for(int j=0; j<self->cols; j++){
            printf("%f ", self->data[i*self->cols + j]);
        }
        if(i != (self->rows -1)){
            printf("\n");
        }
    }
    printf("]\n\n");
}

void Matrix_free(Matrix* self){
    free(self->data);
    free(self);
}


void Matrix_zero(Matrix* self){
     for(int i = 0; i < self->rows; i++){
        for(int j=0; j < self->cols; j++){
            self->data[i*self->cols + j] = 0.0;
        }
    }
}


void Matrix_rand(Matrix* self){
    float random = 0.0;
    for(int i = 0; i < self->rows; i++){
        for(int j=0; j < self->cols; j++){
            random = ((rand()) % 1000)*0.001;
            self->data[i*self->cols + j] = random;
        }
    }
}


void Matrix_scale(Matrix* self, float multiplier){
    for(int i = 0; i < self->rows; i++){
        for(int j = 0; j < self->cols; j++){
            self->data[i*self->cols + j] = self->data[i*self->cols + j] * multiplier;
        }
    }
}


float Matrix_sum(Matrix* self){
    float sum = 0;
    for(int i = 0; i < self->rows; i++){
        for(int j = 0; j < self->cols; j++){
            sum += self->data[i*self->cols + j];
        }
    }
    return sum;
}


float Matrix_sumrow(Matrix* self, int row){
    float sum = 0;
    for(int j = 0; j < self->cols; j++){
        sum += self->data[row * self->cols + j];
    }
    return sum;
}



float Matrix_sumcol(Matrix* self, int col){
    float sum = 0;
    for(int i = 0; i < self->rows; i++){
       sum += self->data[i*self->cols + col];
    }
    return sum;
}

void Matrix_transpose(Matrix* self){ 
    int rows = self->rows;
    int cols = self->cols;
    
    if(rows == cols){
        for(int i = 0; i < rows; i++){
            for(int j=i+1; j < cols; j++){
                float temp = self->data[i*cols +j];
                self->data[i*cols +j] = self->data[j*cols + i];
                self->data[j*cols +i] = temp;
            }
        }    
    }
    else{

        float* new_data = malloc(rows*cols * sizeof(float));
        if(!new_data) return;
        for(int i = 0; i < rows; i++){
            for(int j=0; j < cols; j++){
                new_data[j*rows + i] = self->data[i*cols + j];
            }
        }
        
        
    free(self->data);
    self->data = new_data;
    self->rows = cols;
    self->cols = rows;

    }
}


void Matrix_add(Matrix* self, Matrix* x){

    if(!(self->cols == x->cols)&&(self->rows == x->rows)){
        printf("ERROR with matrix add not the same rows and columns");
        return;
    }
   
    for(int i = 0; i < self->rows; i++){
        for(int j = 0; j < self->cols; j++){
            self->data[i*self->cols + j] = (self->data[i*self->cols + j]) + (x->data[i*x->cols +j]);
        }
    }
    
    
}



void Matrix_subtract(Matrix* self, Matrix* x){

    if(!(self->cols == x->cols)&&(self->rows == x->rows)){
        printf("ERROR with matrix add not the same rows and columns");
        return;
    }
   
    for(int i = 0; i < self->rows; i++){
        for(int j = 0; j < self->cols; j++){
            self->data[i*self->cols + j] = (self->data[i*self->cols + j]) - (x->data[i*x->cols +j]);
        }
    }
    
    
}


void Matrix_map(Matrix* self, float(*operation)(float)){
    for(int i = 0; i < (self->rows * self->cols); i++){
        self->data[i] = operation(self->data[i]);    
    }
    
    
}




void Matrix_hadamard(Matrix* self, Matrix* x){
 
    if(!(self->cols == x->cols)&&(self->rows == x->rows)){
        printf("ERROR with matrix add not the same rows and columns");
        return;
    }
   
    for(int i = 0; i < self->rows; i++){
        for(int j = 0; j < self->cols; j++){
            self->data[i*self->cols + j] = (self->data[i*self->cols + j]) * (x->data[i*x->cols +j]);
        }
    }
    
}





Matrix* Matrix_dot(Matrix* self, Matrix* x){

    int r1 = self->rows;
    int r2 = x->rows;
    int c1 = self->cols;
    int c2 = x->cols;

    if(c1 == r2){
        Matrix* ret = Matrix_make(self->rows, x->cols);
        
        for(int i = 0; i < r1; i++){
            for(int j = 0; j < c2; j++){

                float sum = 0;
                for(int k = 0; k < c1; k++){
                    sum+= (self->data[i*self->cols+k] * x->data[k*x->cols + j]);
                }
                
                ret->data[i*ret->cols + j] = sum;
            }   
        }

        return ret;
    }else{
        printf("improper dimensions");
    }

    return self;
}




void Matrix_copy(Matrix *dest, Matrix *src){
    if((dest->rows != src->rows) || (dest->cols != src->cols)){
        printf("ERROR: src does not match dest");
        return;
    }

    size_t total_bytes = src->rows * src->cols * sizeof(float);

    memcpy(dest->data, src->data, total_bytes);

}


Matrix* Matrix_clone(Matrix* self){
    Matrix* new_matrix = Matrix_make(self->rows, self->cols);
    Matrix_copy(new_matrix, self);
    return new_matrix;    

}


float Matrix_min(Matrix* self){
    float min = FLT_MAX;
    for(int i = 0; i < self->rows;i++){
        for(int j = 0; j < self->cols; j++){
            if(self->data[i*self->cols + j] < min){
                min = self->data[i*self->cols + j]; 
            }
        }
    }

    return min;
}



float Matrix_max(Matrix* self){
    float max = -FLT_MAX;
    for(int i = 0; i < self->rows;i++){
        for(int j = 0; j < self->cols; j++){
            if(self->data[i*self->cols + j] > max){
                max = self->data[i*self->cols + j]; 
            }
        }
    }

    return max;
}




MinMax Matrix_min_max(Matrix* self){
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for(int i = 0; i < self->rows;i++){
        for(int j = 0; j < self->cols; j++){
            float val = self->data[i*self->cols + j];
            if(val <  min){
                min = val; 
            }
            if(val > max){
                max = val; 
            }
            
        }
    }
    MinMax result;
    result.min = min;
    result.max = max;

    return result;
}





//0 for column search 1 for row search
int Matrix_argsmax(Matrix * self, int index, int dir){
    float max = -FLT_MAX;
    int idx = 0;
    if(dir == 0){
        for(int i = 0; i < self->rows;i++){
            if(self->data[i*self->cols + index] > max){
                max = self->data[i*self->cols + index];
                idx = i;
            }
        }
    }else{
        for(int i = 0; i < self->cols;i++){
            if(self->data[index*self->cols + i] > max){
                max = self->data[index*self->cols + i];
                idx = i;
            }
        }

    }
    return idx;
}



//0 for column search 1 for row search
int Matrix_argsmin(Matrix * self, int index, int dir){
    float min = FLT_MAX;
    int idx = 0;

    if(dir == 0){
        for(int i = 0; i < self->rows;i++){
            if(self->data[i*self->cols + index] < min){
                min = self->data[i*self->cols + index]; 
                idx = i;
            }
        }
    }else{
        for(int i = 0; i < self->cols;i++){
            if(self->data[index*self->cols + i] < min){
                min = self->data[index*self->cols + i]; 
                idx=i;
            }
        }

    }
    return idx;
}


MinMax Matrix_args_min_max(Matrix* self, int index, int dir){
    float min = FLT_MAX;
    float max = -FLT_MAX;

    int val;

    if(dir == 0){
        for(int i = 0; i < self->rows;i++){
            val = self->data[i*self->cols + index];
            if(val < min) min = val;
            if(val > max) max = val;
        }
    }else{
        for(int i = 0; i < self->cols;i++){
            val = self->data[index*self->cols + i]; 
            if(val < min) min = val;
            if(val > max) max = val;
        }
    }

    MinMax results;
    results.min = min;
    results.max = max;

    return results;

}



void Matrix_reshape(Matrix *self, int new_rows, int new_cols){
    self->cols = new_cols;
    self->rows = new_rows;
}



void Matrix_addon(Matrix* self, Matrix *added, int dir){

    int new_rows = self->rows;
    int new_cols = self->cols;

    if(dir == 0){
        if(self->rows != added->rows){
            printf("ERROR: not matching rows for addon");
            return;
        }
        new_cols += added->cols; 
        

    }else{
        if(self->cols != added->cols){    
            printf("ERROR: not matching cols for addon");
            return;
        }
        new_rows += added->rows;

    }

    
    float* new_data = (float*)malloc(new_rows * new_cols *sizeof(float));
    if (new_data == NULL) return;

    if(dir == 0){
        float* dest_ptr = new_data;
        float* self_ptr = self->data;
        float* add_ptr = added->data;

        size_t row_bytes = self->cols * sizeof(float);
        size_t new_row_bytes = added->cols *sizeof(float);

        for(int i=0; i< self->rows; i++){
            memcpy(dest_ptr, self_ptr, row_bytes);
            dest_ptr += self->cols;
            self_ptr += self->cols;

            memcpy(dest_ptr, add_ptr, new_row_bytes);
            dest_ptr += added->cols;
            add_ptr += added->cols;

        }

    } else {
         
        size_t self_bytes = self->rows * self->cols * sizeof(float);
        size_t added_bytes = added->rows * added->cols  *sizeof(float);
        
        memcpy(new_data, self->data, self_bytes);
        memcpy(new_data, added->data, added_bytes);

    }

    free(self->data);
    self->data = new_data;
    self->rows = new_rows;
    self->cols = new_cols;
}


































