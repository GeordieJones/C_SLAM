#ifndef MATRIX_H
#define MATRIX_H


typedef struct{
    int rows;
    int cols;
    float* data;
} Matrix;

typedef struct{
    float min;
    float max;
}MinMax;

Matrix* Matrix_make(int rows, int cols);
void Matrix_print(Matrix* self);
void Matrix_free(Matrix* self);
void Matrix_zero(Matrix* self);
void Matrix_rand(Matrix* self);
void Matrix_scale(Matrix* self, float multiplier);
float Matrix_sum(Matrix* self);
float Matrix_sumrow(Matrix* self, int row);
float Matrix_sumcol(Matrix* self, int col);
void Matrix_transpose(Matrix* self);
void Matrix_subtract(Matrix* self, Matrix* x);
void Matrix_add(Matrix* self, Matrix* x);
void Matrix_map(Matrix* self, float(*operation)(float));
void Matrix_hadamard(Matrix* self, Matrix* x);
Matrix* Matrix_dot(Matrix* self, Matrix* x);

void Matrix_reshape(Matrix *self, int new_row, int new_col);
void Matrix_addon(Matrix *self, Matrix *added, int dir);


void Matrix_copy(Matrix* dest, Matrix* src);// use memcopy
Matrix* Matrix_clone(Matrix* self);


MinMax Matrix_min_max(Matrix* self);
MinMax Matrix_args_min_max(Matrix* self, int index, int dir);
float Matrix_min(Matrix *self);
float Matrix_max(Matrix *self);
int Matrix_argsmax(Matrix *self, int index, int dir);// for the given row or column
int Matrix_argsmin(Matrix *self, int index, int dir);



#endif

