#ifndef TENSOR_H
#define TENSOR_H

#define MAX_DIMS 4


typedef struct{
    float* data;
    int ndim;
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int total_elements;
}Tensor;

typedef struct{
    float min;
    int min_loc;
    float max;
    int max_loc;
}MinMax;



Tensor* Tensor_make(int ndim, int* shape);
void Tensor_free(Tensor* self);
float Tensor_at(Tensor* self, int* coords);

void Tensor_print(Tensor* self);

void Tensor_zero(Tensor* self);
void Tensor_rand(Tensor* self);
void Tensor_scale(Tensor* self, float multiplier);
float Tensor_sum(Tensor* self);
void Tensor_sum_axis(const Tensor* self, Tensor* dest, int axis);



void Tensor_copy(const Tensor* src, Tensor* dest);// use memcopy
Tensor* Tensor_clone(const Tensor* self);

int Tensor_matches_shape(const Tensor* t, int ndim, const int* shape);

MinMax Tensor_min_max(Tensor* self);
void Tensor_argsmax(const Tensor* src, Tensor* dest, int axis);

void Tensor_argsmin(const Tensor* src, Tensor* dest, int axis);

void Tensor_map_axis(Tensor* self, float(*operation)(float, int), int axis);

void Tensor_map(Tensor* self, float(*operation)(float));

void Tensor_reshape(Tensor* self, int* new_dims, int new_size);

void Tensor_hadamard(Tensor* self, Tensor* new);

void Tensor_lazy_transpose(Tensor* self, int idx_from, int idx_to);

void Tensor_matmul(Tensor* a, Tensor* b, Tensor* dest);

Tensor* Tensor_eager_transpose(Tensor* self, int idx_from, int idx_to);

float Tensor_vector_dot(Tensor* a, Tensor* b);

Tensor* Tensor_contract(const Tensor* a, const Tensor* b, int axis_a, int axis_b);

Tensor* Tensor_append(Tensor* self, Tensor* addon, int axis);


void Tensor_vector_addition(Tensor* a, Tensor* b);

#endif
