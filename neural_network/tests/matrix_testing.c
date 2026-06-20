#include "../libaries/matrix.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

float square_element(float val){
    return val * val;
}

int main(){

    srand(time(NULL));

    printf("---------Test 1: creation and randomization -----------\n");
    Matrix* m1 = Matrix_make(2,3);
    Matrix_rand(m1);
    printf("Matrix m1 (2x3 Random): \n\n");
    Matrix_print(m1);


    printf("---------Test 2: cloning and scaling ------------------\n");
    Matrix* m2 = Matrix_clone(m1);
    Matrix_scale(m2, 2.0f);
    printf("Matrix m2 (m1 scaled by 2): \n\n");
    Matrix_print(m2);


    printf("---------Test 4: Matrix Transposition ----------------\n");
    printf("Matrix m1 before transpose %dx%d \n", m1->rows, m1->cols);
    Matrix_print(m1);
    Matrix_transpose(m1);
    printf("Matrix m1 transposed to %dx%d \n\n", m1->rows, m1->cols);
    Matrix_print(m1);



    printf("---------Test 5: Dot products and multiplication -------\n");
    Matrix* dot_result = Matrix_dot(m1, m2);
    if(dot_result){    
        printf("Dot product result(m1 * m2) [3x3]: \n");
        Matrix_print(dot_result);
    }


    printf("---------Test 6: Reductions (min/max/sum/argsmax-min)) --\n");
    Matrix* m3 = Matrix_make(2,2);
    m3->data[0] = 5.0f; m3->data[1] = 12.0f; 
    m3->data[2] = 20.0f; m3->data[3] = -3.0f;
    printf("matrix m3: \n");
    Matrix_print(m3);

    printf("Global Max: %.2f\n", Matrix_max(m3));
    printf("Global Min: %.2f\n", Matrix_min(m3));
    printf("sum row 0: %.2f\n", Matrix_sumrow(m3,0));
    printf("sum col 1: %.2f\n", Matrix_sumcol(m3,1));
    printf("ArgsMax of row 1 index across columns: %d\n", Matrix_argsmax(m3, 1, 1));
    printf("ArgsMin of column 0 index across rows: %d\n", Matrix_argsmin(m3, 0, 0));


    printf("---------Test 7: Reshape and Addition ------------------\n");
    Matrix_reshape(m3, 1, 4);
    printf("matrix m3 reshaped to 1x4\n");
    Matrix_print(m3);

    Matrix* m4 = Matrix_make(1,2);
    m4->data[0]= 100.0f;m4->data[1] = 200.0f;
    printf("m4 to append to m3: \n");
    Matrix_print(m4);
    Matrix_addon(m3, m4, 0);
    printf("m3 after side addon (should be 1x6): \n");
    Matrix_print(m3);


    Matrix_free(m1);
    Matrix_free(m2);
    Matrix_free(m3);
    Matrix_free(m4);

    if(dot_result) Matrix_free(dot_result);

    printf("\n All tests returned without complation or runtime errors");
    return 0;

}














