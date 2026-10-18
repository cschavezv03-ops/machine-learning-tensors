#include <stdio.h>
#include "Tensors.h"

int main(void) {
    
    NDArrayHandle A = array(float32(1, 2, 3, 4, 5, 6), shape(2, 3));
    
    NDArrayHandle B = array(float32(10, 20, 30, 40, 50, 60), shape(2, 3));

    //Addition between A and B
    NDArrayHandle C = add(A, B);
    
    //Multiplication with a scalar (broadcasting)
    NDArrayHandle D = mul(A, 2);
    
    //Adding an scalar (broadcasting)
    NDArrayHandle E = add(10, A);

    printf("A:\n%s\n\n", asstr(A));
    printf("B:\n%s\n\n", asstr(B));

    printf("A + B:\n%s\n\n", asstr(C));
    printf("A * 2:\n%s\n\n", asstr(D));
    printf("10 + A:\n%s\n\n", asstr(E));

    free_tensors(A, B, C, D, E);

    return 0;
}