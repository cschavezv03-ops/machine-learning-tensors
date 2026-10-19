#include <stdio.h>
#include "Tensors.h"

int main(void) {

    NDArrayHandle A = array(float32(1, 2, 3, 4, 5, 6), shape(2, 3));

    NDArrayHandle B = array(float32(10, 20, 30), shape(3));

    NDArrayHandle C = add(A, B);

    printf("Shape A: %s\n", shape_str(A));
    printf("Shape B: %s\n", shape_str(B));
    printf("Resultado add(A, B):\n%s\n", asstr(C));

    free_tensors(A, B, C);

    return 0;
}