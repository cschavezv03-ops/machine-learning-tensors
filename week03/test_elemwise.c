
#include <stdio.h>
#include "Tensors.h"

int main(void) {

    NDArrayHandle A = array(
        float32(1, 2, 3, 4, 5, 6),
        shape(2, 3)
    );

    NDArrayHandle B = array(
        float32(10, 20, 30),
        shape(3)
    );

    NDArrayHandle C = array(
        float32(10, 20, 30, 10, 20, 30),
        shape(2, 3)
    );

    //Test 1: broadcasting
    NDArrayHandle R1 = add(A, B);

    //Test 2: same shapes
    NDArrayHandle R2 = elemwise_add(A, C);

    printf("R1 = add(A, B):\n%s\n\n", asstr(R1));
    printf("R2 = elemwise_add(A, C):\n%s\n", asstr(R2));

    free_tensors(A, B, C, R1, R2);

    return 0;
}
