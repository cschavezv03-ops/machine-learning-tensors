#include <stdio.h>
#include "Tensors.h"

int main(void) {

    NDArrayHandle A = array(
        Bool(0, 1, 1, 1, 0, 1),
        shape(2, 3)
    );

    NDArrayHandle B = array(
        Bool(1, 0, 0, 1, 1, 0),
        shape(2, 3)
    );

    NDArrayHandle R1 = not(A);
    NDArrayHandle R2 = and(A, B);
    NDArrayHandle R3 = or(A, B);
    NDArrayHandle R4 = xor(A, B);

    printf("NOT:\n%s\n", asstr(R1));
    printf("dtype: %s\n\n", dtype(R1));

    printf("AND:\n%s\n", asstr(R2));
    printf("dtype: %s\n\n", dtype(R2));

    printf("OR:\n%s\n", asstr(R3));
    printf("dtype: %s\n\n", dtype(R3));

    printf("XOR:\n%s\n", asstr(R4));
    printf("dtype: %s\n\n", dtype(R4));

    free_tensors(A, B, R1, R2, R3, R4);

    return 0;
}