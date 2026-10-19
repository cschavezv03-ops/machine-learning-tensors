
#include <stdio.h>
#include "Tensors.h"

int main(void) {

    // Case 1: Tensor int32
    NDArrayHandle A = array(
        int32(2, 4, 5, 8),
        shape(2, 2)
    );

    // Case 2: Tensor float32
    NDArrayHandle B = array(
        float32(2, 4, 5, 8),
        shape(2, 2)
    );

    // Case 3: Tensor float64
    NDArrayHandle C = array(
        float64(2, 4, 5, 8),
        shape(2, 2)
    );

    // Divisions
    NDArrayHandle R1 = div(20, A);
    NDArrayHandle R2 = div(20, B);
    NDArrayHandle R3 = div(20, C);

    printf("=== CASE 1: INT32 ===\n");
    printf("dtype: %s\n", dtype(R1));
    printf("Result:\n%s\n\n", asstr(R1));

    printf("=== CASE 2: FLOAT32 ===\n");
    printf("dtype: %s\n", dtype(R2));
    printf("Result:\n%s\n\n", asstr(R2));

    printf("=== CASE 3: FLOAT64 ===\n");
    printf("dtype: %s\n", dtype(R3));
    printf("Result:\n%s\n\n", asstr(R3));

    free_tensors(A, B, C, R1, R2, R3);

    return 0;
}