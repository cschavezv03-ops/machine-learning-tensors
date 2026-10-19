
#include <stdio.h>
#include "Tensors.h"

int main(void) {

    NDArrayHandle A = array(
        int32(2, 4, 6, 3, 5, 8),
        shape(2, 3)
    );

    NDArrayHandle B = lcm(A, 6);

    printf("=== ORIGINAL TENSOR ===\n");
    printf("A:\n%s\n", asstr(A));
    printf("dtype A: %s\n", dtype(A));
    printf("shape A: %s\n\n", shape_str(A));

    printf("=== LCM RESULT ===\n");
    printf("B:\n%s\n", asstr(B));
    printf("dtype B: %s\n", dtype(B));
    printf("shape B: %s\n", shape_str(B));

    free_tensors(A, B);

    return 0;
}
