#include <stdio.h>
#include "Tensors.h"

int main(void) {
    
    NDArrayHandle A = array(
        int32(1, 2, 3, 4, 5, 6),
        shape(2, 3)
    );

    NDArrayHandle B = array(
        int32(7, 8, 9, 10, 11, 12),
        shape(2, 3)
    );

    //Row-wise concatenation
    NDArrayHandle C0 = concat(0, A, B);

    //Column-wise concatenation
    NDArrayHandle C1 = concat(1, A, B);

    //Stacking creating axis 0
    NDArrayHandle S0 = stack(0, A, B);

    //Stacking creating axis 1
    NDArrayHandle S1 = stack(1, A, B);

    printf("A:\n%s\n", asstr(A));
    printf("B:\n%s\n", asstr(B));

    printf("\nCONCAT AXIS 0:\n%s\n", asstr(C0));
    printf("Shape: %s\n", shape_str(C0));

    printf("\nCONCAT AXIS 1:\n%s\n", asstr(C1));
    printf("Shape: %s\n", shape_str(C1));

    printf("\nSTACK AXIS 0:\n%s\n", asstr(S0));
    printf("Shape: %s\n", shape_str(S0));

    printf("\nSTACK AXIS 1:\n%s\n", asstr(S1));
    printf("Shape: %s\n", shape_str(S1));

    return 0;
}