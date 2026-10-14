#include <stdio.h>
#include "Tensors.h"

int main(void) {

    //2 samples, 3 characteristics each one
    NDArrayHandle X = array(
        float32(1, 2, 3,
                4, 5, 6),
        shape(2, 3)
    );

    //Weights: 3 inputs, 2 neurons
    NDArrayHandle W = array(
        float32(1, 2,
                3, 4,
                5, 6),
        shape(3, 2)
    );

    //Neuronal layer output
    NDArrayHandle Y = einsum(
        "ij,jk->ik", 0, X, W
    );

    printf("INPUTS:\n%s\n", asstr(X));
    printf("WEIGHTS:\n%s\n", asstr(W));
    
    printf("\nOUTPUTS:\n%s\n", asstr(Y));
    printf("Shape: %s\n", shape_str(Y));
    
    //SVD decomposition of Y
    NDArrayList r = svd(Y);

    printf("\nU:\n%s\n", asstr(idx(r, 0)));
    printf("\nS:\n%s\n", asstr(idx(r, 1)));
    printf("\nVT:\n%s\n", asstr(idx(r, 2)));

    free_tensors(X, W, Y, r);

    return 0;
}