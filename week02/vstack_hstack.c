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

    //Combine horizontally and vertically
    NDArrayHandle H = hstack(A, B);
    NDArrayHandle V = vstack(A, B);

    printf("HSTACK:\n%s\n", asstr(H));
    printf("Shape: %s\n", shape_str(H));

    printf("\nVSTACK:\n%s\n", asstr(V));
    printf("Shape: %s\n", shape_str(V));

    //Divide the horizontally result in 2 parts
    NDArrayList parts = split(H, 2, 1);

    //Access the tensors within the list
    NDArrayHandle p0 = idx(parts, 0);
    NDArrayHandle p1 = idx(parts, 1);

    printf("\nPART 0:\n%s\n", asstr(p0));
    printf("Shape: %s\n", shape_str(p0));

    printf("\nPART 1:\n%s\n", asstr(p1));
    printf("Shape: %s\n", shape_str(p1));

    //The list is proprietary of its elements
    //Do not free p0 and p1 separately
    free_tensors(A, B, H, B, parts);

    return 0;
}