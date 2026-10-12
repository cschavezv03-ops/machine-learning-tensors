#include <stdio.h>
#include "Tensors.h"

int main(void) {
    
    NDArrayHandle grades = array(
        int32(10, 20, 30,
              40, 50, 60,
              70, 80, 90,
              100, 110, 120),
        shape(4, 3)
    );

    //Students arbitrary indices
    NDArrayHandle indices = array(int32(3, 0, 2));

    //Dataset row selection
    NDArrayHandle selection = take(grades, indices, 0);

    printf("Original dataset:\n%s\n", asstr(grades));

    printf("\nSelected rows:\n%s\n", asstr(selection));

    printf("\nShape: %s\n", shape_str(selection));

    free_tensors(grades, selection, indices);

    return 0;
}