#include <stdio.h>
#include "Tensors.h"

int main(void) {

    //DATASET: 3 rows and 4 columns images
    NDArrayHandle images = array(
        int32(1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12,
              13, 14, 15, 16,
              17, 18, 19, 20,
              21, 22, 23, 24),
        shape(2, 3, 4)
    );

    //Operation 1: scalar index
    NDArrayHandle a = slice(images, -1, ":", ":");

    //Operation 2: range of an element
    NDArrayHandle b = slice(images, range(1, 2), ":", ":");

    //Operation 3: last image columns 1 and 3 selection
    NDArrayHandle c = slice(images, -1, ":", range(1, 4, 2));

    //Operation 4: selection of the two first rows of each image
    //NDArrayHandle d = slice(images, ":", range(0, 2), ":"); -> same as below but with slice function
    NDArrayHandle d = slice_axis(images, 1, 0, 2);

    //Operation 5: select the images in reverse order
    NDArrayHandle indices = array(int32(1, 0));

    NDArrayHandle e = take(images, indices, 0);

    printf("ORIGINAL:\n%s\n", asstr(images));

    printf("\nA: %s\n%s\n", shape_str(a), asstr(a));
    printf("\nB: %s\n%s\n", shape_str(b), asstr(b));
    printf("\nC: %s\n%s\n", shape_str(c), asstr(c));
    printf("\nD: %s\n%s\n", shape_str(d), asstr(d));
    printf("\nE: %s\n%s\n", shape_str(e), asstr(e));

    free_tensors(
        images, a, b, c, d, indices, e
    );

    return 0;
}