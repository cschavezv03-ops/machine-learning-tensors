#include <stdio.h>
#include "Tensors.h"

int main(void) {
    //Create a set of two images
    NDArrayHandle images = array(
        int32(10, 20, 30, 40, 50, 60, 70,
        80, 90, 100, 110, 120), shape(2, 2, 3)
    );

    //Flat every image, kepping the batch
    NDArrayHandle flat = flatten(images);

    //Reorganize all images like a 3x2 matrix
    NDArrayHandle reorganized = reshape(
        images, shape(2, 3, 2)
    );

    //Add an axis for the channels
    NDArrayHandle with_channel = expand_dims(images, 3);

    //Delete size 1 axes
    NDArrayHandle recovered = squeeze(with_channel);

    printf("Original:           %s\n", shape_str(images));
    printf("Flatten:            %s\n", shape_str(flat));
    printf("Reshape:            %s\n", shape_str(reorganized));
    printf("Expand dims:        %s\n", shape_str(with_channel));
    printf("Squeeze:            %s\n", shape_str(recovered));

    free_tensors(images, flat, reorganized, with_channel, recovered);
    return 0;
}