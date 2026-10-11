#include <stdio.h>
#include "Tensors.h"

int main(void) {
     NDArrayHandle images = array(
        int32(10, 20, 30, 40, 50, 60, 70,
        80, 90, 100, 110, 120), shape(2, 2, 3)
     );
     
     //Prepare images for a dense layer
     NDArrayHandle flat = flatten(images);
     
     //Add a channel dimension
     NDArrayHandle with_channel = expand_dims(images, -1);

     //Add a dimension before the batch
     NDArrayHandle new_batch = expand_dims(images, 0);

     //Delete the added dimension
     NDArrayHandle recovered = squeeze(with_channel, axis(-1));

     printf("Original:           %s\n", shape_str(images));
     printf("Flatten:            %s\n", shape_str(flat));
     printf("With channel:       %s\n", shape_str(with_channel));
     printf("New batch:          %s\n", shape_str(new_batch));
     printf("Recovered:          %s\n", shape_str(recovered));
     printf("\nFlat images content:\n%s\n", asstr(flat));

     free_tensors(images, flat, with_channel, new_batch, recovered);
     return 0;

}
