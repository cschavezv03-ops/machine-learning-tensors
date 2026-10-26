#include <stdio.h>
#include "Tensors.h"

int main(void){

    NDArrayHandle tensor = array(int32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), shape(2, 3, 2));
    printf("Tensor: %s\n", asstr(tensor));

    NDArrayHandle t_tensor = transpose(tensor, axis(1, 0, 2));
    printf("Transpose: %s\n", asstr(t_tensor));
    return 0;
}
