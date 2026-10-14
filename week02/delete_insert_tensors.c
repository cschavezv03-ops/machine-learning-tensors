#include <stdio.h>
#include "Tensors.h"

int main(void) {

    NDArrayHandle x = array(int32(10, 20, 30, 40, 50));

    //Delete the 2nd index element
    NDArrayHandle eliminated = delete(x, 2);

    //Insert 99 before index 2
    NDArrayHandle scalar = insert_scalar(x, 2, 99);

    //Values we want to insert
    NDArrayHandle new = array(int32(77, 88));

    //Positions to insert
    NDArrayHandle positions = array(int64(1, 3));

    NDArrayHandle inserted_tensor = insert_tensor(x, new, positions);

    //Values to insert through a range
    NDArrayHandle values_range = array(int32(100, 200));

    NDArrayHandle inserted_range = insert_slice(x, values_range, 1, 3, 2);
    
    printf("ORIGINAL:\n%s\n", asstr(x));

    printf("\nDELETE:\n%s\n",
           asstr(eliminated));

    printf("\nINSERT SCALAR:\n%s\n",
           asstr(scalar));

    printf("\nINSERT TENSOR:\n%s\n",
           asstr(inserted_tensor));

    printf("\nINSERT SLICE:\n%s\n",
           asstr(inserted_range));

    printf("\nSHAPES:\n");
    printf("Original: %s\n", shape_str(x));
    printf("Delete: %s\n", shape_str(eliminated));
    printf("Insert scalar: %s\n", shape_str(scalar));
    printf("Insert tensor: %s\n",
           shape_str(inserted_tensor));
    printf("Insert slice: %s\n",
           shape_str(inserted_range));

    free_tensors(x, eliminated, scalar,
        new, positions, inserted_tensor,
        values_range, inserted_range);

    return 0;
}