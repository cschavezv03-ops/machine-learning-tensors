#include <stdio.h>
#include "Tensors.h"

int main(void) {

    NDArrayHandle x = array(int32(10, 20, 30));

    NDArrayList list1 = atleast_1d(x);
    NDArrayList list2 = atleast_2d(x);
    NDArrayList list3 = atleast_3d(x);

    //One input generates one output en each list
    NDArrayHandle a = idx(list1, 0);
    NDArrayHandle b = idx(list2, 0);
    NDArrayHandle c = idx(list3, 0);

    printf("Original: %s\n", shape_str(x));
    printf("1D: %s\n", shape_str(a));
    printf("2D: %s\n", shape_str(b));
    printf("3D: %s\n", shape_str(c));

    printf("\n3D tensor content:\n%s\n", asstr(c));

    //The lists administrate their own results
    free_tensors(list1, list2, list3, x);
    return 0;
}