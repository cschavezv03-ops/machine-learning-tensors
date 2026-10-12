#include <stdio.h>
#include "Tensors.h"

int main(void) {

    //Labels of a dataset of 6 samples
    NDArrayHandle labels = array(
        int32(2, 1, 2, 0, 1, 2)
    );

    //Request the four exits
    NDArrayList result = unique(labels, 1, 1, 1);

    printf("LABELS:\n%s\n", asstr(labels));

    printf("\nUNIQUE VALUES:\n%s\n",
           asstr(idx(result, 0)));

    printf("\nFIRST APPEARANCE:\n%s\n",
           asstr(idx(result, 1)));

    printf("\nREVERSE INDEXES:\n%s\n",
           asstr(idx(result, 2)));

    printf("\nFREQUENCIES:\n%s\n",
           asstr(idx(result, 3)));

    free_tensors(labels, result);

    return 0;
}