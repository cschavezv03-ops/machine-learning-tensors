#include <stdio.h>
#include "Tensors.h"

//Small student data processor

int main(void) {

    NDArrayHandle grades = array(
    int32(10, 20, 30, 40, 50,
          60, 70, 80, 90, 100,
          110, 120, 130, 140, 150,
          160, 170, 180, 190, 200),
        shape(4, 5)
    );

    printf("\nRaw dataset (Student grades: rows -> students; columns -> activities/homeworks):\n%s\n",
        asstr(grades));

    //Select students 1 and 2, activites 0, 2, 4
    NDArrayHandle selection = slice(grades, shape(1, 3), shape(0, 5, 2));

    //Select students 1 and 2, kepping all activities
    NDArrayHandle students = slice_axis(grades, 0, 1, 3);

    //Create an independent copy
    NDArrayHandle modified = copy(grades);

    //Replace a region with 0 in the copy
    slice_assign(modified, 0, range(1, 3), range(0, 5, 2));

    printf("\nSELECTION:\n%s\n", asstr(selection));
    printf("Shape: %s\n", shape_str(selection);

    printf("\nSLICE AXIS:\n%s\n", asstr(students));
    printf("Shape: %s\n", shape_str(students));

    printf("\nMODIFIED:\n%s\n", asstr(modified));

    free_tensors(grades, selection, students, modified);

    return 0;
}