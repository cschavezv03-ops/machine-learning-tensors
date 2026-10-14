#include <stdio.h>
#include "Tensors.h"

int main(void) {

    //Neuronal layer dummy inputs
    NDArrayHandle x = array(float32(-3, -1, 2, 5), shape(1, 4));

    //Received gradient from the next layer
    NDArrayHandle grad = ones(shape(1, 4));

    //ReLU
    NDArrayHandle y_relu = relu(x);

    //ReLU backward
    NDArrayHandle g_relu = _relu_backward(grad, x);

    //Sigmoid
    NDArrayHandle y_sigmoid = sigmoid(x);

    //Sigmoid backward
    //It receives y_signmoid, not x
    NDArrayHandle g_sigmoid = _sigmoid_backward(grad, y_sigmoid);

    //Softmax along the columns
    NDArrayHandle y_softmax = softmax(x, 1);

    printf("INPUT:\n%s\n", asstr(x));

    printf("\nRELU:\n%s\n", asstr(y_relu));

    printf("\nRELU BACKWARD:\n%s\n",
           asstr(g_relu));

    printf("\nSIGMOID:\n%s\n", asstr(y_sigmoid));

    printf("\nSIGMOID BACKWARD:\n%s\n",
           asstr(g_sigmoid));

    printf("\nSOFTMAX:\n%s\n", asstr(y_softmax));

    printf("\nSHAPE SOFTMAX: %s\n",
           shape_str(y_softmax));

    free_tensors(x, grad, y_relu, g_relu, y_sigmoid, g_sigmoid, y_softmax);
    
    return 0;
}