
#ifndef EVALUATION_H
#define EVALUATION_H

#include "data.h"
#include "neural_network.h"

double find_best_threshold(
    NeuralNetwork* nn,
    SensorData val[],
    int val_size
);

void evaluate_model(
    NeuralNetwork* nn,
    SensorData data[],
    int size,
    double threshold
);

#endif
