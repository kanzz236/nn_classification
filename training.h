
#ifndef TRAINING_H
#define TRAINING_H

#include "data.h"
#include "neural_network.h"

double weighted_bce_loss(double y, double y_hat);

void train_network(
    NeuralNetwork* nn,
    SensorData train[],
    int train_size
);

#endif
