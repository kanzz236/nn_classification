
#ifndef DATA_H
#define DATA_H

#include "config.h"

typedef struct {
    double raw_features[5];
    double features[NUM_FEATURES];
    int target;
} SensorData;

int load_csv(const char* filename, SensorData dataset[]);

void shuffle_dataset(SensorData dataset[], int size);

void feature_engineering_and_scaler(
    SensorData train[], int train_size,
    SensorData val[], int val_size,
    SensorData test[], int test_size
);

#endif
