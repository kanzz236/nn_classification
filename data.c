
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "data.h"

int load_csv(const char* filename, SensorData dataset[]) {
    FILE* file = fopen(filename, "r");

    if (!file) {
        printf("Loi: Khong the mo file '%s'!\n", filename);
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), file);

    int count = 0;

    while (fgets(line, sizeof(line), file) &&
           count < MAX_SAMPLES) {

        line[strcspn(line, "\r\n")] = 0;

        char* token = strtok(line, ",");
        if (!token) continue;

        strtok(NULL, ","); // UDI
        strtok(NULL, ","); // Product ID

        int valid = 1;

        for (int i = 0; i < 5; i++) {
            token = strtok(NULL, ",");

            if (!token) {
                valid = 0;
                break;
            }

            dataset[count].raw_features[i] = atof(token);
        }

        if (!valid) continue;

        token = strtok(NULL, ",");

        if (token) {
            dataset[count].target = atoi(token);
            count++;
        }
    }

    fclose(file);
    return count;
}

void shuffle_dataset(SensorData dataset[], int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        SensorData temp = dataset[i];
        dataset[i] = dataset[j];
        dataset[j] = temp;
    }
}

void feature_engineering_and_scaler(
    SensorData train[], int train_size,
    SensorData val[], int val_size,
    SensorData test[], int test_size
) {
    SensorData* all_data[3] = {
        train, val, test
    };

    int sizes[3] = {
        train_size, val_size, test_size
    };

    // Feature Engineering
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < sizes[s]; i++) {

            for (int j = 0; j < 5; j++) {
                all_data[s][i].features[j] =
                    all_data[s][i].raw_features[j];
            }

            // Temperature Difference
            all_data[s][i].features[5] =
                all_data[s][i].raw_features[1] -
                all_data[s][i].raw_features[0];

            // Mechanical Power
            all_data[s][i].features[6] =
                all_data[s][i].raw_features[2] *
                all_data[s][i].raw_features[3];
        }
    }

    // Calculate Mean and Std using TRAIN only
    double means[NUM_FEATURES] = {0};
    double stds[NUM_FEATURES] = {0};

    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            means[j] += train[i].features[j];
        }
    }

    for (int j = 0; j < NUM_FEATURES; j++) {
        means[j] /= train_size;
    }

    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            double diff = train[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }

    for (int j = 0; j < NUM_FEATURES; j++) {
        stds[j] = sqrt(stds[j] / train_size);
    }

    // Z-score normalization
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < sizes[s]; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {

                if (stds[j] != 0) {
                    all_data[s][i].features[j] =
                        (all_data[s][i].features[j] - means[j])
                        / stds[j];
                }
            }
        }
    }
}
