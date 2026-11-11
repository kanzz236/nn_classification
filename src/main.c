
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "config.h"
#include "data.h"
#include "neural_network.h"
#include "training.h"
#include "evaluation.h"

int main() {

    srand(42);

    // 1. Load Dataset
    SensorData* dataset =
        malloc(sizeof(SensorData) * MAX_SAMPLES);

    if (!dataset) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    int total_samples = load_csv("ai4i2020.csv", dataset);

    printf(
        "Da tai %d dong du lieu "
        "(7 Features | Seed=42 | Architecture=64-32)\n",
        total_samples
    );

    if (total_samples == 0) {
        free(dataset);
        return 1;
    }

    // 2. Shuffle Dataset
    shuffle_dataset(dataset, total_samples);

    // 3. Split Dataset
    int train_size = (int)(total_samples * 0.70);
    int val_size = (int)(total_samples * 0.15);
    int test_size = total_samples - train_size - val_size;

    SensorData* train_data =
        malloc(sizeof(SensorData) * train_size);

    SensorData* val_data =
        malloc(sizeof(SensorData) * val_size);

    SensorData* test_data =
        malloc(sizeof(SensorData) * test_size);

    if (!train_data || !val_data || !test_data) {
        printf("Memory allocation failed!\n");

        free(dataset);
        free(train_data);
        free(val_data);
        free(test_data);

        return 1;
    }

    for (int i = 0; i < train_size; i++)
        train_data[i] = dataset[i];

    for (int i = 0; i < val_size; i++)
        val_data[i] = dataset[train_size + i];

    for (int i = 0; i < test_size; i++)
        test_data[i] =
            dataset[train_size + val_size + i];

    // 4. Feature Engineering & Scaling
    feature_engineering_and_scaler(
        train_data, train_size,
        val_data, val_size,
        test_data, test_size
    );

    // 5. Initialize Neural Network
    NeuralNetwork nn;
    initialize_network(&nn);

    // 6. Training
    printf("\n========== TRAINING ==========\n");

    train_network(&nn, train_data, train_size);

    // 7. Validation
    printf("\n========== VALIDATION ==========\n");

    double best_threshold =
        find_best_threshold(&nn, val_data, val_size);

    evaluate_model(
        &nn, val_data, val_size, best_threshold
    );

    // 8. Testing
    printf("\n========== TEST ==========\n");

    evaluate_model(
        &nn, test_data, test_size, best_threshold
    );

    // 9. Free Memory
    free(dataset);
    free(train_data);
    free(val_data);
    free(test_data);

    return 0;
}
