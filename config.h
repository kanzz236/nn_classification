#ifndef CONFIG_H
#define CONFIG_H

#define MAX_SAMPLES 10000
#define NUM_FEATURES 7

// Neural Network Architecture
#define H1 64
#define H2 32

// Training
#define EPOCHS 500
#define BATCH_SIZE 32

// Hyperparameters
#define LEARNING_RATE 0.0012
#define MOMENTUM 0.90
#define POS_WEIGHT 2.20
#define L2_LAMBDA 0.00015

#endif
