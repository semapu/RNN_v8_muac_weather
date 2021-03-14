"""
Author: Sergi Mas Pujol
Last update: 18/12/2020

Python version: 3.6
"""

import numpy as np
import sys


def updated_detect_regulations_binary_mean_std(model, X_test, y_test, threshold, batch_size_test):
    """
    It computes the mean/std accuracy, recall, and precision using batches and having group de labels/preditions
     using "threshold"

    Args:
        model: [keras.models]
        X_test: [array] [samples, timesteps, features]
        y_test: [array] [samples, timesteps, features]
        threshold: [int] Minimum number of positive labels to consider a samples as positive
        batch_size_test: [int] number of sample used to compute the mean/std values

    Returns:
        np.mean(accuracy), np.std(accuracy), np.mean(recall), np.std(recall), np.mean(precision), np.std(precision)
    """

    # Initialize the final variables
    accuracy = []
    recall = []
    precision = []

    num_samples_set = X_test.shape[0]

    for _ in range(int(2 * np.floor(num_samples_set / batch_size_test))):
        start_set_random = np.random.randint(num_samples_set - batch_size_test)
        predictions = np.round(model.predict(X_test[start_set_random:start_set_random + batch_size_test, :, :]))
        labels = y_test[start_set_random:start_set_random + batch_size_test, :, :]

        # Initialize the internal variables
        TP, TN, FP, FN = 0, 0, 0, 0

        for prediction, label in zip(predictions, labels):
            # Group the predictions adn the label using "threshold"
            sum_prediction = np.sum(prediction)
            sum_test_sample = np.sum(label)

            if sum_test_sample <= threshold and sum_prediction <= threshold:
                TN += 1
            elif not sum_test_sample > threshold and sum_prediction >= threshold:
                FP += 1
            elif not sum_test_sample < threshold and sum_prediction <= threshold:
                FN += 1
            elif sum_test_sample >= threshold and sum_test_sample >= threshold:
                TP += 1

        accuracy.append((TP + TN) / (TP + FP + FN + TN))
        recall.append(TP / (TP + FN))
        precision.append(TP / (TP + FP))

    return np.mean(accuracy), np.std(accuracy), np.mean(recall), np.std(recall), np.mean(precision), np.std(precision)


def confusion_matrix_sequencial_output_mean_std(model, X_test, y_test, Tx, batch_size_test):
    """
    Compute the mean/std accuracy, recall, and precision using batches.py

    Args:
        model: [keras.models]
        X_test: [array] [samples, timesteps, features]
        y_test: [array] [samples, timesteps, features]
        Tx: [int] Number of timesteps in each sample
        batch_size_test: [int] number of sample used to compute the mean/std values

    Returns:
        np.mean(accuracy), np.std(accuracy), np.mean(recall), np.std(recall), np.mean(precision), np.std(precision)
    """

    # Initialize the final variables
    accuracy = []
    recall = []
    precision = []

    # Generate predictions for a random set
    num_samples_set = X_test.shape[0]

    for _ in range(int(2 * np.floor(num_samples_set / batch_size_test))):
        start_set_random = np.random.randint(num_samples_set - batch_size_test)
        predictions = np.round(model.predict(X_test[start_set_random:start_set_random + batch_size_test, :, :]))
        labels = y_test[start_set_random:start_set_random + batch_size_test, :, :]

        # Initialize the internal variables
        TP, TN, FP, FN = 0, 0, 0, 0

        for prediction, label in zip(predictions, labels):
            for i in range(Tx):
                # Analyse the prediction of the model
                # The labels fo all five samples are the same. Therefore, we will use the last label iterated
                if label[i] == 0 and prediction[i] == 0:  # No regulation & Pred. no regulation
                    TN += 1
                elif label[i] == 0 and prediction[i] == 1:  # No regulation & Pred. regulation
                    FP += 1
                elif label[i] == 1 and prediction[i] == 0:  # Regulation & Pred. no regulation
                    FN += 1
                elif label[i] == 1 and prediction[i] == 1:  # Regulation & Pred. regulation
                    TP += 1

        accuracy.append((TP + TN) / (TP + FP + FN + TN))
        recall.append(TP / (TP + FN))
        precision.append(TP / (TP + FP))

    return np.mean(accuracy), np.std(accuracy), np.mean(recall), np.std(recall), np.mean(precision), np.std(precision)


def updated_detect_regulations_binary(model, X_test, y_test, threshold):
    """
    Analyses the predictions returned by the model. It allows us to know if there are sample close to the
    target but not exactly equal.

    Args:
        model: keras model
        X_test: Input samples
        y_test: Labels of the input samples
        threshold: Minimum number of positive labels to consider a samples as positive

    Returns:
        correct: Samples with correct prediction in each timestep
        semi_correct: Sample with more or equal "percentage" of correct predictions
        incorrect: Percentage of samples with more than "percentage" number of predictions different
                   than the ground-truth.
    """

    # Initialize the output variables
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Generate all the predictions
    predictions = np.round(model.predict(X_test))

    for prediction, label in zip(predictions, y_test):

        sum_prediction = np.sum(prediction)
        sum_test_sample = np.sum(label)

        if sum_test_sample <= threshold and sum_prediction <= threshold:
            TN += 1
        elif not sum_test_sample > threshold and sum_prediction >= threshold:
            FP += 1
        elif not sum_test_sample < threshold and sum_prediction <= threshold:
            FN += 1
        elif sum_test_sample >= threshold and sum_test_sample >= threshold:
            TP += 1

    # Ordering the values to create the confusion matrix
    conf_matrix = np.zeros((2, 2))
    conf_matrix[0, 0] = TP
    conf_matrix[1, 1] = TN
    conf_matrix[0, 1] = FP
    conf_matrix[1, 0] = FN

    return TP, FP, TN, FN, conf_matrix


def detect_regulations_binary(model, X_test, y_test):
    """
    Analyses the predictions returned by the model. It allows us to know if there are sample close to the
    target but not exactly equal.

    Args:
        model: keras model
        X_test: Input samples
        y_test: Labels of the input samples

    Returns:
        correct: Samples with correct prediction in each timestep
        semi_correct: Sample with more or equal "percentage" of correct predictions
        incorrect: Percentatge of samples with more than "percentage" number of predictions different
                   than the ground-truth.
    """

    # Initialize the output variables
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Generate all the predictions
    predictions = np.round(model.predict(X_test))

    for prediction, label in zip(predictions, y_test):

        sum_prediction = np.sum(prediction)
        sum_test_sample = np.sum(label)

        if sum_test_sample == 0 and sum_prediction == 0:  # No hotspots & No hotspot
            TN += 1
        elif sum_test_sample == 0 and sum_prediction >= 10:  # >= 5 or >= 10
            FP += 1
        elif sum_prediction >= 10 and sum_prediction == 0:  # >= 5 or >= 10
            FN += 1
        elif sum_test_sample >= 10 and sum_test_sample >= 10:  # >= 5 or >= 10
            TP += 1

    # Ordering the values to create the confusion matrix
    conf_matrix = np.zeros((2, 2))
    conf_matrix[0, 0] = TP
    conf_matrix[1, 1] = TN
    conf_matrix[0, 1] = FP
    conf_matrix[1, 0] = FN

    return TP, FP, TN, FN, conf_matrix


def similarity_sequential_output_percentage_correct(model, X_test, y_test, percentage: int):
    """
    Analyses the predictions returned by the model. It allows us to know if there are sample close to the
    target but not exactly equal.

    Args:
        model: keras model
        X_test: Input samples
        y_test: Labels of the input samples
        percentage: Percentage of similarity we want to measure {Range between 0 and 100}


    Returns:
        correct: Samples with correct prediction in each timestep
        semi_correct: Sample with more or equal "percentage" of correct predictions
        incorrect: Percentatge of samples with more than "percentage" number of predictions different
                   than the ground-truth.
    """

    # Initialize the output variables
    equal = 0
    similar = 0
    incorrect = 0

    # Generate all the predictions
    predictions = np.round(model.predict(X_test))

    for prediction, label in zip(predictions, y_test):

        sum_prediction = np.sum(prediction)
        sum_test_sample = np.sum(label)

        percentage_correct_predictions = sum_prediction / sum_test_sample

        if (prediction == label).all():
            equal += 1

        elif percentage_correct_predictions >= percentage/100:
            similar += 1

        else:
            incorrect += 1

    return equal / predictions.shape[0], similar / predictions.shape[0], incorrect / predictions.shape[0]


def confusion_matrix_sequencialOutput(model, X_test, y_test, Tx):
    """
    Computes the PREDICTIONS and returns the TP, FP, TN, FN

    Input:
        * samples[list-str] -> List with the names of the images we want to predict 
        * labels[list-int] -> 0 = No hotspot | 1 = hotspot

    Output:
        * TP[int] -> True possitive results
        * FP[int] -> False positive results
        * TN[int] -> True negative results
        * FN[int] -> False negative results
        * conf_matrix[matrix] -> Confusion matrix
    """

    # Initialize the output variables
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # generate all the predictions 
    # predictions = np.round(model.predict(X_test))
    predictions = np.round(model.predict(X_test))

    for prediction, label in zip(predictions, y_test):
        for i in range(Tx):
            # Analyse the prediction of the model
            # The labels fo all five samples are the same. Therefore, we will use the last label iterated
            if(label[i] == 0 and prediction[i] == 0): # No hotspots & No hotspot
                TN+=1
            elif(label[i] == 0 and prediction[i] == 1):
                FP+=1
            elif(label[i] == 1 and prediction[i] == 0):
                FN+=1
            elif(label[i] == 1 and prediction[i] == 1):
                TP+=1
            
        # Ordering the values to create the confusion matrix
        conf_matrix = np.zeros((2,2))
        conf_matrix[0, 0] = TP
        conf_matrix[1, 1] = TN
        conf_matrix[0, 1] = FP
        conf_matrix[1, 0] = FN

    return TP, FP, TN, FN, conf_matrix


def confusion_matrix(model, X_test, y_test):
    """
    Computes the PREDICTIONS and returns the TP, FP, TN, FN

    Input:
        * samples[list-str] -> List with the names of the images we want to predict 
        * labels[list-int] -> 0 = No hotspot | 1 = hotspot

    Output:
        * TP[int] -> True possitive results
        * FP[int] -> False positive results
        * TN[int] -> True negative results
        * FN[int] -> False negative results
        * conf_matrix[matrix] -> Confusion matrix
    """

    # Initialize the output variables
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Initialize the confusion matrix
    conf_matrix = np.zeros((2, 2))

    # generate all the predictions 
    # predictions = np.round(model.predict(X_test))
    predictions = model.predict(X_test)

    # Check if the labels belong to binary labels or to one-hot vectors
    if y_test.shape[1] == 1:  # Single binary labels
        for prediction, label in zip(predictions, y_test):

            # Analyse the prediction of the model
            # The labels fo all five samples are the same. Therefore, we will use the last label iterated
            if label == 0 and np.round(prediction) == 0:  # No hotspots & No hotspot
                TN += 1
            elif label == 0 and np.round(prediction) == 1:
                FP += 1
            elif label == 1 and np.round(prediction) == 0:
                FN += 1
            elif label == 1 and np.round(prediction) == 1:
                TP += 1
    elif y_test.shape[1] == 2:  # One-hot labels (two classes)
        for prediction, label in zip(predictions, y_test):

            # Analyse the prediction of the model
            # The labels fo all five samples are the same. Therefore, we will use the last label iterated
            if np.argmax(label) == 0 and np.argmax(prediction) == 0:  # No hotspots & No hotspot
                TN += 1
            elif np.argmax(label) == 0 and np.argmax(prediction) == 1:
                FP += 1
            elif np.argmax(label) == 1 and np.argmax(prediction) == 0:
                FN += 1
            elif np.argmax(label) == 1 and np.argmax(prediction) == 1:
                TP += 1

    else:
        sys.exit('metrics/confusion_matrix: Labels no possible (1D or 2D vectors)')

    # Ordering the values to create the confusion matrix
    conf_matrix[0, 0] = TP
    conf_matrix[1, 1] = TN
    conf_matrix[0, 1] = FP
    conf_matrix[1, 0] = FN

    return TP, FP, TN, FN, conf_matrix
