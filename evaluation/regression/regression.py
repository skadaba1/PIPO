import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit


def exponential_model(x, a, b):
    return a * np.exp(b * x)

def linear_model(x, a, b):
        return a * x + b

# Sample data
def fit_linear_model(probs, scores, model, initial_guess=None, plot=True):
    logprobs =  np.array(probs) # probs
    logscores = np.array(scores)   # logscores

    # Initial guess for the parameters
    initial_guess = [1, 0] if initial_guess is None else initial_guess

    # Number of folds for K-Fold Cross Validation
    n_splits = 10

    # Initialize K-Fold Cross Validation
    kf = KFold(n_splits=n_splits, shuffle=True)

    # List to store RMSE values for each fold
    rmse_list = []
    rrmse_list = []
    r2_list = []

    # Perform K-Fold Cross Validation
    for fold, (train_index, test_index) in enumerate(kf.split(logprobs), start=1):
        logprobs_train, logprobs_test = logprobs[train_index], logprobs[test_index]
        logscores_train, logscores_test = logscores[train_index], logscores[test_index]

        # Fit the model
        params, covariance = curve_fit(model, logprobs_train, logscores_train, p0=initial_guess)

        # Predict the log-scores on the test set
        predicted_logscores = model(logprobs_test, *params)

        # Compute the RMSE
        rmse = np.sqrt(mean_squared_error(logscores_test, predicted_logscores))
        rmse_list.append(rmse)
        rrmse_list.append(rmse / np.average(logscores_test))

        # Compute the R^2 coefficient
        r2 = r2_score(logscores_test, predicted_logscores)
        r2_list.append(r2)

        # Plot results for the last fold
        if fold == n_splits:
            if(plot):
                plt.scatter(predicted_logscores, logscores_test)
                #plt.scatter(logprobs_test, predicted_logscores, label='Predicted Logscores', color='red')
                plt.xlabel('Predicted Library Scores')
                plt.ylabel('Measured Library Scores')

                # plot line of best fit in red
                plt.title(f'Fold {fold} - Actual vs Predicted Library Scores')
                plt.show()
            else:
                pass

    print("\n")
    # Print RMSE for the last fold
    print(f'RMSE for Fold {n_splits}: {rmse_list[n_splits-1]}')
    print(f'RRMSE for Fold {n_splits}: {rrmse_list[n_splits-1]}')
    print(f'R^2 for Fold {n_splits}: {r2_list[n_splits-1]}')
    print("\n")
    print(f'Average RMSE for all K: {np.average(rmse_list)}')
    print(f'Average RRMSE for all K: {np.average(rrmse_list)}')
    print(f'Average R^2 for all K: {np.average(r2_list)}')
    print("\n")

    # Plot RMSE for all folds
    if(plot):
        plt.plot(range(1, n_splits + 1), r2_list, marker='o')
        plt.xlabel('Fold')
        plt.ylabel('RMSE')
        plt.title('RMSE for Each Fold')
        plt.show()
    return linear_model


