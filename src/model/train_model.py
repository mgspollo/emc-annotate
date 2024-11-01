from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from src.data.import_data import read_test_data, read_test_metadata
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


def aggregate_features_old(columns_to_aggregate=["is_power_supply_present", "is_processor_present",
                                                 "is_load_present"]):
    df = read_test_data()
    for col in columns_to_aggregate:
        df_all_true = pd.DataFrame()
        df_all_false = pd.DataFrame()
        for i, row in df.iterrows():
            row = row.to_dict()
            if row[col]:
                df_true = row["test_signal"].iloc[::5, :]
                if i == 0:
                    df_all_true = df_true.rename(columns={"intensity": str(row["test_id"])})
                else:
                    df_all_true = pd.merge(df_all_true,
                                           df_true.rename(columns={"intensity": str(row["test_id"])}),
                                           on='frequency')
            else:
                df_false = row["test_signal"].iloc[::5, :]
                if i == 0:
                    df_all_false = df_false.rename(columns={"intensity": str(row["test_id"])})
                else:
                    df_all_false = pd.merge(df_all_false,
                                            df_false.rename(columns={"intensity": str(row["test_id"])}),
                                            on='frequency')
        df_all_true = df_all_true.set_index("frequency")
        df_all_true["mean"] = df_all_true.mean(axis=1)
        df_all_true = df_all_true.reset_index()
        df_all_false = df_all_false.set_index("frequency")
        df_all_false["mean"] = df_all_false.mean(axis=1)
        df_all_false = df_all_false.reset_index()

    fig.add_scatter(x=df_all_ambient['frequency'], y=df_all_ambient['mean'], mode='lines', name="mean ambient")


def aggregate_feature(columns_to_aggregate=["display_output_protocol", "display_receive_protocol"]):
    max_test_id = 10080
    df_all_signal = read_test_data(max_test_id)
    df_metadata = read_test_metadata(max_test_id)
    columns_to_aggregate.append("test_id")
    df_tests = df_metadata[columns_to_aggregate]
    df_tests = df_tests.dropna()
    df_tests["is_protocol_constant"] = df_tests["display_output_protocol"] == df_tests["display_receive_protocol"]
    test_ids = [str(i) for i in df_tests['test_id'].tolist()]
    df_all_signal_tests = df_all_signal[test_ids]
    index = df_all_signal_tests.index
    X = df_all_signal_tests.T.to_numpy()
    y = df_tests["is_protocol_constant"].astype(int).to_numpy()
    return X, y, index


def train_model():
    X, y, index = aggregate_feature()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    X_reduced = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2)

    # Reshape for CNN
    X_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_cnn_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define the CNN model
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_cnn, y_train, epochs=10, validation_split=0.2)

    loss, accuracy = model.evaluate(X_cnn_test, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')

    return model


X, y, index = aggregate_feature()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

# Assume X and y are already defined as your data and labels

# Cross-validation
model = GradientBoostingClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
# scores = cross_val_score(model, X, y, cv=5)
# print(f'Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}')
#
# # Learning curves
# train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
# train_mean = np.mean(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
# plt.plot(train_sizes, test_mean, 'o-', label='Cross-Validation Score')
# plt.xlabel('Training Examples')
# plt.ylabel('Score')
# plt.legend(loc='best')
# plt.show()

feature_importances = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.plot(index, feature_importances)
plt.xlabel('Frequency Index')
plt.ylabel('Feature Importance')
plt.title('Feature Importances Across the Spectrum')
plt.show()

# # Bootstrap resampling
# n_iterations = 100
# n_size = int(len(X) * 0.8)
# scores = list()
# for i in range(n_iterations):
#     X_resample, y_resample = resample(X, y, n_samples=n_size)
#     model.fit(X_resample, y_resample)
#     score = model.score(X, y)
#     scores.append(score)
# print(f'Bootstrap Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}')

# Simple model with regularization
model = LogisticRegression(penalty='l2', C=0.1)
scores = cross_val_score(model, X, y, cv=5)
print(f'Regularized Model Accuracy: {scores.mean():.4f} ± {scores.std():.4f}')

# Decision tree model
model = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(model, X, y, cv=5)
print(f'Simple Model Accuracy: {scores.mean():.4f} ± {scores.std():.4f}')

if __name__ == "__main__":
    train_model()
