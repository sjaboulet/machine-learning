# Machine Learning Fundamentals & Optimization

A comprehensive machine learning portfolio project covering Deep Learning optimization, Supervised Learning (Classification & Regression), and Unsupervised Learning (Clustering).

## Table of Contents
1. [Part 1: Deep Learning Optimization](#part-1-deep-learning-optimization)
2. [Part 2.1: Classification (Credit Card Fraud)](#part-21-classification-credit-card-fraud)
3. [Part 2.2: Regression (Bikeshare Demand)](#part-22-regression-bikeshare-demand)
4. [Part 2.3: Clustering (Spotify Audio Features)](#part-23-clustering-spotify-audio-features)
5. [Dependencies & Installation](#dependencies--installation)

---

## Part 1: Deep Learning Optimization
**Goal:** Train a Convolutional Neural Network (CNN) on the MNIST dataset to reach exactly 0.97 validation accuracy as quickly as possible (measured in wall-clock time).

* **Baseline Model:** Standard CNN architecture taking ~23.65 seconds.
* **Optimization 1 (Sub-Epoch Evaluation):** Implementing a custom Keras Callback (`TimeAndAccuracyStopper`) to evaluate validation accuracy every 50 batches instead of waiting for the epoch to end.
* **Optimization 2 (Parameter Reduction):** Drastically reducing the number of Conv2D filters and Dense neurons to cut computational overhead, successfully reaching 97% accuracy in **12.33 seconds** (processing batches twice as fast at 6ms/step).

---

## Part 2.1: Classification (Credit Card Fraud)
**Goal:** Identify fraudulent credit card transactions in a highly imbalanced dataset (99.8% legitimate vs. 0.2% fraud).

* **Dataset:** Kaggle Credit Card Fraud Detection.
* **Problem:** Standard accuracy metrics are useless. A naive baseline model missed 36% of all fraudulent transactions (Recall: 0.64).
* **Techniques & Preprocessing:** * Implemented **Class Weights** (`class_weight='balanced'`) to algorithmically penalize the model for missing fraud.
  * Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to mathematically generate synthetic fraud examples and balance the training data.
* **Results:** Both SMOTE and Class Weighting models successfully captured **92%** of all fraud (Recall: 0.92), proving that handling imbalanced data is strictly necessary for real-world deployment.

---

## Part 2.2: Regression (Bikeshare Demand)
**Goal:** Predict the continuous exact number of daily bike rentals based on environmental and meteorological factors.

* **Dataset:** UCI Capital Bikeshare Demand.
* **Preprocessing:** Dropped strictly correlated features to prevent data leakage (`casual` and `registered`) and applied One-Hot Encoding to categorical variables like `season` and `weathersit`.
* **Models Compared:** * **Linear Regression:** Baseline parametric model (MAE: 581.18 bikes, R²: 0.8436).
  * **Random Forest Regressor:** Complex non-parametric ensemble model (MAE: 452.67 bikes, R²: 0.8762).
* **Results:** The Random Forest successfully captured non-linear weather relationships (e.g., sharp drops in rentals during snow), reducing daily prediction error by nearly 130 bicycles.

---

## Part 2.3: Clustering (Spotify Audio Features)
**Goal:** Discover hidden groupings (clusters) within a large library of music based purely on mathematical audio profiles (Danceability, Energy, Valence, Acousticness, Instrumentalness), without using genre labels.

* **Dataset:** Kaggle Spotify Tracks Audio Features (Author: Maharshi Pandya).
* **Techniques:** * **StandardScaler:** To ensure equal geometric distance weighting.
  * **K-Means Clustering:** Segmented the data into 4 distinct clusters.
  * **PCA (Principal Component Analysis):** Reduced the 5-dimensional audio profile down to 2 dimensions for visual validation.
* **Evaluation:** * **Silhouette Score:** 0.268 (Confirmed distinct but realistically overlapping musical genres).
  * **Davies-Bouldin Index:** 1.204 (Confirmed clusters are compact and well-separated).
* **Results:** Successfully isolated distinct "vibes" pure through math:
  * **Cluster 0:** Rock/Metal (High Energy, Low Valence)
  * **Cluster 1:** Acoustic/Folk (Low Energy, High Acousticness)
  * **Cluster 2:** EDM/Techno (High Energy, High Instrumentalness)
  * **Cluster 3:** Pop/Upbeat (High Energy, High Valence)

---

## Dependencies & Installation

To run these notebooks, ensure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow