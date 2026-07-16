#🔍 Python K-Nearest Neighbors (KNN) Classifier

A from-scratch implementation of the K-Nearest Neighbors machine learning algorithm built entirely in pure Python.

This project demonstrates the core mathematical concepts behind supervised classification algorithms without relying on heavy data science frameworks like Scikit-Learn, Pandas, or Numpy.

## 🚀 Features
* **Zero ML Libraries:** All logic, from distance calculation to feature scaling, is built using standard Python and the built-in `math module`.
* **Euclidean Distance Engine:** Calculates the straight-line distance between data points in an n-dimensional space using vector math.
* **Dynamic Feature Scaling:** Implements Min-Max Normalization to scale all continuous features down to a 0.0 - 1.0 range, dynamically adapting to any number of columns. This prevents magnitude bias
* **Majority Voting:** Extracts the top k nearest neighbors and determines the final classification using frequency analysis (mode)

## 🛠️ Tech Stack
* **Language:** Python
* **Math:**  Standard Python `math` module

## 💻 Installation & Usage

Because this project relies solely on the standard Python library, there are no external dependencies to install!

1. Clone the repository:
   ```bash
   git clone [https://github.com/Vinetix/knn-classifier.git](https://github.com/Vinetix/knn-classifier.git)
   cd knn-classifier

2. Run the main script from your terminal:
   ```bash
   python knn_classifier.py

## 📊 How it works under the hood

* A 2D dataset (e.g., Age and Income) and a list of corresponding labels are defined
* A new, unclassified data point is introduced.	
* The engine temporarily merges the new point with the dataset to calculate global minimums and maximums for Feature Scaling
* The normalized data is split back up, and the Euclidean distance is calculated between the new point and every point in the                           
  training set.
* The dataset is sorted by distance, and the top k closest neighbors hold a "vote" to classify the new data point.
