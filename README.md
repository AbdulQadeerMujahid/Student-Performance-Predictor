# 🧾 Student Performance Predictor

## Overview

The **Student Performance Predictor** is a web-based application that analyzes and predicts student academic outcomes using **machine learning** and **data visualization**. Built with **Streamlit**, it allows educators to identify at-risk students and explore performance trends.

---

## Features

* Clean and preprocess student data
* Exploratory Data Analysis (EDA) with histograms, correlation heatmaps, and pie charts
* Predict Pass/Fail using:

  * Logistic Regression
  * Decision Tree (ID3)
  * Naive Bayes
* Group students with **K-Means clustering**
* Interactive Streamlit interface for predictions and visualizations

---

## Dataset

* ~40,000 student records
* Attributes: Previous Grades, Attendance Rate, Study Hours per Week
* Derived Pass/Fail label:

  * Pass: Grades ≥ 50 & Attendance ≥ 70
  * Fail: Otherwise

---

## Usage

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:

   ```bash
   streamlit run streamlit_student_visualizer_v2.py
   ```

---

## Results

* Decision Tree (ID3) achieved ~90% accuracy
* Logistic Regression ~85%
* Naive Bayes ~82%
* K-Means identified clusters: High Performers vs Low Performers

---

## Recommendations

* Add more features (class participation, exam scores)
* Use ensemble models for better accuracy
* Deploy on cloud for broader access

---

