<img src="https://github.com/Kirtika290/HEART_DISEASE_PREDICTING_MODEL_/blob/main/HEART%20image.jpg?raw=true" alt="Heart Disease Prediction" width="1200" height="500" />

<h1 style="font-size: 50px;"> HEART_DISEASE_PREDICTING_MODEL_</h1>

## 1. Introduction
The Heart Disease Prediction Model is a machine learning project that predicts the likelihood of heart disease based on various health indicators. Heart disease is one of the leading causes of death worldwide, making early detection crucial for preventive care. By analyzing patient data, this model aims to help healthcare providers identify high-risk patients, ultimately improving healthcare outcomes.

<div style="display: flex; justify-content: space-around; gap: 10px;">
    <img src="https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Heart%20Image%20GitHub%202.jpg?raw=true" alt="Heart Image 1" width="250" height="250" style="object-fit: cover;"/>
    <img src="https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Heart%20Image%20GitHub.jpg?raw=true" alt="Heart Image 2" width="250" height="250" style="object-fit: cover;"/>
    <img src="https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Stethescope%20Image%20GitHub.jpg?raw=true" alt="Stethoscope Image" width="250" height="250" style="object-fit: cover;"/>
</div>



It is also important that a doctor be present so that they could treat them. To make things worse, the tests usually take a long time before diagnosing whether a person would suffer from heart disease. However, it would be handy and better if we automate this process which ensures that we save a lot of time and effort on the part of the doctor and patient.

## 2. Machine Learning and Data Science
Machine learning and data science techniques are applied in this project to build predictive models from medical data. We explore different machine learning algorithms and data science practices, including data preprocessing, feature engineering, and model evaluation, to create a model capable of accurate heart disease risk prediction. The project uses Python, Scikit-learn, and TensorFlow, along with libraries like Pandas and Matplotlib for data handling and visualization.

## 3. Data
The dataset used for this project contains various medical features such as age, gender, blood pressure, cholesterol levels, and other health-related metrics. Each record represents a patient, with columns detailing their health metrics and a target variable indicating the presence or absence of heart disease. Key steps in data processing include:
- Handling missing values
- Encoding categorical variables
- Scaling numerical features

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## 4. Exploratory Data Analysis (EDA)
EDA helps uncover patterns and insights within the data, facilitating better feature selection and model building. EDA in this project includes:
- Statistical summary of data
- Distribution analysis of key health metrics
- Analysis of correlations between features and the target variable
- Identifying outliers and anomalies

This step lays the foundation for informed data preprocessing and model selection.

## 5. Visualizations
In this section, we will be visualizing some interests plots and trends in the data which is used by machine learning models to make predictions. We will also evaluate the performance of different machine learning and deep learning models on a given dataset by comparing various metrics. To identify the best version of each model, we will examine their hyperparameters.

To begin, we will review the list of rows and columns (features) in our dataset, which includes age, sex, cp (chest pain), chol, and others. This will provide insight into the types of features present and help us determine if additional features are necessary for analysis.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20213338.png)

Next, we will review the dataset for completeness by examining non-null values across each feature. With 303 entries, we confirm the dataset includes information on 1025 patients. Memory usage is minimal, so additional memory optimization steps, such as downcasting, are not needed.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20214418.png)

The heatmap provides a visual representation of the correlation between various features in the dataset, making it easy to identify relationships and patterns among the variables. Each cell in the heatmap corresponds to the correlation coefficient between two features, with values ranging from -1 to 1. A value of 1 indicates a perfect positive correlation, while -1 signifies a perfect negative correlation, and a value of 0 indicates no correlation. In the heatmap, the color intensity signifies the strength of these correlations: lighter colors suggest stronger relationships. For instance, features such as 'thalach' (maximum heart rate achieved) and 'slope' (slope of the peak exercise ST segment) may exhibit a positive correlation, implying that as one feature increases, the other tends to increase as well. Conversely, negative correlations, such as those between 'slope' and 'oldpeak', suggest that as one feature increases, the other decreases. This visualization aids in identifying which features may influence heart disease outcomes and can guide further analysis in predictive modeling and feature selection.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20215048.png)

This illustrates an array of axes objects generated from a plotting function, typically used for creating a grid of subplots in a Matplotlib figure. Each entry in the array corresponds to a subplot that is configured to display a specific feature of the dataset, allowing for a comprehensive visual analysis of multiple variables simultaneously. The titles indicate the features being plotted, including 'age,' 'sex,' 'cp' (chest pain type), 'trestbps' (resting blood pressure), 'chol' (cholesterol level), 'fbs' (fasting blood sugar), 'restecg' (resting electrocardiographic results), 'thalach' (maximum heart rate achieved), 'exang' (exercise induced angina), 'oldpeak' (depression induced by exercise relative to rest), 'slope' (slope of the peak exercise ST segment), 'ca' (number of major vessels colored by fluoroscopy), 'thal' (thalassemia), and 'target' (presence or absence of heart disease). This arrangement of subplots facilitates a detailed examination of the relationships and distributions among these variables, enabling researchers to identify patterns, trends, and potential correlations that may influence heart disease outcomes.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20215634.png)


This plot effectively illustrates the distribution of patients with heart disease compared to those without, highlighting a significant prevalence of heart disease among the sample. The larger number of patients diagnosed with heart disease not only emphasizes the importance of addressing this health issue but also provides a robust dataset for developing predictive classifiers. The balanced class distribution indicates that the dataset is well-structured, allowing for reliable evaluations of machine learning models using metrics such as accuracy. This balance is crucial as it ensures that the model is trained on a diverse set of examples, reducing the risk of bias and improving its generalization capabilities. Overall, this visualization underscores the dataset's potential to inform meaningful insights and contribute to the advancement of heart disease diagnostics and treatment strategies.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-06%20194858.png)

This represents a pie chart to visualize the distribution of target classes in the dataset, providing insights into the proportions of each class. The pie chart features two segments, colored in light green and salmon, representing the different target categories, and includes percentage labels for clarity. By employing an equal aspect ratio, the chart ensures a balanced presentation, making it easy to compare the class distributions at a glance. This visualization is crucial for understanding the dataset's balance, informing subsequent modeling decisions, and ensuring that appropriate metrics are chosen for evaluation. The accompanying print statements further display the shapes of the training and test datasets, offering a comprehensive overview of the data split.

![image](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20224619.png)

The box plot displayed above provides a visual summary of the distribution of features within the heart disease dataset. Each box represents the interquartile range (IQR), which encapsulates the middle 50% of the data points, with the line inside each box indicating the median value. The whiskers extend to represent the range of the data, while any points outside this range are considered outliers. This visualization enables us to quickly assess the central tendency, variability, and potential outliers for each feature associated with heart disease. By analyzing the box plots, we can identify features with significant variability, which may warrant further investigation during the data preprocessing stage. Additionally, the clear representation of outliers provides insights into individual patient data that may differ substantially from the rest of the population, potentially influencing model performance in predicting heart disease outcomes. Overall, the box plot serves as an essential tool for understanding the characteristics of the dataset, guiding subsequent analyses and modeling efforts.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-06%20195534.png)

This is a representation of a line plot to visualize the performance of a K-Neighbors Classifier across different values of \( k \) (the number of neighbors). The figure is set to a large size of 50 by 20 inches for enhanced visibility. The x-axis represents the number of neighbors, ranging from 1 to 20, while the y-axis displays the corresponding scores achieved by the classifier for each \( k \) value, with the scores plotted in red.

The plot includes text annotations at each data point, showing the exact score for each \( k \), facilitating easier interpretation of the results. The x-ticks are explicitly defined to ensure each neighbor value is clearly labeled. With a prominent title, "K Neighbors Classifier Score for Different K Values," set in a large font size, this visualization effectively conveys how varying the number of neighbors influences the model's performance, helping to identify the optimal \( k \) for maximizing accuracy in predictions.

![image alt](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20224209.png)

This representation  calculates and visualizes the mean cross-validation score for a Random Forest Classifier, providing a clear evaluation of its performance across multiple folds. The mean score, calculated from the individual cross-validation scores, is printed to the console, allowing for easy interpretation of the model's overall effectiveness. A horizontal dashed line is drawn on the plot to indicate the mean score, enabling a visual reference point against the individual fold scores. This visualization not only highlights the variability in performance across the different folds but also aids in assessing the stability and reliability of the model. By setting the y-axis limits from 0 to 1, we can clearly observe the scoring range and better understand how the model performs relative to potential thresholds for success. This detailed analysis is crucial for determining whether the Random Forest Classifier is a suitable choice for predicting heart disease, guiding further model tuning and selection based on its cross-validation performance.

![image](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20232830.png)

The represents a line plot illustrating the cross-validation scores for a Logistic Regression model across ten different folds. Each point on the line represents the score achieved in a specific fold, with markers highlighting the individual performance for clarity. A red dashed line is included to indicate the mean score, offering a reference point for evaluating the model's overall effectiveness. The plot is titled "Logistic Regression Cross-Validation Scores," with labeled axes for better understanding; the x-axis denotes the cross-validation fold number, while the y-axis displays the corresponding score. By setting the y-axis limits between 0 and 1, the visualization clearly depicts the scoring range, facilitating an intuitive assessment of the model's performance. The inclusion of a grid enhances readability, making it easier to analyze the trends and variability in the scores across the folds. This comprehensive visual representation aids in determining the reliability and stability of the Logistic Regression model in predicting outcomes, crucial for informed decision-making in model selection and evaluation.

![image](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20224601.png)

The scatter plot that visually represents the relationship between cholesterol levels and resting blood pressure among patients in the dataset. Utilizing the Seaborn library, the plot differentiates data points based on the presence of coronary heart disease (CHD) by applying color coding to the target variable. Additionally, if the dataset includes a gender feature (denoted as 'sex_1'), different styles are used for male and female data points, enhancing the visualization's clarity. The plot is titled "Scatter Plot of Cholesterol vs. Resting Blood Pressure," and the axes are labeled appropriately to indicate the scaled measurements for cholesterol and resting blood pressure. The inclusion of a legend with the title 'CHD Presence' allows for easy interpretation of the scatter plot, making it an effective tool for exploring the potential correlation between these two critical health indicators in relation to heart disease.

![image](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20224641.png)

This represent a  box plot that illustrates the distribution of resting blood pressure (trestbps) in relation to the presence of coronary heart disease (CHD) in the dataset. Utilizing the Seaborn library, the plot distinguishes between patients with and without CHD by plotting the target variable on the x-axis, where 1 indicates the presence of heart disease and 0 indicates its absence. The box plot visually summarizes the central tendency and variability of resting blood pressure within each group, highlighting key statistics such as the median, quartiles, and potential outliers. The plot is titled "Box Plot of Resting Blood Pressure by CHD Presence," with appropriately labeled axes to facilitate easy interpretation of the relationship between blood pressure levels and heart disease status, providing valuable insights into cardiovascular health metrics.

![image](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-05%20224656.png)

This is a representation of a training and validation accuracy of a neural network model across epochs. In this graph, the x-axis represents the number of epochs, which are the complete passes of the training data through the model. The y-axis shows accuracy, indicating how well the model is performing in terms of correct predictions. The `history.history['accuracy']` and `history.history['val_accuracy']` data points come from the model's training history, where `accuracy` represents the model's accuracy on the training set and `val_accuracy` on the validation set. By analyzing the plot, you can assess the model's performance over time, observing trends that indicate if the model is learning effectively, overfitting, or underfitting. Ideally, you would see both lines converge at a high accuracy level, showing that the model generalizes well to unseen data.

![image](https://github.com/bishtkartikey/Heart-Disease-Prediction-Model/blob/main/Screenshot%202024-11-06%20101910.png)

##  Machine Learning Models
A range of machine learning models were implemented to determine which performs best for this problem. The models include:
- **Logistic Regression**: A baseline model that predicts the probability of heart disease presence.
- **K-Nearest Neighbors (KNN)**: A simple algorithm that classifies patients based on the similarity of their health metrics.
- **Decision Tree**: A model that segments data based on feature values, useful for interpretability.
- **Random Forest**: An ensemble model that improves accuracy by aggregating multiple decision trees.
- **Neural Network (TensorFlow)**: A deep learning model designed to capture complex relationships between features.

Each model was evaluated using accuracy, precision, recall, and F1-score, helping identify the best-performing model for heart disease prediction.

##  Outcomes
The models achieved over 80% accuracy in predicting heart disease risk, with the Random Forest and Neural Network models showing the best performance. Key outcomes include:
- **Model Accuracy**: Overall accuracy achieved in distinguishing high-risk patients.
- **Feature Importance**: Analysis of the most important features contributing to heart disease, such as cholesterol levels and age.
- **Evaluation Metrics**: Detailed evaluation of precision, recall, and F1-score for each model, enabling a balanced assessment of model strengths and weaknesses.

##  Future Scope
Future enhancements for this project include:
- **Additional Data**: Incorporating more patient data from diverse demographics to improve model generalizability.
- **Advanced Feature Engineering**: Exploring derived features, such as lifestyle factors or medication history, for deeper insights.
- **Hyperparameter Tuning**: Further fine-tuning model parameters to enhance accuracy.
- **Web Application Integration**: Developing a user-friendly interface to allow healthcare professionals to input data and receive risk predictions instantly.
- **Deployment**: Deploying the model as a web application or mobile app for broader accessibility.

