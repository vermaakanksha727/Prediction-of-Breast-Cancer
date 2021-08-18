# Prediction-of-Breast-Cancer
## Introduction:

Breast cancer occurs when a malignant (cancerous) tumour originates in the breast. As breast cancer tumours mature, they may metastasize to other parts of the body. For the diagnosis
and treatment of cancer, precise prediction of tumours is critically significant.Among the current techniques, supervised machine learning methods are the most popular in cancer diagnosis.The Breast Cancer datasets for prediction i have taken from Wisconsin. The dataset contains the samples of malignant(cancerous) and benign(non-cancerous) tumour cells.

## Objective:
* To use PCA to decrease the dimension of the data simply means that PCA will combine the 30 variables into much lesser principal components
* To cluster the data based on the new variables (principal components) obtained through PCA
* Build model using dataset obtained through PCA to predict whether breast cell tissue is malignant(cancerous) or Benign(non-cancerous).

## Data Description:
Data set is available here-[Breast Cancer Wisconsin Data](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) and also availble as in scikit-learn library ,i have loaded data from scikit-learn library.

Target (M = malignant, B = benign)
Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter

d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, >field 1 is Mean Radius, field 10 is Radius SE, field 20 is Worst Radius.


## Concepts Used In Prediction:
1.*Principal Component Analysis*: Principal component analysis is a method of extracting important variables in form of components from a large set of components available in a data set. It extracts low dimensional set of features from a high dimensional data set with a motive to capture as much information as possible. It is always performed on a symmetric correlation or covariance matrix. This means that the matrix should be numeric and have standardized data. 

2.*k-Means Clustering*:Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible.The elbow method runs k-means clustering on the dataset for a range of values for k (say from 1-10) and then for each value of k computes an average score for all clusters.

3.*Support Vector Machine*: Support vector machines (SVMs) learning algorithm will be used to build the predictive model. SVMs are one of the most popular classification algorithms, and have an elegant way of transforming nonlinear data so that one can use a linear algorithm to fit a linear model to the data. SVMs allow for complex decision boundaries, even if the data has only a few features. They work well on low dimensional and high-dimensional data.

## Data Preprocessing:
Data preprocessing or cleaning of the data is one of the major if not the most important prerequisite in data analysis. Proper preparation of data will lead to reliable and comprehensible results. Here are a few steps which I found necessary for my analysis-
* Checking for missing values. Quite fortunately there were no missing values in my dataset and hence I could use the entire original data without having to manipulate any point.
* Checking for data imbalance

![imbalance](https://user-images.githubusercontent.com/78952426/129872582-853bfd83-dbe1-45f5-8587-634fcd8eae23.png)
* checked correlation between different variables

## Pricipal Component Analysis-

Before carrying out PCA it is essential to standardize the data. Standardization makes sure that each of the original variables contribute equally to the analysis and then We instantiate the PCA function and set the number of components (features) that we want to consider. We’ll set it to “30” to see the explained variance of all the generated components, before deciding where to make the cut. Then we “fit” our scaled X data to the PCA function. get the curve:

![image](https://user-images.githubusercontent.com/78952426/129872908-a601ff45-d7f4-4e05-943c-e5dbbf45c3ad.png)

Cumulative Variance Ratio	Explained Variance Ratio
0	0.442720	          0.442720
1	0.632432	          0.189712
2	0.726364	          0.093932
3	0.792385	          0.066021
4	0.847343	          0.054958
5	0.887588	          0.040245
6	0.910095	          0.022507
7	0.925983	          0.015887
8	0.939879	          0.013896
9	0.951569	          0.011690

Looking at the dataframe above, when we use PCA to reduce our 30 predicting variables down to 10 components, we can still explain over 95% of the variance. The other 20 components explain less than 5% of the variance, so we can cut them. Using this logic, we will use PCA to reduce the number of components from 30 to 10.

## K-Means Clustering:
Before determining the clusters how can we decide how many clusters we want to divide our data into? The clusters should be formed in such a way that they are homogenous among themselves while sharing heterogeneity with other clusters.In K-Means clustering we use the Within Cluster Sum of Squares in the elbow method plot, to find the optimum number of clusters.

![elbow](https://user-images.githubusercontent.com/78952426/129873140-2639b659-38b6-439d-855f-a03a31079af6.png)


For selecting the number of cluters I have used a metric called the Silhouette score which measures the closeness of each point in one cluster to the points in neighboring clusters.The number of clusters corresponding to the highest Silhouette score is the best choice for analysis.

![cluster](https://user-images.githubusercontent.com/78952426/129873263-75b77771-96f3-4c0f-837e-190b397889aa.png)

As can be seen from the graph that 2 clusters give the best results.

## Prediction Using Support Vector Machine:

 I have constructed a predictive model using SVM machine learning algorithm to predict the diagnosis of a breast tumor. The diagnosis of a breast tumor is a binary variable (benign or malignant). I also evaluate the model using confusion matrix the receiver operating curves (ROC), which are essential in assessing and interpreting the fitted model.

## Model Evaluation:

```output
Confusion Matrix:
 [[ 62   2]
 [  2 105]]

Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.97      0.97        64
           1       0.98      0.98      0.98       107

    accuracy                           0.98       171
   macro avg       0.98      0.98      0.98       171
weighted avg       0.98      0.98      0.98       171
```
We can see that accuracy of our model is 98% and here We are using recall as our performance metric because we are dealing with diagnosing cancer — and are most concerned with minimizing False Negative prediction errors in our model.
 
ROC-Curve:

![roc](https://user-images.githubusercontent.com/78952426/129873356-93a06f18-de17-4a1c-8088-3761b555b260.png)


## Conclusion:

Used PCA to find the principal components and before modeling checked for how many clusters we want to divide our data using k-means clustering . Finally, I preformed Support Vector Machines model to predict cancer based on PCA. The accuracy rate of this model is 98%.


