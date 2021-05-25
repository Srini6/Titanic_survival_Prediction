# Titanic_survival_Prediction
Dataset is obtained from Kaggle. With the data, a logistic regression algorithm and a neural network is designed to predict the survival of the passenger.
https://www.kaggle.com/c/titanic/overview

**About the Dataset**
The data is split into training and test data and available in csv file.
The datails of the dataset is available in Kaggle under the link https://www.kaggle.com/c/titanic/data.

**Project**
This project is done in MATLAB. The training and test dataset are loaded from the csv files. The working of the project is explained in a step by step manner.

1. Preprocessing of the data for training an algorithm includes removing unnecessary information from the data variables, merging two or more releveant information to a single variable and finding variable that eventually represent same information for classification. 
2. A dataset with 4 features is obtained from the prepossing. To train the model, normalization of data is performed. 
3. Two classification algorithms are trained. Logistic regressing and Neural network.
4. Details of Logistic regression.
  alpha = 2;
  iterations = 500;
  After training the algorithm, the model is used to predict the test data. Prediction with accuracy of **92.583732 percent** is acheived.
5. Neural network 
   input_layer_size = 4;  
   hidden_layer_size = 1; 
   output_layer_size = 2; 
   After traning the neural network, the prediction with accuracy **94.736842 percent** is acheived. 

**How to Run it in your PC**
1. Install matlab.
2. Download the files from this repository.
3. Run main.m
4. The code is well classified into multiple section and you can find addition comments in there.

PROST!
