# Weather Application

### Description :
A CNN that predicts the rainfall in Skelton based on the rainfall levels in other nearby districts. The data standardizer first cleans the raw data and standardizes it. A generalised neural network coded from scratch without any machine learning libraries is then used on training data for a set number of epochs, with validation data assisting in ensuring the fit is not too loose or tight. The values of the weights in the network in the final epoch are used on unseen testing data. The results of this are destandardized and plotted onto a graph against the actual results.

#### After 10000 Epochs
<img src="https://user-images.githubusercontent.com/78224090/193465239-ad876ca0-4446-4c62-8094-8842da6ebe08.PNG" width="500" />

During the training phase, at each 1000th epoch an average of the error was outputted. The network initially has an error rate of 65% and after 10000 epochs, has an error average of 12%. This metric is from the validation dataset to which the network is not adjusted against. 

<img src="https://user-images.githubusercontent.com/78224090/193465031-c0e27cc6-72e2-42f5-a495-6f71d01e4daf.PNG" width="900" />
RED : Actual Testing Data, BLUE : Network Prediction

### Technologies used :
    - Python with pandas and matplot

### Functionalities implemented :
    - Cleaning the data of invalid rows
    - Standardizing the data and storing it to be used in the neural network
    - A neural network with adjustable parameters
    - Running the neural network against the training and validation data
    - A graph indicating how accurate the neural networks predictions are against the testing data

### Setup :
    (01) - Clone the repository and run rainfallAnn.py
    (02) - Adjust parameters like network structure or epoch count to observe different results
