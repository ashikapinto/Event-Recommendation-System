# Event-Recommendation-System
Data Analytics Project-2019

-The dataset has 6 files. But as sizes of three of the files were greater than 100MB , we could not upload them.

-datapreparation.py has all the cleaning and preprocessing of data along with creation of similarity matrices for users and events

-The different sparse matrices created are also uploaded along with the file, Explaination of each of these sparse matrices and how they are created is explained with the code

-Feature_extraction.py has code for extracting new features and rewriting the new test and train datasets

# Training the model

-The file model.ipynb contains the code to generate the sgdclassifier model given the newly generated test and train data with corresponding recommender scores.

# Testing the trained model and generating the result in desired format.

- Use the file test.ipynb to test the model on the test dataset. The results.csv will be generated with the prediction for the interested column.

-Using the results.csv we generate the final_result.csv which contains the recommendations in the order such that the event with highest recommender score will be in the beginning followed by the events with comparitively lower recommender scores.

-FOR TESTING, PLEASE USE test.ipynb.

