# Event-Recommendation-System
Data Analytics Project-2019

-The dataset has 6 files. But as sizes of three of the files were greater than 100MB , we could not upload them.

-datapreparation.py has all the cleaning and preprocessing of data along with creation of similarity matrices for users and events

-The different sparse matrices created are also uploaded along with the file: a)user_event_response: sparse matrix that contains the index of the user and the index of the event followed by user's response to the event. user's response to event will be 1 if he is interested , -1 is not interested , and 0 if neither.(but these 0's wont be written to the sparse matrix)

b)temp.mtx:sparse matrix with index of each user and their preprocessed variables

c)user_matrix.mtx: normalized temp.mtx

d)similaritymatrix.mtx: contains similarity measure for each user-user pair
