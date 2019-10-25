from __future__ import division
import itertools
import datetime
import hashlib
import scipy.spatial.distance as ssd
from sklearn.preprocessing import normalize
import locale
import numpy as np
import pycountry
import scipy.io as sio
import scipy.sparse as ss
from collections import defaultdict

#STEP 1-ceating user_event_response matrix 
#We are first creating emply sets to hold all the unique users and events
distinct_users = set()
distinct_events = set()
#Next we create empty dictionaries, one to hold all events presented to each user and another to hold all users each event is presented to.
events_for_each_user = defaultdict(set)
users_for_each_event = defaultdict(set)
#The below lines of code fill the two sets and two dictionaries with respective contents as mentioned above
for data in ["event\\train.csv", "event\\test.csv"]:
  file = open(data, 'r')
  file.readline().strip().split(",")
  #print(f)
  for line in file:
    columns = line.strip().split(",")
    distinct_users.add(columns[0])
    distinct_events.add(columns[1])
    events_for_each_user[columns[0]].add(columns[1])
    users_for_each_event[columns[1]].add(columns[0])
  file.close()
#dok_matrix is used to generate a sparse matrix(to keep only those entries that are non zero as row index,column index,value)
user_event_response = ss.dok_matrix((len(distinct_users), len(distinct_events)))
#Creating two dictionaries to store users and events with their indices.
user_index=dict()
event_index = dict()
#filling user_index with user:index
for i, l in enumerate(distinct_users):
  user_index[l] = i
#dilling event_index with event:index
for i, l in enumerate(distinct_events):
  event_index[l] = i

train = open("event\\train.csv", 'r')
#skip the first row
train.readline()
#Here we are filling the sparse matrix: the sparse matrix contains: for each event-user in the training data-> the index of the user 
#and the index of the event according to the user_index and event_index list followed by user's response to the event
for l in train:
  col = l.strip().split(",")
  #print(col)
  i = user_index[col[0]]
  j = event_index[col[1]]
  user_event_response[i, j] = int(col[4]) - int(col[5])
#user's response to event will be 1 if he is interested , -1 is not interested , and 0 if neither.(but these 0's wont be written to the sparse matrix)
train.close()
#Now we are writing this sparse matrix to a mtx file
sio.mmwrite("event\\user_event_response", user_event_response)

#________________________________________________________________________________________________________________________________________

#STEP 2-finding unique user and event pairs
 
distinct_user_pairs = set()
distinct_event_pairs = set()
# Now we will find all unique user pairs and event pairs
#  These should be users who are linked via an event or events that are linked via a user in either the training or test sets. 
for e in distinct_events:
  u_list = users_for_each_event[e]
  if len(u_list) > 2:
    distinct_user_pairs.update(itertools.combinations(u_list, 2))

for u in distinct_users:
      e_list = events_for_each_user[u]
      if len(e_list) > 2:
        distinct_event_pairs.update(itertools.combinations(e_list, 2))
#________________________________________________________________________________________________________________________________________


# STEP-3 : Encoding the variables

# Locale is a column in users.csv. But it is categorical. We need to encode this. But we know that with new entries, there could 
#be new locales, so in order to encode it properly accounting all possible locales, we create a dictionary that contains all possible
#locales mapped to unique id's. We use the locale library for this
locales_id = defaultdict(int)
for i, loc in enumerate(locale.locale_alias.keys()):
  locales_id[loc] = i + 1
#print(localeIdMap)

# Similarly , we need to encode the location,again with new entries, there could 
#be new countries, so in order to encode it properly accounting all possible countries, we create a dictionary that contains all possible
#countries mapped to unique id's. We use the pycountry library for this
# load countries
country_id = defaultdict(int)
for i, country in enumerate(pycountry.countries):
  #print(c)
  country_id[country.name.lower()] = i + 1 
#print(countryIdMap)
#print(ctryIdx)

#To encode gender: male->1 female-> 2
gender_id = defaultdict(int, {"male":1, "female":2})
#print(genderIdMap)


#Following are the functions to encode each of the six variables in users.csv and events.csv
#locale is encoded using the locales_id dictionary created above
def encode_locale(l):
    return locales_id[l.lower()]
#gender is encoded using the gender_id dictionary created above
def encode_gender(g):
    return gender_id[g]

def encode_birth(b):
    try:
      return 0 if b == "None" else int(b)
    except:
      return 0

def encode_joined_month(d):
    val = datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ")
    return "".join([str(val.year), str(val.month)])

def encode_timezone(t):
    try:
      return int(t)
    except:
      return 0

#location is encoded using the country_id dictionary created above
def encode_country(c):
    if (isinstance(c, str) and len(c.strip()) > 0 and c.rfind("  ") > -1):
      #print(location[location.rindex("  "):].lower())
      return country_id[c[c.rindex("  ") + 2:].lower()]
    else:
      return 0

def feature_hash(value):
    if len(value.strip()) == 0:
      return -1
    else:
      return int(hashlib.sha224(value).hexdigest()[0:4], 16)

def float_value(value):
    if len(value.strip()) == 0:
      return 0.0
    else:
      return float(value)
#____________________________________________________________________________


#STEP 4 : creating user matrix with index of each user and their preprocessed variables
n = len(user_index.keys())
#print(n)
file = open("event\\users.csv", 'r')
columns = file.readline().strip().split(",")
#print(len(columns))
user_mat = ss.dok_matrix((n, len(columns) - 1)) 
for l in file:
  col = l.strip().split(",")
  # consider the users if they are in train.csv
  if col[0] in user_index:
    i = user_index[col[0]]
    user_mat[i, 0] = encode_locale(col[1])
    #print(encode_locale(col[1]))
    user_mat[i, 1] = encode_birth(col[2])
    user_mat[i, 2] = encode_gender(col[3])
    user_mat[i, 3] = encode_joined_month(col[4])
    user_mat[i, 4] = encode_country(col[5])
    user_mat[i, 5] = encode_timezone(col[6])
file.close()

sio.mmwrite("event\\temp", user_mat)

#______________________________________________________________________________
#STEP 5: finding similarity between each pair of users
#Normalise the user matrix
user_mat = normalize(user_mat, norm="l1", axis=0, copy=False)
sio.mmwrite("event\\user_matrix", user_mat)
# calculate the user similarity matrix
similarity_mat = ss.dok_matrix((n, n))
#diagonal elements are one as the similarity between a user and himself is 1
for i in range(0, n):
  similarity_mat[i, i] = 1.0
# for each unique pair in distinct_user_pairs, the correlation similarity measure is found by comparing the feature vectors for the users
#and that entry is stored in the similarity matrix
#this matrix is then written into the similaritymatrix.mtx file
for a,b in distinct_user_pairs:
  i = user_index[a]
  j = user_index[b]
  if similarity_mat[i,j]==0.0:
    coeff = ssd.correlation(user_mat.getrow(i).todense(),
      user_mat.getrow(j).todense())
    similarity_mat[i, j] = coeff
    similarity_mat[j, i] = coeff
sio.mmwrite("event\\similaritymatrix", similarity_mat)

# To generate the event - event similarity matrix
# There are two main aspects that can be considered the event meta data and the event content based on which the similarity 
# matrix can be generated. So, we decide to generate two similarity matrices for the events one based on the event meta data
# for which the similarity measure used is correlation and the other based on the content which uses the cosine similarity measure.

# Reading the file for the processing.
fin = open("F:\\Rachana\\SEM-5\\Data_Analytics\\Project\\Datasets\\events.csv", 'r')
# Skipping the first line as it includes the column names.
fin.readline()
# Determining the number of distinct events.
n_evts = len(event_index.keys())
# Generating the sparse matrices to store the feature vectors for each event
# 1. Based on the event meta data
evt_prop_matrix = ss.dok_matrix((n_evts, 7))
# 2. Based on the event content
evt_cont_matrix = ss.dok_matrix((n_evts, 100))
z = 0
# Populating the previously created sparse matrices with the feature vectors corresponding to the distinct events by reading the events.csv file.
for line in fin.readlines():
    z += 1
    cols = line.strip().split(",")
    e_id = cols[0]
    if e_id in event_index:
        i = event_index[e_id]
        # Cleaning the data read from the file by using the functions defined previously
        # 1. Populating the event Properties matrix
        evt_prop_matrix[i, 0] = encode_joined_month(cols[2]) # start_time
        evt_prop_matrix[i, 1] = feature_hash(bytes(cols[3], "utf-8")) # city
        evt_prop_matrix[i, 2] = feature_hash(bytes(cols[4], "utf-8")) # state
        evt_prop_matrix[i, 3] = feature_hash(bytes(cols[5], "utf-8")) # zip
        evt_prop_matrix[i, 4] = feature_hash(bytes(cols[6], "utf-8")) # country
        evt_prop_matrix[i, 5] = float_value(cols[7]) # lat
        evt_prop_matrix[i, 6] = float_value(cols[8]) # lon
        # 2. Populating the event content matrix
        for j in range(9, 109):
            evt_cont_matrix[i, j-9] = cols[j]
fin.close()
# Normalizing the columns of the matrices generated to make sure that the values measured on the different scales 
# do no add unnecessary meaning or correlation.
# For both the matrices l1 norm or Manhattan distance is being used.
evt_prop_matrix = normalize(evt_prop_matrix, norm="l1", axis=0, copy=False)
sio.mmwrite("EV_eventPropMatrix", evt_prop_matrix)
evt_cont_matrix = normalize(evt_cont_matrix, norm="l1", axis=0, copy=False)
sio.mmwrite("EV_eventContMatrix", evt_cont_matrix)

# Objects that calculate the correlation and cosine similarity measure when two feature vectors are given
psim=ssd.correlation
csim=ssd.cosine

# Generating the similarity matrices to later populate with the similarity measures.
evt_prop_sim = ss.dok_matrix((n_evts, n_evts))
evt_cont_sim = ss.dok_matrix((n_evts, n_evts))

# Populating the similarity matrices by calculating the correlation and cosine similarity for the meta data based and content based
# matrices.
for e1, e2 in distinct_event_pairs:
    i = event_index[e1]
    j = event_index[e2]
    if evt_prop_sim[i,j] == 0.0:
        epsim = psim(evt_prop_matrix.getrow(i).todense(), evt_prop_matrix.getrow(j).todense())
        evt_prop_sim[i, j] = epsim
        evt_prop_sim[j, i] = epsim
    if evt_cont_sim[i,j] == 0.0:
        ecsim = csim(evt_cont_matrix.getrow(i).todense(), evt_cont_matrix.getrow(j).todense())
        evt_cont_sim[i, j] = ecsim
        evt_cont_sim[j, i] = ecsim
sio.mmwrite("EV_eventPropSim", evt_prop_sim)
sio.mmwrite("EV_eventContSim", evt_cont_sim)


#The friends of the specified user are found out 
#The idea is that: (a) people with more friends are more likely to attend events 
#(b) if your friend is going, its more likely for you to go as well

#Determining the number of distinct users
nusers = len(user_index.keys())
numFriends = np.zeros((nusers))
userFriends = ss.dok_matrix((nusers, nusers))
#reading the data for preprocessing
#the user_friends.csv wasnt uploaded due to huge size
fin = open("C:\\Users\\anvitha poosarla\\Downloads\\user_friends.csv", 'r')
# skip header
fin.readline()  
ln = 0
for line in fin:
      cols = line.strip().split(",")
      user = cols[0]
      if user in user_index :
        friends = cols[1].split(" ")
        i = user_index[user]
        numFriends[i] = len(friends)
        for friend in friends:
          if friend in user_index :
            j = user_index[friend]
            # the objective of this score is to infer the degree to
            # and direction in which this friend will influence the
            # user's decision, so we sum the user/event score for
            # this user across all training events.
            eventsForUser =user_event_response.getrow(j).todense()
            score = eventsForUser.sum() / np.shape(eventsForUser)[1]
            userFriends[i, j] += score
            userFriends[j, i] += score
        ln += 1
fin.close()
# normalize the arrays
sumNumFriends =numFriends.sum(axis=0)
numFriends = numFriends / sumNumFriends
sio.mmwrite("C:\\Users\\anvitha poosarla\\Downloads\\event-recommendation-engine-challenge\\num_friends", np.matrix(numFriends))
#using the l1 norm or the manhattan distance
userFriends = normalize(userFriends, norm="l1", axis=0, copy=False)
sio.mmwrite("C:\\Users\\anvitha poosarla\\Downloads\\event-recommendation-engine-challenge\\user_friends", userFriends)