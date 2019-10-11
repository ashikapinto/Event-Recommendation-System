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
