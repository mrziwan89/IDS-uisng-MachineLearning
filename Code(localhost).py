import pandas as pd
import streamlit as st
from streamlit import caching
from tensorflow.keras.utils import get_file
import tensorflow as tf
import requests
import os
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
#metrics to find the accuracy of the model
from sklearn import metrics
#C:\Users\Razor\.keras\datasets\kddcup.data_10_percent.gz

#path = get_file('kddcup.data_10_percent.gz', origin=
#        'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')

path = 'kddcup.data_10_percent.gz'


#****DEPLOYMENT PART****
st.title('Perfecting Intrusion Detection System using Machine Learning')
st.markdown('#')
"""
Intrusion Detection System is a software application to detect network intrusion using various machine 
learning algorithms. IDS monitors a network or system for malicious activity and protects a computer network from unauthorized access from users,
including perhaps insider. The intrusion detector learning task is to build a predictive model (i.e. a classifier) capable of distinguishing between
‘bad connections’ (intrusion/attacks) and a ‘good (normal) connections’.
This is the [dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

"""

st.code('''
import pandas as pd
import streamlit as st
from streamlit import caching
from tensorflow.keras.utils import get_file
import tensorflow as tf
import requests
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn import metrics

''', language='python')
st.subheader('Step-1')
st.header('Data Preprocessing')
st.code('''


path = 'kddcup.data_10_percent.gz'

df = pd.read_csv(path, header=None)
df.dropna(inplace=True,axis=1)
df.columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome'
]

''', language='python')

#
st.subheader('Step-2')
st.header('Data Processing & Cleaning')
st.code('''
def expand_categories(values):
    #here result is an empty array where the values will be further appended from the dataset 
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    # dividing the total value in the given array with the total length 
    return "[{}]".format(",".join(result))
    #seprating the result values with ,

@st.cache()
def analyze(df):
    cols = df.columns.values
    total = float(len(df))

    #print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            #getting unique values from the dataset
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            #print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

analyze(df)

@st.cache(suppress_st_warning=True)
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
# Encode text values to dummy variables(i.e. [1,0,0],
# [0,1,0],[0,0,1] for red,green,blue)

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Now encode the feature vector
def encodeVector():
    encode_numeric_zscore(df, 'duration')
    encode_text_dummy(df, 'protocol_type')
    encode_text_dummy(df, 'service')
    encode_text_dummy(df, 'flag')
    encode_numeric_zscore(df, 'src_bytes')
    encode_numeric_zscore(df, 'dst_bytes')
    encode_text_dummy(df, 'land')
    encode_numeric_zscore(df, 'wrong_fragment')
    encode_numeric_zscore(df, 'urgent')
    encode_numeric_zscore(df, 'hot')
    encode_numeric_zscore(df, 'num_failed_logins')
    encode_text_dummy(df, 'logged_in')
    encode_numeric_zscore(df, 'num_compromised')
    encode_numeric_zscore(df, 'root_shell')
    encode_numeric_zscore(df, 'su_attempted')
    encode_numeric_zscore(df, 'num_root')
    encode_numeric_zscore(df, 'num_file_creations')
    encode_numeric_zscore(df, 'num_shells')
    encode_numeric_zscore(df, 'num_access_files')
    encode_numeric_zscore(df, 'num_outbound_cmds')
    encode_text_dummy(df, 'is_host_login')
    encode_text_dummy(df, 'is_guest_login')
    encode_numeric_zscore(df, 'count')
    encode_numeric_zscore(df, 'srv_count')
    encode_numeric_zscore(df, 'serror_rate')
    encode_numeric_zscore(df, 'srv_serror_rate')
    encode_numeric_zscore(df, 'rerror_rate')
    encode_numeric_zscore(df, 'srv_rerror_rate')
    encode_numeric_zscore(df, 'same_srv_rate')
    encode_numeric_zscore(df, 'diff_srv_rate')
    encode_numeric_zscore(df, 'srv_diff_host_rate')
    encode_numeric_zscore(df, 'dst_host_count')
    encode_numeric_zscore(df, 'dst_host_srv_count')
    encode_numeric_zscore(df, 'dst_host_same_srv_rate')
    encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
    encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
    encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
    encode_numeric_zscore(df, 'dst_host_serror_rate')
    encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
    encode_numeric_zscore(df, 'dst_host_rerror_rate')
    encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')
    return 0

encodeVector()
# display 5 rows
df.dropna(inplace=True,axis=1)
df[0:5]


''',language='python')


# This file is a CSV, just no CSV extension or headers
# Download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
df = pd.read_csv(path, header=None)
#print("Read {} rows.".format(len(df)))

df.dropna(inplace=True,axis=1) # For now, just drop NA's 
# (rows with missing values)
# The CSV file has no column heads, so add them
df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

@st.cache()
def expand_categories(values):
    #here result is an empty array where the values will be further appended from the dataset 
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    # dividing the total value in the given array with the total length 
    return "[{}]".format(",".join(result))
    #seprating the result values with ,

@st.cache()
def analyze(df):
    cols = df.columns.values
    total = float(len(df))

    #print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            #getting unique values from the dataset
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            #print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

analyze(df)

@st.cache(suppress_st_warning=True)
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
# Encode text values to dummy variables(i.e. [1,0,0],
# [0,1,0],[0,0,1] for red,green,blue)

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Now encode the feature vector
def encodeVector():
    encode_numeric_zscore(df, 'duration')
    encode_text_dummy(df, 'protocol_type')
    encode_text_dummy(df, 'service')
    encode_text_dummy(df, 'flag')
    encode_numeric_zscore(df, 'src_bytes')
    encode_numeric_zscore(df, 'dst_bytes')
    encode_text_dummy(df, 'land')
    encode_numeric_zscore(df, 'wrong_fragment')
    encode_numeric_zscore(df, 'urgent')
    encode_numeric_zscore(df, 'hot')
    encode_numeric_zscore(df, 'num_failed_logins')
    encode_text_dummy(df, 'logged_in')
    encode_numeric_zscore(df, 'num_compromised')
    encode_numeric_zscore(df, 'root_shell')
    encode_numeric_zscore(df, 'su_attempted')
    encode_numeric_zscore(df, 'num_root')
    encode_numeric_zscore(df, 'num_file_creations')
    encode_numeric_zscore(df, 'num_shells')
    encode_numeric_zscore(df, 'num_access_files')
    encode_numeric_zscore(df, 'num_outbound_cmds')
    encode_text_dummy(df, 'is_host_login')
    encode_text_dummy(df, 'is_guest_login')
    encode_numeric_zscore(df, 'count')
    encode_numeric_zscore(df, 'srv_count')
    encode_numeric_zscore(df, 'serror_rate')
    encode_numeric_zscore(df, 'srv_serror_rate')
    encode_numeric_zscore(df, 'rerror_rate')
    encode_numeric_zscore(df, 'srv_rerror_rate')
    encode_numeric_zscore(df, 'same_srv_rate')
    encode_numeric_zscore(df, 'diff_srv_rate')
    encode_numeric_zscore(df, 'srv_diff_host_rate')
    encode_numeric_zscore(df, 'dst_host_count')
    encode_numeric_zscore(df, 'dst_host_srv_count')
    encode_numeric_zscore(df, 'dst_host_same_srv_rate')
    encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
    encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
    encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
    encode_numeric_zscore(df, 'dst_host_serror_rate')
    encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
    encode_numeric_zscore(df, 'dst_host_rerror_rate')
    encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')
    return 0

encodeVector()
# display 5 rows
df.dropna(inplace=True,axis=1)
df[0:5]


with st.echo():
    # This is the numeric feature vector, as it goes to the neural net

    x_columns = df.columns.drop('outcome')
    x = df[x_columns].values
    dummies = pd.get_dummies(df['outcome']) # Classification
    outcomes = dummies.columns
    num_classes = len(outcomes)
    y = dummies.values

    df.groupby('outcome')['outcome'].count()

st.subheader('Step-3')
st.header('Data Model Training and Testing')
with st.echo():
    # Convert to numpy - Classification
    # Create a test/train split.  25% test
    # Split into train/test
    @st.cache(allow_output_mutation=True, max_entries=10, ttl=3600)
    def trainModel():
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42)
        # dataset have classes and each and every class has its own data. 

        #Create neural net
        #model = Sequential()
        #model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
        #model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
        #model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
        #model.add(Dense(1, kernel_initializer='normal'))
        #model.add(Dense(y.shape[1],activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='adam')
        #monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        #                        patience=5, verbose=1, mode='auto',
        #                           restore_best_weights=True)
        #model.fit(x_train,y_train,validation_data=(x_test,y_test),
        #          callbacks=[monitor],verbose=2,epochs=2)
        #model.save('saved_model_h5.h5')


        #Minor Testing
        loaded_model = tf.keras.models.load_model('saved_model_h5.h5')
        pred = loaded_model.predict(x_test)
        # saving the tested values form the predict function in a numpy arrays.
        pred1 = np.argmax(pred,axis=1)
        y_eval = np.argmax(y_test,axis=1)
        #using metric function to calculate the final efficiency of the model
        s = metrics.accuracy_score(y_eval, pred1)
        return s
    score = trainModel()
    
    
    print("Validation score: {}".format(score))
    st.header("Accuracy: " + "{:.3f}".format(score*100, ))
    #C:\Users\Razor\.keras\datasets\kddcup.data_10_percent.gz
    print('FINNISH')
    st.write('FINNISH')
