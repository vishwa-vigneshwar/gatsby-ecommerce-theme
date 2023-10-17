# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:53.826146Z","iopub.execute_input":"2023-10-17T05:14:53.826560Z","iopub.status.idle":"2023-10-17T05:14:55.298272Z","shell.execute_reply.started":"2023-10-17T05:14:53.826533Z","shell.execute_reply":"2023-10-17T05:14:55.297162Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_score,learning_curve
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:55.300032Z","iopub.execute_input":"2023-10-17T05:14:55.300424Z","iopub.status.idle":"2023-10-17T05:14:58.390632Z","shell.execute_reply.started":"2023-10-17T05:14:55.300389Z","shell.execute_reply":"2023-10-17T05:14:58.389754Z"}}
true=pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
fake=pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.392212Z","iopub.execute_input":"2023-10-17T05:14:58.392933Z","iopub.status.idle":"2023-10-17T05:14:58.414472Z","shell.execute_reply.started":"2023-10-17T05:14:58.392897Z","shell.execute_reply":"2023-10-17T05:14:58.413430Z"}}
true.head(50)
true["subject"].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.417113Z","iopub.execute_input":"2023-10-17T05:14:58.417593Z","iopub.status.idle":"2023-10-17T05:14:58.430751Z","shell.execute_reply.started":"2023-10-17T05:14:58.417557Z","shell.execute_reply":"2023-10-17T05:14:58.429769Z"}}
fake.head()
fake["subject"].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.432034Z","iopub.execute_input":"2023-10-17T05:14:58.432342Z","iopub.status.idle":"2023-10-17T05:14:58.458488Z","shell.execute_reply.started":"2023-10-17T05:14:58.432319Z","shell.execute_reply":"2023-10-17T05:14:58.457308Z"}}
true.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.459824Z","iopub.execute_input":"2023-10-17T05:14:58.460200Z","iopub.status.idle":"2023-10-17T05:14:58.485052Z","shell.execute_reply.started":"2023-10-17T05:14:58.460165Z","shell.execute_reply":"2023-10-17T05:14:58.484035Z"}}
fake.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.487045Z","iopub.execute_input":"2023-10-17T05:14:58.487759Z","iopub.status.idle":"2023-10-17T05:14:58.496917Z","shell.execute_reply.started":"2023-10-17T05:14:58.487721Z","shell.execute_reply":"2023-10-17T05:14:58.496281Z"}}
true.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.498050Z","iopub.execute_input":"2023-10-17T05:14:58.498770Z","iopub.status.idle":"2023-10-17T05:14:58.510326Z","shell.execute_reply.started":"2023-10-17T05:14:58.498737Z","shell.execute_reply":"2023-10-17T05:14:58.509412Z"}}
fake.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.511899Z","iopub.execute_input":"2023-10-17T05:14:58.512487Z","iopub.status.idle":"2023-10-17T05:14:58.545235Z","shell.execute_reply.started":"2023-10-17T05:14:58.512454Z","shell.execute_reply":"2023-10-17T05:14:58.544076Z"}}
true.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.549302Z","iopub.execute_input":"2023-10-17T05:14:58.549846Z","iopub.status.idle":"2023-10-17T05:14:58.560178Z","shell.execute_reply.started":"2023-10-17T05:14:58.549807Z","shell.execute_reply":"2023-10-17T05:14:58.559451Z"}}
fake.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.561580Z","iopub.execute_input":"2023-10-17T05:14:58.562162Z","iopub.status.idle":"2023-10-17T05:14:58.572480Z","shell.execute_reply.started":"2023-10-17T05:14:58.562128Z","shell.execute_reply":"2023-10-17T05:14:58.571670Z"}}
true["label"]=1
fake["label"]=0

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.573804Z","iopub.execute_input":"2023-10-17T05:14:58.574446Z","iopub.status.idle":"2023-10-17T05:14:58.589535Z","shell.execute_reply.started":"2023-10-17T05:14:58.574420Z","shell.execute_reply":"2023-10-17T05:14:58.588815Z"}}
true.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.590683Z","iopub.execute_input":"2023-10-17T05:14:58.590968Z","iopub.status.idle":"2023-10-17T05:14:58.601643Z","shell.execute_reply.started":"2023-10-17T05:14:58.590945Z","shell.execute_reply":"2023-10-17T05:14:58.600879Z"}}
fake.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.602984Z","iopub.execute_input":"2023-10-17T05:14:58.603293Z","iopub.status.idle":"2023-10-17T05:14:58.618157Z","shell.execute_reply.started":"2023-10-17T05:14:58.603263Z","shell.execute_reply":"2023-10-17T05:14:58.617311Z"}}
data=pd.concat([fake,true],ignore_index=True)
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.619023Z","iopub.execute_input":"2023-10-17T05:14:58.619813Z","iopub.status.idle":"2023-10-17T05:14:58.633083Z","shell.execute_reply.started":"2023-10-17T05:14:58.619768Z","shell.execute_reply":"2023-10-17T05:14:58.632002Z"}}
X=data["text"]
y=data["label"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:14:58.634222Z","iopub.execute_input":"2023-10-17T05:14:58.634837Z","iopub.status.idle":"2023-10-17T05:15:13.352758Z","shell.execute_reply.started":"2023-10-17T05:14:58.634812Z","shell.execute_reply":"2023-10-17T05:15:13.351938Z"}}
vectorizer=CountVectorizer()
X_train_vectors=vectorizer.fit_transform(X_train)
X_test_vectors=vectorizer.transform(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:15:13.353994Z","iopub.execute_input":"2023-10-17T05:15:13.354385Z","iopub.status.idle":"2023-10-17T05:15:27.725439Z","shell.execute_reply.started":"2023-10-17T05:15:13.354351Z","shell.execute_reply":"2023-10-17T05:15:27.724466Z"}}
vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(data['text'])
X_train, X_test, y_train, y_test = train_test_split(X_vectors, data['label'], test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:15:27.726815Z","iopub.execute_input":"2023-10-17T05:15:27.727091Z","iopub.status.idle":"2023-10-17T05:15:27.734792Z","shell.execute_reply.started":"2023-10-17T05:15:27.727068Z","shell.execute_reply":"2023-10-17T05:15:27.733470Z"}}
new_texts = ["This news article is definitely fake.",
             "The research study confirms the truth of the news."]
new_texts_vectors = vectorizer.transform(new_texts)
predictions = classifier.predict(new_texts_vectors)
for text, label in zip(new_texts, predictions):
    print(f"Text: {text}\nPrediction: {'Fake' if label == 0 else 'True'}\n")


# %% [code] {"execution":{"iopub.status.busy":"2023-10-17T05:15:27.735949Z","iopub.execute_input":"2023-10-17T05:15:27.736351Z","iopub.status.idle":"2023-10-17T05:15:45.797504Z","shell.execute_reply.started":"2023-10-17T05:15:27.736310Z","shell.execute_reply":"2023-10-17T05:15:45.796614Z"}}
true_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
fake_df['label'] = 0
true_df['label'] = 1
combined_df = pd.concat([fake_df, true_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
X = combined_df['title'] + " " + combined_df['text']
y = combined_df['label']
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)
classifier = MultinomialNB(alpha=1.0)
classifier.fit(X_vectors, y)
def predict_label(input_title):
    input_text = ""  # You can add additional user input for the text if necessary
    input_data = input_title + " " + input_text
    input_vector = vectorizer.transform([input_data])
    label = classifier.predict(input_vector)[0]
    return label
input_title ="WASHINGTON (Reuters) - The special counsel"
predicted_label = predict_label(input_title)
if predicted_label == 0:
    print("Predicted Label: Fake")
else:
    print("Predicted Label: True")


# %% [code]
