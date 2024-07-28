import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#df = pd.read_csv('Cleaned-Data.csv', sep=',')
#df.isnull()
#df.isnull().sum().sum()
#df.dropna(inplace=True)
#X = df[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'None_Sympton',
#        'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'None_Experiencing', 'Gender_Male', 'Contact_Yes']]
#y = df['Severity']
#y = np.array(df.iloc[:, 13:14])
#print (type(y))
#le = LabelEncoder()
#y = le.fit_transform(y.reshape(-1))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)

#classifier = RandomForestClassifier()

#classifier.fit(X_train, y_train)


#from sklearn.svm import SVC
#sv = SVC(kernel='linear').fit(X_train,y_train)


#pickle.dump(sv, open('model.pkl', 'wb'))
df = pd.read_csv("Cleaned-Data.csv", sep=',')
# Contact_Yes	Severity_Mild	Severity_Moderate	Severity_None	Severity_Severe
df.isnull()
df.isnull().sum().sum()
df.dropna(inplace=True)
print(df.head())

#X = df[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'None_Sympton',
 #       'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'None_Experiencing', 'Age_0-9', 'Age_10-19',
  #      'Age_20-24', 'Age_25-59', 'Age_60+', 'Gender_Male', 'Contact_Yes']]
X = df[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'None_Sympton',
        'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'None_Experiencing', 'Gender_Male', 'Contact_Yes']]
y = df['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

pickle.dump(classifier, open("model.pkl", "wb"))
