from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data  = pd.read_csv(r'Social_Network_Ads.csv')
print(data.head())

data.drop(['User ID'], inplace = True, axis = 1)
data.drop(['Gender'], inplace = True, axis = 1) #dropping Gender as it has low correlation with target
print(data)

y = data['Purchased']
X = data.drop(['Purchased'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = .2)

rf_cl = RandomForestClassifier(random_state = 8, n_estimators=20, max_depth = 20, criterion='entropy')
rf_cl.fit(x_train, y_train)

#pickle the trained model
import pickle
with open('model.pkl','wb') as model_file:
  pickle.dump(rf_cl, model_file)