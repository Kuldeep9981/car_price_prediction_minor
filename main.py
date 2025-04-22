import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv("car_data_minor.csv")
new_df = df.drop(columns=['Unnamed: 0','Location','Colour','Seats'])
new_df.rename(columns={"Name":'Model',"company_name":'Brand'},inplace=True)

# splitting into input and output columns
x = new_df.drop(columns='Price')
y = new_df['Price']

# split train data and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=435)

# categorical data columns convert into numerical
ohe=OneHotEncoder()
ohe.fit(x[['Model','Transmission','Fuel_Type','Brand']])

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_,handle_unknown='ignore'),['Model','Transmission','Fuel_Type','Brand']),remainder='passthrough')

model_lr=LinearRegression()
pipe=make_pipeline(column_trans,model_lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)

# calculate the accuracy
r2_score(y_test,y_pred)

# save the file
pickle.dump(pipe,open('LinearREgressionModel.pkl','wb'))