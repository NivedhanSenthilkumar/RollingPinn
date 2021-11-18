import pandas as pd
import streamlit as st
import numpy as np
import openpyxl
from sklearn.linear_model import LinearRegression
import pickle

concat = pd.read_excel('concat.xlsx')
X = concat[['GA-Organicsearches','GA-Users']]
Y = concat['Weborders']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size = 0.2, random_state = 10)

model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

#Saving Model
pickle_out = open("Weborders.pkl", mode="wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# loading the trained model
pickle_in = open('Weborders.pkl', 'rb')
Orders = pickle.load(pickle_in)


def prediction(OrganicSearches, Users):
    # Making predictions
    prediction = Orders.predict([[OrganicSearches, Users]])
    return prediction


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    Organicsearch = st.sidebar.slider(label='OS', min_value=4.0, max_value=16.0, value=10.0, step=0.1)
    Users = st.sidebar.slider(label='Users', min_value=0.00, max_value=2.00, value=1.00, step=0.01)

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(OrganicSearches, Users)
        st.success('Your loan is {}'.format(result))
        print(result)


if __name__ == '__main__':
    main()




