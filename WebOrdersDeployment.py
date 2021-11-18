import pandas as pd
import streamlit as st
import numpy as np
import pycaret

concat = pd.read_excel('concat.xlsx')

from pycaret.regression import *
s = setup(concat, target = 'Weborders', transform_target = True, log_experiment = True, experiment_name = 'WEB')

#Enable models
models(internal=True)[['Name', 'GPU Enabled']]

# compare baseline models
best = compare_models()
print(best)

et_model = create_model('et')
predict_model(et_model)
save_model(et_model, model_name = 'extra_tree_model')


def predict_quality(model, df):
    predictions_data = predict_model(estimator=model, data=concat)
    return predictions_data['Label'][0]

model = load_model('extra_tree_model')

st.title('Wine Quality Classifier Web App')
st.write('This is a web app to classify the quality of your wine based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the classifier.')

Organicsearch = st.sidebar.slider(label='OS', min_value=4.0,
                                  max_value=16.0,
                                  value=10.0,
                                  step=0.1)
Users = st.sidebar.slider(label='Users', min_value=0.00,
                          max_value=2.00,
                          value=1.00,
                          step=0.01)

features = {'GA-Organicsearches': Organicsearch, 'GA-Users': Users}

features_df = pd.DataFrame([features])

st.table(features_df)

if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write(' Based on feature values, your wine quality is ' + str(prediction))


