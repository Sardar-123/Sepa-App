import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_excel('data.xlsx')
    data.columns = data.columns.str.strip()
    data.fillna('', inplace=True)
    return data

def preprocess_data(data):
    label_encoders = {}
    for column in ['Change Type', 'Element Name', 'Source Field', 'Type']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data[['Change Type', 'Element Name', 'Source Field', 'Type', 'Year']]
    y = data['Impact']
    return X, y, label_encoders

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Load data
data = load_data()
X, y, label_encoders = preprocess_data(data)
model = train_model(X, y)

# Create Streamlit app
st.title('Impact Prediction for Added/Removed Elements')

# Input fields
element_name = st.text_input('Enter Element Name:')
change_type = st.selectbox('Select Change Type:', options=['Add', 'Remove'])
source_field = st.text_input('Enter Source Field:')
type_field = st.text_input('Enter Type:')
year = st.number_input('Enter Year:', min_value=2000, max_value=2100, value=2023)

# Prediction
if st.button('Predict Impact'):
    input_data = {
        'Change Type': [change_type],
        'Element Name': [element_name],
        'Source Field': [source_field],
        'Type': [type_field],
        'Year': [year]
    }
    input_df = pd.DataFrame(input_data)

    # Encode input data
    for column in ['Change Type', 'Element Name', 'Source Field', 'Type']:
        le = label_encoders[column]
        input_df[column] = input_df[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # Predict impact
    prediction = model.predict(input_df)
    st.write('Predicted Impact:', prediction[0])

# Visualizations
st.subheader('Data Visualizations')

# Count of added and removed elements
change_counts = data['Change Type'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=change_counts.index, y=change_counts.values, palette='viridis')
plt.title('Count of Added and Removed Elements')
plt.xlabel('Change Type')
plt.ylabel('Count')
st.pyplot(plt)

# Yearly count of changes
yearly_changes = data.groupby(['Year', 'Change Type']).size().unstack(fill_value=0)
yearly_changes.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 8))
plt.title('Yearly Count of Added and Removed Elements')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Change Type')
st.pyplot(plt)
