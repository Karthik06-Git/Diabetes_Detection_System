### Diabetes Detection Application backend

### Flask app

from flask import Flask, render_template, request

app = Flask(__name__)



# Loading pickle Model
import pickle as pk
pickle_file = open("My_final_diabetes_RF_model.pickle","rb")
model = pk.load(pickle_file)

# importing all neccessary things for Model-prediction
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

X_train_df = pd.read_csv("X_train_diabetes_to_preprocess.csv")



### Preprocessing Methods as used in building ML-model
# Scaling
num_cols = X_train_df.select_dtypes(include=["int64","float64"]).columns.tolist()

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train_df[num_cols])

# Encoding
cat_cols = X_train_df.select_dtypes(include=["object"]).columns.tolist()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop="first", dtype="int64", handle_unknown="ignore")
encoder.fit(X_train_df[cat_cols])


# func to get all preprocessing done for the input record-data
def preprocess(data):
    # scale the Numerical values in input data
    data[num_cols] = scaler.transform(data[num_cols])           
    # encode the Categorical values in input data
    encoded_data = encoder.transform(data[cat_cols])
    encoded_data_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

    # combine the Numerical, Categorical data into a final input data-point
    input_data = pd.concat([data[num_cols], encoded_data_df], axis=1)

    return input_data





    

# _______________________________________________________________________________________________




@app.route('/')
def home():
    return render_template("index.html")


@app.route('/detect', methods=['POST'])
def detect():
    if request.method=='POST':
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['glucose_level'])
        gender = str(request.form['gender'])
        smoking_history = str(request.form['smoking']) 

    data = pd.DataFrame({
        "age":[age], "bmi":[bmi], "HbA1c_level":[HbA1c_level], "blood_glucose_level":[blood_glucose_level],
        "gender":[gender], "smoking_history":[smoking_history]                 
    })

    input = preprocess(data)
    # passing this preprocessed data to ML-model
    output = model.predict(input)

    if output==1:
        pred = "Diabetic."
    elif output==0:
        pred = "not Diabetic."

    return render_template("index.html", prediction="Person is {}".format(pred))





if __name__ == "__main__":
    app.run(debug=True) 









