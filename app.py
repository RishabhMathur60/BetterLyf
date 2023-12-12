from flask import Flask, render_template, request

# Import the necessary logic from "med.py"
# Replace the following lines with the necessary logic from "med.py"
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings

# Flask app initialization
app = Flask(__name__)

# Global variables for med.py logic
clf = None
description_list = {}
precautionDictionary = {}
severityDictionary = {}
reduced_data = None
cols = []


def initialize_med():
    global clf, description_list, precautionDictionary
    # Replace the necessary logic from "med.py" to initialize the variables
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    training = pd.read_csv('Training_Dataset.csv')
    cols = training.columns
    cols = cols[:-1]
    x = training[cols]
    y = training['Prognosis']

    reduced_data = training.groupby(training['Prognosis']).max()

    # mapping strings to numbers
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)

    getDescription()
    getprecautionDict()


def getDescription():
    global description_list
    with open('Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getprecautionDict():
    global precautionDictionary
    with open('Precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def predict_disease(symptoms_exp, num_days):
    def sec_predict(symptoms_exp):
        df = pd.read_csv('Training_Dataset.csv')
        X = df.iloc[:, :-1]
        y = df['Prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1

        return rf_clf.predict([input_vector])

    def print_disease(node):
        node = node[0]
        val = node.nonzero()
        disease = re.inverse_transform(val[0])
        return list(map(lambda x: x.strip(), list(disease)))

    def calc_condition(exp, days):
        sum = 0
        for item in exp:
            sum = sum + severityDictionary[item]
        if ((sum * days) / (len(exp) + 1) > 13):
            return "You should take the consultation from a doctor."
        else:
            return "It might not be that bad, but you should take precautions."

    symptoms_dict = {}

    for index, symptom in enumerate(cols):
        symptoms_dict[symptom] = index

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name in symptoms_exp:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_exp.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            response = []

            response.append("Possible diseases:")
            response.extend(present_disease)
            response.append("")

            response.append("Symptoms present:")
            response.extend(symptoms_present)
            response.append("")

            response.append("Symptoms given:")
            response.extend(symptoms_given)
            response.append("")

            response.append("Precautions:")
            response.extend(precautionDictionary[present_disease[0]])
            response.append("")

            confidence_level = (1.0 * len(symptoms_present)) / len(symptoms_given)
            response.append("Confidence level: " + str(confidence_level))

            return response

    global clf
    tree_ = clf.tree_
    feature_name = [
        cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(cols).split(",")
    symptoms_present = []

    for symptom_exp in symptoms_exp:
        if symptom_exp.strip() in chk_dis:
            symptoms_present.append(symptom_exp.strip())

    response = recurse(0, 1)

    if response:
        response.append(calc_condition(symptoms_present, num_days))

    return response


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']

    # Perform the necessary logic from "med.py" based on user input
    # Replace the following lines with the necessary logic from "med.py"
    symptoms_exp = re.findall(r'\w+', user_message.lower())
    num_days = 7  # Default value, can be adjusted based on user input

    response = predict_disease(symptoms_exp, num_days)

    return {'bot_response': response}


if __name__ == '__main__':
    initialize_med()
    app.run(port=5050,debug=True)
