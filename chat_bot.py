import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, _tree
import csv
import warnings
training = pd.read_csv('Training_Dataset.csv')
testing = pd.read_csv('Testing_Dataset.csv')
columns = training.columns
columns = columns[:-1]
x = training[columns]
y = training['Prognosis']
y1 = y
less_data = training.groupby(training['Prognosis']).max()
warnings.filterwarnings("ignore", category=DeprecationWarning)
lea = preprocessing.LabelEncoder()
lea.fit(y)
y = lea.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[columns]
testy = testing['Prognosis']
testy = lea.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())
model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = columns
SeverityDictionary = dict()
Description_list = dict()
PrecautionDictionary = dict()

Symptoms_Dict = {}

for index, symptom in enumerate(x):
    Symptoms_Dict[symptom] = index
def getprecautionDict():
    global PrecautionDictionary
    with open('Precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            PrecautionDictionary.update(_prec)
def getSeverityDict():
    global SeverityDictionary
    with open('Severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                SeverityDictionary.update(_diction)
        except:
            pass
def getDescription():
    global description_list
    with open('Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            Description_list.update(_description)
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + SeverityDictionary[item]
    if ((sum * days) / (len(exp) + 1) > 13):
        print("\nIt would be advisable for you to consult a physician.")
    else:
        print("\nIt might not be as bad as you think, but you should still take precautions.")
def getInfo():
    print("-----------------------------------MED BOT-----------------------------------")
    print("Your Name Please?", end="  ->  ")
    name = input("")
    print("Hello,", name)
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []
def sec_predict(Symptoms_Exp):
    df = pd.read_csv('Training_Dataset.csv')
    X = df.iloc[:, :-1]
    y = df['Prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    Symptoms_Dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(Symptoms_Dict))
    for item in Symptoms_Exp:
        input_vector[[Symptoms_Dict[item]]] = 1

    return rf_clf.predict([input_vector])
def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = lea.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []
    
    while True:
        print("\nFeel free to tell your symptom which you're experiencing", end="  -> ")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("\nProblem you can have according to your symptom:")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("\nPlease Enter a Valid Symptom.")

    while True:
        try:
            num_days = int(input("\nOkay. From How Many Days you? : "))
            break
        except:
            print("Please Enter a Valid Input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_columns = less_data.columns
            symptoms_given = red_columns[less_data.loc[present_disease].values[0].nonzero()]
            print("\nAre you experiencing any ")
            Symptoms_Exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "?: ", end='')
                while True:
                    inp = input("")
                    if (inp == "yes" or inp == "no"):
                        break
                    else:
                        print("\nPlease Provide a Proper Valid Answers i.e. (yes/no) : ", end="")
                if (inp == "yes"):
                    Symptoms_Exp.append(syms)

            second_prediction = sec_predict(Symptoms_Exp)
            calc_condition(Symptoms_Exp, num_days)
            if (present_disease[0] == second_prediction[0]):
                print("\nYou may have", present_disease[0])
                print()
                print(Description_list[present_disease[0]])

            else:
                print("You may have", present_disease[0], "or", second_prediction[0])
                print()
                print(Description_list[present_disease[0]])
                print(Description_list[second_prediction[0]])

            precution_list = PrecautionDictionary[present_disease[0]]
            print("\nTake following measures :")
            for i, j in enumerate(precution_list):
                print(i + 1, ")", j)

    recurse(0, 1)
    
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, columns)
print("----------------------------------------------------------------------------------------")
