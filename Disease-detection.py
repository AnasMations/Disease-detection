import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,plot_confusion_matrix,plot_roc_curve,precision_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
#from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

st.markdown(
    """
    <style>
    .main{
    
    }
    </style>
    """,
    unsafe_allow_html=True
)

df = pd.read_csv('Data/dataset.csv')
df1 = pd.read_csv('Data/Symptom-severity.csv')

@st.cache(allow_output_mutation=True)
def model():
    # **Read and shuffle the dataset**
    df = pd.read_csv('Data/dataset.csv')
    df = shuffle(df, random_state=42)

    # **Remove the trailing space from the symptom columns**
    cols = df.columns
    data = df[cols].values.flatten()

    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)

    df = pd.DataFrame(s, columns=df.columns)

    # **Fill the NaN values with zero**

    df = df.fillna(0)

    # **Symptom severity rank**

    df1 = pd.read_csv('Data/Symptom-severity.csv')

    # **Get overall list of symptoms**

    df1['Symptom'].unique()

    # **Encode symptoms in the data with the symptom rank**

    vals = df.values
    symptoms = df1['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]

    d = pd.DataFrame(vals, columns=cols)

    # **Assign symptoms with no rank to zero**

    d = d.replace('dischromic _patches', 0)
    d = d.replace('spotting_ urination', 0)
    df = d.replace('foul_smell_of urine', 0)

    # **Check if entire columns have zero values so we can drop those values**

    (df[cols] == 0).all()

    print("Number of symptoms used to identify the disease ", len(df1['Symptom'].unique()))
    print("Number of diseases that can be identified ", len(df['Disease'].unique()))

    # Compare linear relationships between attributes using correlation coefficient generated using correlation heatmap

    #plt.figure(figsize=(10, 10))
    #sns.heatmap(df.corr(), cmap='PuBu', annot=False)


    # **Get the names of diseases from data**

    df['Disease'].unique()

    # ### Select the features as symptoms column and label as Disease column
    #
    # Explination: A **feature** is an input; **label** is an output.
    # A feature is one column of the data in your input set. For instance, if you're trying to predict the type of pet someone will choose, your input features might include age, home region, family income, etc. The label is the final choice, such as dog, fish, iguana, rock, etc.
    #
    # Once you've trained your model, you will give it sets of new input containing those features; it will return the predicted "label" (pet type) for that person.

    data = df.iloc[:, 1:].values
    labels = df['Disease'].values

    # ## Splitting the dataset to training (80%) and testing (20%)
    #
    # Separating data into training and testing sets is an important part of evaluating data mining models. Typically, when you separate a data set into a training set and testing set, most of the data is used for training, and a smaller portion of the data is used for testing. By using similar data for training and testing, you can minimize the effects of data discrepancies and better understand the characteristics of the model.
    # After a model has been processed by using the training set, we test the model by making predictions against the test set. Because the data in the testing set already contains known values for the attribute that you want to predict, it is easy to determine whether the model's guesses are correct.
    #
    # * Train Dataset: Used to fit the machine learning model.
    # * Test Dataset: Used to evaluate the fit machine learning model.

    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


    # Random forest code:

    from sklearn.naive_bayes import GaussianNB
    rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators=500, max_depth=8)
    rnd_forest.fit(x_train, y_train)
    preds = rnd_forest.predict(x_test)
    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    print('F1-score% =', f1_score(y_test, preds, average='macro') * 100, '|', 'Accuracy% =',
          accuracy_score(y_test, preds) * 100)
    # sns.heatmap(df_cm)

    # kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    # rnd_forest_train = cross_val_score(rnd_forest, x_train, y_train, cv=kfold, scoring='accuracy')
    # pd.DataFrame(rnd_forest_train, columns=['Scores'])
    # print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (
    # rnd_forest_train.mean() * 100.0, rnd_forest_train.std() * 100.0))
    #
    # kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    # rnd_forest_test = cross_val_score(rnd_forest, x_test, y_test, cv=kfold, scoring='accuracy')
    # pd.DataFrame(rnd_forest_test, columns=['Scores'])
    # print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (
    # rnd_forest_test.mean() * 100.0, rnd_forest_test.std() * 100.0))

    return rnd_forest

rnd_forest = model()
# # Fucntion to manually test the models

def predd(psymptoms, x):
    print(psymptoms)
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]

    psy = [psymptoms]

    pred2 = x.predict(psy)
    return pred2[0]


model()

#----Data import----
Disease_description = pd.read_csv('Data/Disease_description.csv')
Disease_precaution = pd.read_csv('Data/Disease_precaution.csv')
Symptom_severity = pd.read_csv('Data/Symptom-severity.csv')
symptomsList = Symptom_severity["Symptom"].tolist()
symptomsList.insert(0, "")

# def diseaseInfo(disease):
#     disease_description = "Description"
#     disease_precaution = "Precaution"
#
#     st.header("You have " + disease)
#     st.subheader(disease_description)
#     st.subheader(disease_precaution)

def diseaseInfo(desase):
    df = pd.read_csv('Data/Disease_precaution.csv')
    df2 = pd.read_csv('Data/Disease_description.csv')

    index = df.index
    condition = df["Disease"] == desase

    string_index = index[condition]
    string_index_list = string_index.tolist()
    DeIn=string_index_list[0]
    st.subheader(desase)
    st.write(df2.loc[DeIn,"Description" ])
    st.write(df.loc[DeIn, : ])

header = st.container()
modelTraining = st.container()

#----Streamlit----

with header:
    st.image('header.jpg')
    st.title("Disease detection website")
    st.header("Model Training")

with modelTraining:
    sel_col, disp_col = st.columns(2)
    choiceSize = st.slider("Insert the number of symptoms", min_value=1, max_value=17, value=3)
    choiceList = []
    for i in range(0, choiceSize):
        choice = st.selectbox("Symptom "+str(i+1), options=symptomsList, key=i)
        if not(choice in choiceList)and choice!=' ':
            choiceList.append(choice)
        else:
            st.write("Symptom "+str(i+1)+" and Symptom "+str(choiceList.index(choice)+1)+ "\nCan't have same symptom selected more than once\n")
    
    for i in range(choiceSize, 17):
        choiceList.append(0)

    submit = st.button("Predict")
    #st.text(choiceList)

    if(submit==True):
        if not(' 'in choiceList):
            diseaseInfo( predd(choiceList, rnd_forest) )
        else:
            st.write("Can't recieve empty inputs")
        #diseaseInfo("Acne")
