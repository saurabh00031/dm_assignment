import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CSVFile
from django.http import HttpResponse,JsonResponse
from rest_framework.parsers import FileUploadParser,MultiPartParser,FormParser
from .models import CSVFile
from .serializers import CSVFileSerializer
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from django.http import FileResponse
import uuid
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score



class CSVFileUploadView(APIView):
    parser_classes = ( MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = CSVFileSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




def home(request):
    return HttpResponse("Hello homepage")


def assignment1(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    


    data = pd.read_csv(node[0].file)
    mean_data=data.mean()
    median_data = data.median()
    mode_data = data.mode().iloc[0]  
    var_data = data.var()
    sd_data = data.std()


    print("mean",mean_data[0])
    print("median",median_data[0])
    print("mode",mode_data[0])
    print("variance",var_data[0])
    print("standard deviation",sd_data[0])
    

    my_data={
        "name":node[0].name,
        "mean":mean_data.to_dict(),
        "median":median_data.to_dict(),
        "mode":mode_data.to_dict(),
        "variance":var_data.to_dict(),
        "std":sd_data.to_dict()
    }
    return JsonResponse(my_data)






from io import StringIO

def assignment1_que2(request):
        node=CSVFile.objects.all()
        print(node[0].name)
        if len(node)==0 :
            return HttpResponse("No csv file in database !!")
        

        csv_file=node[0].file

        print("boundary 1")

        df = pd.read_csv(csv_file)

        df = df.drop('variety', axis=1)
  
        # Calculate various dispersion measures
        data = df.values.flatten()
        data = np.sort(data)

        # Range
        data_range = np.ptp(data)

        # Quartiles
        quartiles = np.percentile(data, [25, 50, 75])

        # Interquartile Range (IQR)
        iqr = quartiles[2] - quartiles[0]

        # Five-Number Summary
        five_number_summary = {
            "Minimum": np.min(data),
            "Q1 (25th Percentile)": quartiles[0],
            "Median (50th Percentile)": quartiles[1],
            "Q3 (75th Percentile)": quartiles[2],
            "Maximum": np.max(data)
        }

        csv_file.seek(0)  # Ensure the file pointer is at the beginning
        csv_data = csv_file.read().decode('utf-8')
        csv_buffer = StringIO(csv_data)

        column_name="sepal.length"
        column_name2="sepal.width"

        column_values=[]
        column_values2=[]

        csv_reader = csv.DictReader(csv_buffer)
        for row in csv_reader:
            if column_name in row:
                   column_values.append(row[column_name])

        csv_file.seek(0)  # Ensure the file pointer is at the beginning
        csv_data = csv_file.read().decode('utf-8')
        csv_buffer = StringIO(csv_data)
        csv_reader = csv.DictReader(csv_buffer)

        for row in csv_reader:
            if column_name2 in row:
                   column_values2.append(row[column_name2])
    
        

        result = {
            "Range": data_range,
            "Quartiles": quartiles.tolist(),
            "Interquartile": iqr,
            "Five": five_number_summary,
            "values":column_values,
            "values2":column_values2
        }

        return JsonResponse(result)


def assignment2(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    Attr1="sepal.length"
    Attr2="sepal.width"

    print("boundary 1")

    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    contingency_table = pd.crosstab(df[Attr1], df[Attr2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    alpha=0.7  # we have set that.....
    fl=0

    if p <= alpha:
        print(f"The p-value ({p}) is less than or equal to the significance level ({alpha}).")
        print("The selected attributes are correlated.")
        fl=1
    else:
        print(f"The p-value ({p}) is greater than the significance level ({alpha}).")
        print("The selected attributes are not correlated.")
        fl=0
    
    print(expected)
    correlation_coefficient = df[Attr1].corr(df[Attr2])
    covariance = df[Attr1].cov(df[Attr2])

    min_value = df[Attr1].min()
    max_value = df[Attr1].max()

    df[Attr2] = (df[Attr1] - min_value) / (max_value - min_value)

    mean = df[Attr1].mean()
    std_dev = df[Attr1].std()
    df[Attr2] = (df[Attr1] - mean) / std_dev


    mean = df[Attr1].mean()
    std_dev = df[Attr1].std()
    df[Attr2] = (df[Attr1] - mean) / std_dev

    max_abs = df[Attr1].abs().max()
    df[Attr2] = df[Attr1] / (10 ** len(str(int(max_abs))))



    if fl:
       my_data={
        "name":node[0].name,
        "result":"correlated",
        "p":p,
        "chi2":chi2,
        "dof":dof,
        "a1":Attr1,
        "a2":Attr2
       }

       return JsonResponse(my_data)
    
    my_data={
        "name":node[0].name,
        "result":"not correlated",
        "p":p,
        "chi":chi2,
        "dof":dof,
        "a1":Attr1,
        "a2":Attr2
       }

    return JsonResponse(my_data)



def assignment3(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    strr="info"
    
    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

   
    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']

    if strr=="info":
       clf = DecisionTreeClassifier(criterion='entropy')
    elif strr=="gini":
       clf= DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=42)
    elif strr=="gain":
       clf = DecisionTreeClassifier(criterion='entropy', splitter='best')




    clf.fit(X, Y)
    decision_tree_text = export_text(clf, feature_names=X.columns.tolist())
    tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    tree_image_path = 'static/plot/image.png'
    os.makedirs(os.path.dirname(tree_image_path), exist_ok=True)
    plt.savefig(tree_image_path)
    my_data={"name":file_name,"text":decision_tree_text}
    return JsonResponse(my_data)
               




def assignment3_confuse_matrix(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    misclassification_rate = 1 - accuracy
    sensitivity = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    response_data = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'misclassification_rate': misclassification_rate,
        'sensitivity': sensitivity,
        'precision': precision,
    }

    return JsonResponse(response_data, status=status.HTTP_200_OK)


def assignment4(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    clf.fit(X_train, y_train)


    rules = export_text(clf, feature_names=list(X.columns.tolist()))

    y_pred = clf.predict(X)
    accuracy = accuracy_score(Y, y_pred)

    coverage = len(y_pred) / len(Y) * 100

    rule_count = len(rules.split('\n'))

    my_data = {
        "name":file_name,
        'rules': rules,
        'accuracy': accuracy,
        'coverage': coverage,
        'toughness': rule_count,
    }
    return JsonResponse(my_data)


def assignment5(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']
    my_data={
        "name":file_name
    }
    return JsonResponse(my_data)



###################################################################################################

# views.py

# Chi-Square Value: 1922.9347363945576

# P-Value: 2.6830523867648017e-17

from scipy.stats import chi2_contingency,zscore,pearsonr
import tempfile
from django.shortcuts import render
import json
# Create your views here.
from rest_framework.parsers import FileUploadParser
import csv
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.views import View
import csv
import math
from django.http import JsonResponse
from django.views import View
import csv
from django.http import HttpResponse
import json
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import csv
import statistics
import numpy as np
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views import View
import statistics
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency,zscore,pearsonr
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import datasets
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.tree import export_text
from django.views.decorators.csrf import csrf_exempt
import logging
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import tempfile
import shutil
import math
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse, FileResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

import pandas as pd
import os

import numpy as np
from django.http import JsonResponse
from scipy.stats import chi2_contingency
from scipy.stats import chi2

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import graphviz
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle


class RegressionClass(APIView):
    @method_decorator(csrf_exempt)
    def get(self, request, *args, **kwargs):
        if request.method == 'GET':
            try:
                
                node=CSVFile.objects.all()
                print(node[0].name)

                if len(node)==0 :
                    return HttpResponse("No csv file in database !!")

                print("boundary 1")

                algo="KNN"

                data = pd.read_csv(node[0].file)

                df = pd.DataFrame(data)
                df = shuffle(df, random_state=42)

                target_class = df.columns[-1]

                object_cols = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype=='object' and col != target_class]
                numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != target_class]

                X = df[numeric_cols+object_cols]
                y = df[target_class]

                # print(X.head())
                # print(y.head())
                

                X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
                ordinal_encoder = OrdinalEncoder()

                if target_class in object_cols :
                    object_cols = [col for col in object_cols if col != target_class]
                    y_train[target_class] = OrdinalEncoder.fit_transform(y_train[target_class])
                    y_test[target_class] = OrdinalEncoder.fit_transform(y_test[target_class])

                X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
                X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])


                if algo == "Linear" : 
                    cm = self.logistic_regression(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "Naive" : 
                    cm = self.naive_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "KNN" : 
                    cm = self.knn_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "ANN" : 
                    cm = self.ann_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                
                return JsonResponse({"accuracy": "accuracy"})
            except Exception as e :
                print(e)
                return JsonResponse({"error => ": str(e)}, status=status.HTTP_200_OK, safe=False)

    def preprocess(self, df):
        numerical_columns = df.select_dtypes(include=[int, float])

        # Select columns with a number of unique values less than 4
        unique_threshold = 4
        selected_columns = []
        for column in df.columns:
            if len(df[column].unique()) < unique_threshold and df[column].dtype == 'object':
                selected_columns.append(column)

        # Combine the two sets of selected columns (numerical and unique value threshold)
        final_selected_columns = list(set(numerical_columns.columns).union(selected_columns))

        # Create a new DataFrame with only the selected columns
        filtered_df = df[final_selected_columns]

        from sklearn.preprocessing import LabelEncoder

        # Assuming 'filtered_df' is your DataFrame with object-type columns to be encoded
        encoder = LabelEncoder()

        for column in filtered_df.columns:
            if filtered_df[column].dtype == 'object':
                filtered_df[column] = encoder.fit_transform(filtered_df[column])
        
        return filtered_df


    def logistic_regression(self, X, y, X_train, X_test, y_train, y_test):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # Load your dataset (e.g., Iris or Breast Cancer)
        # X, y, X_train, X_test, y_train, y_test = load_data()
        # Split the data into training and testing sets
        
        # Create and train the regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)

        print(y_pred.shape)
        print(y_test.shape)
        cm = confusion_matrix(y_test, y_pred).tolist()
        print(cm)

        
        return cm

    def naive_classifier(self, X, y, X_train, X_test, y_train, y_test):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score

        

        # Create and train the Naïve Bayes classifier
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = nb_classifier.predict(X_test)

        # Calculate accuracy
        cm = confusion_matrix(y_test, y_pred).tolist()

        return cm

    def knn_classifier(self, X, y, X_train, X_test, y_train, y_test):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        

        # Create and train the k-NN classifier with different values of k
        k_values = [1, 3, 5, 7]
        cm = []
        accuracy_scores = []


        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            
            # Make predictions
            y_pred = knn_classifier.predict(X_test)
            
            # Calculate accuracy
            metrix = confusion_matrix(y_test, y_pred).tolist()
            cm.append({'confusion_matrix': metrix})


            print(metrix)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # Plot the error graph
        plt.figure(figsize=(8, 6))  # Adjust figure size as needed
        plt.plot(k_values, accuracy_scores)
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        # plt.show()

        plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\KNNplot.png")
    
        return cm

    def ann_classifier(self, X, y, X_train, X_test, y_train, y_test):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_iris, load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier

        

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the ANN classifier
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)

        # Plot the error graph (iteration vs error)
        plt.figure(figsize=(8, 6))
        plt.plot(mlp.loss_curve_)
        plt.title('Error Graph (Iteration vs Error)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        # plt.show()

        plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\ANNplot.png")

        # Evaluate the classifier
        y_pred = mlp.predict(X_test)

        cm = confusion_matrix(y_test, y_pred).tolist()

        return cm