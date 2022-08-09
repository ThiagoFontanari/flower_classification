# Importing the necessary modules
# Realizando a importação dos módulos necessários
from pandas import read_csv as rd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Creating a dataset with the CSV file and verifying its information
# Criando um dataset com o arquivo CSV e verificando suas informações
dataset = rd("iris.csv")
print("\n ->Checking the shape of the dataset\n ->Verificando o shape do dataset\n\n",dataset.head,"\n")

# Counting how many samples of each species there are in the dataset, to check the balance of the samples
# Contando quantas amostras de cada espécie existem no dataset, para verificar o balanceamento das amostras
print(" ->Counting the elements of each class in the dataset\n ->Contagem ddos elementos de cada classe do dataset\n")
print(dataset[['Species']].value_counts())

# Splitting the dataset into training and testing data, at a ratio of 30%-70%
# In the variable x, the data of the formats were stored
# In the variable y, the species classifications of the flowers were stored

# Dividindo o dataset em dados de treino e teste, à uma proporção de 30%-70%
# Na variável x, foram armazenados os dados dos formatos
# Na variável y, foram armazenadas as classificações de éspecie das flores
x = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = dataset[['Species']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Implementation of the classification model, in this case using the Decision Tree Classifier
# Implementação do modelo de classificação, neste caso utilizando a Decision Tree Classifier 
modelo = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=3, random_state=1)

# Building the decision tree, based on the training set
# Construindo a árvore de decisão, baseado no conjunto de treinamento
modelo.fit(x_train, y_train)

# Generating predictions using the model, using the separate data for testing
# Gerando previsões utilizando o modelo, utilizando os dados separados para teste
previsoes = modelo.predict(x_test)

# Calculating the model's accuracy, using the generated predictions and comparing with the test data
# Calculando a acurácia do modelo, utilizando as previsões geradas e comparando com os dados de teste
print("\n ->Checking the model's prediction accuracy after training\n ->Verificando a acurácia de previsão do modelo após o treinamento\n")
print("Model's accuracy | Acurácia do modelo: ",accuracy_score(y_test, previsoes),"\n")

# Calculating and displaying the accuracy of predictions made by class
# Calculando e exibindo a acurácia das previsões feitas por classe
print("\n ->Prediction accuracy for each class\n" " ->Acurácia de previsão para cada classe\n")
print("\n",classification_report(y_test, previsoes))




