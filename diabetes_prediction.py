import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#importing  the dataset.
dataset = pd.read_csv("C:/Users/Sarvesh A Behare/Desktop/NITD/mentor/dataset/diabetes_prediction_dataset.csv")

#printing few starting data of dataset.
print("\nDataset: \n",dataset.head())


#computing summary of the dataset which includes count, mean, max, min, standard deviation etc.
summary_stats = dataset.describe()
print("\nSummary Stats\n",summary_stats)


# keeping the "No Info" in the smoking  history category as while trying to replace it with the mode of the dataset
# we are getting No Info as the mode of the dataset and it would be better to keep it.


#changing the ordinal data to the numeric data.
label_encoder = LabelEncoder()
dataset['smoking_history'] = label_encoder.fit_transform(dataset['smoking_history'])
dataset['gender'] = label_encoder.fit_transform(dataset['gender'])


#normalizing the numeric features
scaler = MinMaxScaler()
columns_to_normalize = ['age','smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
dataset[columns_to_normalize] = scaler.fit_transform(dataset[columns_to_normalize])


print("\nDataset: \n",dataset.head())

num_features = len(dataset.columns) - 1  # Exclude the target variable

population_size = 10
num_generations = 30

# Function to calculate fitness score
def fitness_score(chromosome):
    selected_features = [i for i in range(num_features) if chromosome[i] == 1]
    x = dataset.iloc[:, :-1].values[:, selected_features]
    y = dataset.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    y_predicted = model.predict(x_test)
    
    fitness = accuracy_score(y_test, y_predicted)
    
    return fitness

#making random population
populations = [[random.randint(0, 1) for _ in range(num_features)] for _ in range(population_size)]

for generation in range(num_generations):
    print("Generation:", generation + 1)
    fitness_values = [fitness_score(chromosome) for chromosome in populations]
    indices = np.argsort(fitness_values)[-2:]
    new_chromosomes = [populations[i] for i in indices]
    print("new chromosomes:", new_chromosomes)
    
    # Crossover and mutation (single-point crossover and bit-flip mutation)
    new_population = []
    for _ in range(population_size // 2):
        parent1, parent2 = new_chromosomes
        crossover_point = random.randint(0, num_features - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        # Mutation
        mutation_rate = 0.1
        for i in range(num_features):
            if random.random() < mutation_rate:
                child1[i] = 1 - child1[i]
            if random.random() < mutation_rate:
                child2[i] = 1 - child2[i]
        
        new_population.extend([child1, child2])
    
    populations = new_population

# Select the best chromosome as the solution
best_chromosome = max(populations, key=fitness_score)
print("Best chromosome:", best_chromosome)
fit = fitness_score(best_chromosome)
print("Accuracy is :", fit)