# -*- coding: utf-8 -*-
import random
import json
#import geopandas as gpd
from deap import creator, base, tools, algorithms

# Creación de tipos
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

#Tamaño del cromosoma
ind_size = 1

# Obtener el tamaño de la población
path_input_file = 'calificaciones_filtrado.JSON'
with open(path_input_file,"r") as input_file:
    data = json.load(input_file)
pop_size = len(data['features'])

#cali = gpd.read_file('calificaciones_filtrado.JSON') 
#pop_size = len(cali) 

# Función de Fitness
def evalFitness(individual):    
    return sum(individual),

# Inicialización
toolbox = base.Toolbox()
toolbox.register("bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.bit, pop_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, ind_size)

# Operadores
toolbox.register("evaluate", evalFitness)
toolbox.register("mate", tools.cxTwoPoint, indpb=0.5) #http://deap.readthedocs.io/en/master/api/tools.html#operators
toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
toolbox.register("select", tools.selBest)

# Algoritmo Genético 
pop = toolbox.population()
cxpb, mutpb, ngen = 0.5, 0.1, 50
res = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, verbose = False) #http://deap.readthedocs.io/en/1.0.x/api/algo.html#complete-algorithms

print('---POBLACIÓN FINAL---')
print(pop)
