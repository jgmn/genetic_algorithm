# -*- coding: utf-8 -*-
import random
import geopandas as gpd
from deap import creator, base, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Obtener el tamaño del cromosoma
cali = gpd.read_file('calificaciones_filtrado.JSON') 
ind_size = len(cali)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de Fitness
def evaluateModel(individual):
    return sum(individual),

toolbox.register("evaluate", evaluateModel)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

ngen = 100
pop_size = 1
pop = toolbox.population(pop_size)
res = algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.1, ngen = ngen, verbose = False) #http://deap.readthedocs.io/en/1.0.x/api/algo.html#complete-algorithms

print('---POBLACIÓN FINAL---')
print(pop)