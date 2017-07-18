# -*- coding: utf-8 -*-
import random
import numpy 
import json
from deap import creator, base, tools, algorithms

# Creación de tipos
creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

#Tamaño de la población
pop_size = 1

# Obtener el tamaño del cromosoma
with open('calificaciones_filtrado.JSON', "r") as input_file:
    data = json.load(input_file)

ind_size = len(data['features'])

# Función de fitness
def evalFitness(individual):
    suma = 0
    for index, elem in enumerate(individual):
        if(elem == 1):
            properties = data['features'][index]['properties']
            poblacion = properties['poblacion']
            trafico = properties['trafico']
            tiempo = properties['tiempo_medio']
            tweets = properties['tweets']
            suma = suma + poblacion + trafico + tiempo + tweets
    return suma,

# Inicialización
toolbox = base.Toolbox()
toolbox.register("bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.bit, ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, pop_size)

# Operadores
toolbox.register("evaluate", evalFitness)
toolbox.register("mate", tools.cxTwoPoint, indpb = 0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
toolbox.register("select", tools.selBest)

# Algoritmo genético
pop = toolbox.population()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", numpy.min)
stats.register("avg", numpy.mean)
stats.register("max", numpy.max)
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.1, ngen = 50, stats = stats, verbose = False)

print('---POBLACIÓN FINAL---')
print(pop)

print('---LOGBOOK---')
print(logbook)

pop=pop[0]

# Filtrado de población
result = {}
result['type'] = data['type']
result['crs'] = data['crs']
result['features'] = []

for index, elem in enumerate(pop):
    if(elem == 1):
        feature = data['features'][index]
        result['features'].append(feature)        

result['crs']['properties']['name'] = "urn:ogc:def:crs:EPSG::4326"

# Escribir el archivo de salida
path_output_file = 'calificaciones_filtrado.JSON'

with open(path_output_file, "w") as output_file:
    json.dump((result), output_file, indent = 3)
