# -*- coding: utf-8 -*-
import random
import numpy 
import json
from deap import creator, base, tools, algorithms

def algoritmo_genetico(ind_size, pop_size, cali):
    # Creación de tipos
    creator.create("FitnessMax", base.Fitness, weights = (1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMax)
    
    # Función de fitness
    def evalFitness(individual):
        suma = 0
        for index, elem in enumerate(individual):
            if(elem == 1):
                properties = cali['features'][index]['properties']
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
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
    toolbox.register("select", tools.selBest)
    
    # Algoritmo genético
    pop = toolbox.population()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.1, ngen = 50, stats = stats, verbose = False)
    
    return pop, logbook

def puntos_de_recarga(pop, cali):
    # Filtrado de población
    result = {}
    result['type'] = cali['type']
    result['crs'] = cali['crs']
    result['features'] = []
    
    for index, elem in enumerate(pop[0]):
        if(elem == 1):
            feature = cali['features'][index]
            result['features'].append(feature)        
    
    result['crs']['properties']['name'] = "urn:ogc:def:crs:EPSG::4326"
    
    # Escribir el archivo de salida
    with open('estaciones_de_recarga.JSON', "w") as output_file:
        json.dump((result), output_file, indent = 3)
        
def main():
    with open('calificaciones_filtrado.JSON', "r") as input_file:
        cali = json.load(input_file)
    
    # Tamaño del cromosoma y población
    ind_size = len(cali['features'])
    pop_size = 1
    
    pop, logbook = algoritmo_genetico(ind_size, pop_size, cali)
    puntos_de_recarga(pop, cali)
    
    print('POBLACIÓN FINAL')
    print(pop)
    
    print('LOGBOOK')
    print(logbook)
    
if __name__ == "__main__":
    main()