# -*- coding: utf-8 -*-
import random
import numpy 
import json
import matplotlib.pyplot as plt
from datetime import datetime
from deap import creator, base, tools, algorithms

def execute_genetic_algorithm(ind_size, pop_size, cali):
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
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.1, ngen = 100, stats = stats, verbose = False)

    return pop, logbook

def save_charging_stations(pop, cali, date):
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
    with open("estaciones_de_recarga_"+date+".JSON", "w") as output_file:
        json.dump((result), output_file, indent = 3)
        
def save_logbook(logbook, date):   
    logbook_json = {}
    logbook_json['log'] = logbook
    
    with open("logbook_"+date+".JSON", "w") as output_file:
        json.dump(logbook_json, output_file, indent = 3)
        
def save_graph(logbook, date): 
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")
        
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_max, "b-", label = "Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Max Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, fit_avg, "r-", label = "Average Fitness")
    ax2.set_ylabel("Avg Fitness", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")
    
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc = None)
    
    fig.savefig("grafica_fitness_"+date+".PNG")
    #plot.show()
        
def main():
    with open('calificaciones_filtrado.JSON', "r") as input_file:
        cali = json.load(input_file)
    
    # Tamaño del cromosoma y población
    ind_size = len(cali['features'])
    pop_size = 1
    
    pop, logbook = execute_genetic_algorithm(ind_size, pop_size, cali)
    
    date = str(datetime.now())
    date = date.replace("-", "")
    date = date.replace(":", "")
    date = date.replace(".", "")
    date = date.replace(" ", "")
    
    save_charging_stations(pop, cali, date)
    save_logbook(logbook, date)
    save_graph(logbook, date)
        
    print('POBLACIÓN FINAL')
    print(pop)
    
    print('\nLOGBOOK')
    print(logbook)
    
if __name__ == "__main__":
    main()