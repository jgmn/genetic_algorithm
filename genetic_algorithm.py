# -*- coding: utf-8 -*-
"""
@author: J. Gerardo Moreno N.
"""
import random
import numpy 
import json
import geopandas as gpd
import matplotlib.pyplot as plt
from folium import Map, GeoJson, LayerControl
from datetime import datetime
from time import time 
from deap import creator, base, tools, algorithms

#----------------------EXECUTE_GENETIC_ALGORITHM------------------------------#
# DESCRIPCIÓN: Ejecuta la versión estática del algoritmo genético.            #
# PARÁMETROS:                                                                 #
#   ENTRADA: ind_size: Tamaño de los individuos.                              #
#            pop_size: Tamaño de la población.                                #
#            cxpb: Probabilidad de cruzar dos individuos.                     #
#            mutpb: Probabilidad de mutar un individuo.                       #
#            ngen: Número de generaciones.                                    #
#            pdi_df: GeoDataFrame Puntos de Interés.                          #
#            voro_df: GeoDataFrame Voronoi.                                   # 
#   SALIDA:  pop: Población final.                                            #
#            logobook: Estadísticas del algoritmo genético.                   #
#            hof: Mejor individuo de la población.                            #   
#-----------------------------------------------------------------------------#
def execute_genetic_algorithm(ind_size, pop_size, cxpb, mutpb, ngen, pdi_df, voro_df):
    # Creación de tipos.
    creator.create("FitnessMax", base.Fitness, weights = (1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMax)
    
    # Precálculo de totales. 
    pob_total = sum(pdi_df['poblacion'])
    trafico_total = sum(pdi_df['trafico'])
    tweets_total = sum(pdi_df['tweets'])
    tiempo_total = sum(pdi_df['tiempo'])
    area_total = sum(voro_df['geometry'].area) #El área está en m^2.
    
    # Función de fitness.
    def evalFitness(individual):
        utilidad, poblacion, trafico, tweets, tiempo = 0, 0, 0, 0, 0
        poblacion, trafico, tweets, tiempo, area = 0, 0, 0, 0, 0
        utilidad, coste, coste_unitario = 0, 0, 1 
        num_estaciones = sum(individual) / ind_size

        for index, elem in enumerate(individual):
            if(elem == 1):
                poblacion += pdi_df['poblacion'][index] / pob_total
                trafico += pdi_df['trafico'][index] / trafico_total
                tweets += pdi_df['tweets'][index] / tweets_total
                tiempo += pdi_df['tiempo'][index] / tiempo_total
                area += voro_df['geometry'][index].area / area_total

        utilidad += poblacion + trafico + tiempo + tweets 
        coste = area + coste_unitario * num_estaciones
        utilidad = utilidad - coste

        return utilidad,
    
    # Inicialización.
    toolbox = base.Toolbox()
    toolbox.register("bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.bit, ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, pop_size)
    
    # Operadores.
    toolbox.register("evaluate", evalFitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
    toolbox.register("select", tools.selTournament, tournsize = 3)
    
    # Algoritmo genético.
    pop = toolbox.population()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = cxpb, mutpb = mutpb, ngen = ngen, stats = stats, halloffame = hof, verbose = False)

    return pop, logbook, hof
#-----------------------------------------------------------------------------#

#-------------------------SAVE_CHARGING_STATIONS------------------------------#
# DESCRIPCIÓN: Guarda las estaciones de recarga en un archivo json.           #
# PARÁMETROS:                                                                 #
#   ENTRADA: best: El mejor individuo de la población.                        #
#            pdi_df: GeoDataFrame Puntos de Interés.                          #
#            date: Fecha.                                                     #    
#   SALIDA:  estaciones_de_recarga_<date>.json.                               #
#-----------------------------------------------------------------------------#
def save_charging_stations(best, pdi_df, date):
    for index, elem in enumerate(best[0]):
        if(elem == 0):
            pdi_df.drop(index, axis = 0, inplace = True)
    
    pdi_df = pdi_df.to_crs({'init': 'epsg:4326'})
    pdi_df.to_file('estaciones_de_recarga_'+date+'.json', driver = "GeoJSON")
#-----------------------------------------------------------------------------#

#------------------------------SAVE_LOGBOOK-----------------------------------#
# DESCRIPCIÓN: Guarda las estadísticas del algoritmo genético en un archivo   #
#              json.                                                          #
# PARÁMETROS:                                                                 #
#   ENTRADA: logbook: Estadísticas del algoritmo genético.                    #
#            date: Fecha.                                                     #    
#   SALIDA:  logbook_<date>.json.                                             #
#-----------------------------------------------------------------------------#    
def save_logbook(logbook, date):   
    logbook_json = {}
    logbook_json['log'] = logbook
    
    with open("logbook_"+date+".json", "w") as output_file:
        json.dump(logbook_json, output_file, indent = 3)
#-----------------------------------------------------------------------------#

#-------------------------------SAVE_GRAPH------------------------------------#
# DESCRIPCIÓN: Guarda la gráfica de evolución de la función objetivo a        #
#              a través de las generaciones.                                  #
# PARÁMETROS:                                                                 #
#   ENTRADA: logbook: Estadísticas del algoritmo genético.                    #
#            date: Fecha.                                                     #    
#   SALIDA:  grafica_fitness_<date>.png.                                      #
#-----------------------------------------------------------------------------#         
def save_graph(logbook, date): 
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
        
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_max, "b-", label = "Función Objetivo")
    ax1.set_xlabel("Generaciones")
    ax1.set_ylabel("Función Objetivo", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
        
    lns = line1
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    
    fig.savefig("grafica_fitness_"+date+".png")
#-----------------------------------------------------------------------------#

#--------------------------------SAVE_MAP-------------------------------------#
# DESCRIPCIÓN: Guarda el mapa web para visualizar el diagrama de Voronoi y    #
#              las estaciones de recarga.                                     #
# PARÁMETROS:                                                                 #
#   ENTRADA: logbook: Estadísticas del algoritmo genético.                    #
#            date: Fecha.                                                     #    
#   SALIDA:  valencia_<date>.html.                                            #
#-----------------------------------------------------------------------------#      
def save_map(date):
    valencia = [39.4561165311493, -0.3545661635]
    mapa = Map(location = valencia, tiles = 'OpenStreetMap', zoom_start = 10)
    GeoJson(open('voronoi.json'), name = 'Diagrama de Voronoi').add_to(mapa)
    GeoJson(open('estaciones_de_recarga_'+date+'.json'), name = 'Estaciones de Recarga').add_to(mapa)
    LayerControl().add_to(mapa)
    mapa.save('valencia_'+date+'.html')
#-----------------------------------------------------------------------------#
    
def main():
    date = str(datetime.now())
    date = date.replace("-", "")
    date = date.replace(":", "")
    date = date.replace(".", "")
    date = date.replace(" ", "")

    # Leer archivos de entrada.
    pdi_df = gpd.read_file('puntos_de_interes.json')
    voro_df = gpd.read_file('voronoi.json')

    # Cambiar el formato de coordenadas para el cálculo de áreas.
    voro_df = voro_df.to_crs({'init': 'epsg:25830'})
    
    # Definir la configuración del algoritmo genético. 
    ind_size = len(pdi_df)
    pop_size = 300
    cxpb, mutpb, ngen = 0.5, 0.2, 100

    # Ejecutar algoritmo genético.
    pop, logbook, best = execute_genetic_algorithm(ind_size, pop_size, cxpb, mutpb, ngen, pdi_df, voro_df)

    # Guardar los resultados del algoritmo genético. 
    save_charging_stations(best, pdi_df, date)
    save_logbook(logbook, date)
    save_graph(logbook, date)
    save_map(date)

    # Desplegar resultados del algoritmo genético.
    print('POBLACIÓN FINAL')
    print(pop)
    
    print('\nMEJOR INDIVIDUO')
    print(best)
    
    print('\nLOGBOOK')
    print(logbook)
    
if __name__ == "__main__":
    # Calcular tiempo de ejecución.
    tiempo_inicial = time()
    main()
    tiempo_final = time()
    tiempo_ejecucion = tiempo_final - tiempo_inicial
    print('\nTiempo de ejecución: ', '%.2f'% (tiempo_ejecucion/60), 'minutos')
