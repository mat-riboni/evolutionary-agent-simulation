import numpy as np
import copy
import random
from typing import List
import settings as c
from brain import Brain

def evolve_population(agents: List) -> List[Brain]:
    ELITISM_COUNT = 2
    TOURNAMENT_SIZE = 3
    
    agents.sort(key=lambda x: x.fitness, reverse=True)
    new_brains = []
    
    for i in range(ELITISM_COUNT):
        new_brains.append(copy.deepcopy(agents[i].brain))
        
    while len(new_brains) < len(agents):
        parent1 = tournament_selection(agents, size=TOURNAMENT_SIZE)
        parent2 = tournament_selection(agents, size=TOURNAMENT_SIZE)
        
        child_brain = crossover(parent1.brain, parent2.brain)
        mutate_brain(child_brain)
        
        new_brains.append(child_brain)
        
    return new_brains

def tournament_selection(population: List, size: int = 3):
    tournament = random.sample(population, size)
    return max(tournament, key=lambda x: x.fitness)

def crossover(brain1: Brain, brain2: Brain) -> Brain:
    child = Brain(c.INPUT_SIZE, c.HIDDEN_SIZE, c.OUTPUT_SIZE)
    alpha = random.uniform(0.0, 1.0)
    
    def blend(mat1, mat2):
        return (mat1 * alpha) + (mat2 * (1.0 - alpha))

    child.w1 = blend(brain1.w1, brain2.w1)
    child.b1 = blend(brain1.b1, brain2.b1)
    
    child.w2 = blend(brain1.w2, brain2.w2)
    child.b2 = blend(brain1.b2, brain2.b2)
    
    child.w3 = blend(brain1.w3, brain2.w3)
    child.b3 = blend(brain1.b3, brain2.b3)
    
    return child

def mutate_brain(brain: Brain) -> None:
    def mutate_matrix(mat):
        mask_fine = np.random.rand(*mat.shape) < c.MUTATION_RATE 
        noise_fine = np.random.randn(*mat.shape) * (c.MUTATION_STRENGTH * 0.5) 
        mat[mask_fine] += noise_fine[mask_fine]
        
        mask_shock = np.random.rand(*mat.shape) < (c.MUTATION_RATE * 0.1)
        noise_shock = np.random.randn(*mat.shape) * (c.MUTATION_STRENGTH * 5.0)
        mat[mask_shock] += noise_shock[mask_shock]

    mutate_matrix(brain.w1); mutate_matrix(brain.b1)
    mutate_matrix(brain.w2); mutate_matrix(brain.b2)
    mutate_matrix(brain.w3); mutate_matrix(brain.b3)