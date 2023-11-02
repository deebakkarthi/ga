#!/usr/bin/env python3

import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy


class TreeType(int):
    def __new__(cls):
        return super(TreeType, cls).__new__(cls, random.choice([0, 1, 2]))


class PopSize(int):
    def __new__(cls):
        return super(PopSize, cls).__new__(cls, random.choice([100, 200, 300]))


class TreeGenMethod(int):
    def __new__(cls):
        return super(TreeGenMethod, cls).__new__(cls, random.choice([0, 1, 2]))


class InitialDepth(int):
    def __new__(cls, decicion=False):
        if decicion:
            return super(InitialDepth, cls).__new__(cls, random.randint(2, 8))
        else:
            return super(InitialDepth, cls).__new__(cls, random.randint(2, 15))


class OffspringDepth(int):
    def __new__(cls, decicion=False):
        if decicion:
            return super(OffspringDepth, cls).__new__(cls, random.randint(2, 8))
        else:
            return super(OffspringDepth, cls).__new__(cls, random.randint(2, 15))


class SelectionMethod(int):
    def __new__(cls):
        return super(SelectionMethod, cls).__new__(cls, random.choice([0, 1]))


class SelectionSize(int):
    def __new__(cls):
        return super(SelectionSize, cls).__new__(cls, random.randint(2, 10))


class ReproductionRate(int):
    def __new__(cls):
        return super(ReproductionRate, cls).__new__(cls, random.randint(0, 100))


class MutType(int):
    def __new__(cls):
        return super(MutType, cls).__new__(cls, random.choice([0, 1]))


class MaxMutDepth(int):
    def __new__(cls):
        return super(MaxMutDepth, cls).__new__(cls, random.randint(2, 6))


class ControlFlow(int):
    def __new__(cls):
        return super(ControlFlow, cls).__new__(cls, random.choice([0, 1]))


class OperatorCombination(int):
    def __new__(cls):
        return super(OperatorCombination, cls).__new__(cls, random.randint(0, 6))


class FitnessType(int):
    def __new__(cls):
        return super(FitnessType, cls).__new__(cls, random.randint(0, 4))


class NrGeneration(int):
    def __new__(cls):
        return super(NrGeneration, cls).__new__(cls, random.randint(50, 200))


def individual_create():
    """
    tree_type - ARITHMETIC, LOGICAL, DECISION
    pop_size - 100, 200, 300
    tree_gen_method - FULL, GROW, HALF_AND_HALF
    initial_depth- [2, 15] [2, 8] for decision trees
    offspring_depth - [2, 15] [2, 8] for decision trees
    selection_size - [2, 10]
    reproduction_rate - [0, 100]
    mut_type - GROW, SHRINK
    max_mut_depth - [2, 6]
    control_flow - FIXED, RANDOM
    op_combination - [0, 6] CX_MUT, CX, MUT, CX_MUT_PRESET, MUT_RAND_CX, CX_RAND_MUT, CREATE
    fitness_type - [0, 4] ACC, F_SCORE, W_ACC, RAND_W_ACC, TPR
    nr_gen - [50, 200]
    """
    tmp = []
    tmp.append(TreeType())
    tmp.append(PopSize())
    tmp.append(TreeGenMethod())
    tmp.append(InitialDepth(decicion=(tmp[0] == 2)))
    tmp.append(OffspringDepth(decicion=(tmp[0] == 2)))
    tmp.append(SelectionMethod())
    tmp.append(SelectionSize())
    tmp.append(ReproductionRate())
    tmp.append(MutType())
    tmp.append(MaxMutDepth())
    tmp.append(ControlFlow())
    tmp.append(OperatorCombination())
    tmp.append(FitnessType())
    tmp.append(NrGeneration())
    return tmp


def mutFlipValue(individual, indpb=0.05):
    for idx, val in enumerate(individual):
        if random.random() < indpb:
            individual[idx] = type(val)()
    return (individual,)


def eval_autoga(individual):
    print(individual)
    return (sum(individual),)


def main():
    random.seed(42)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, individual_create
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_autoga)
    toolbox.register("mate", tools.cxUniform, indpb=0.8)
    toolbox.register("mutate", mutFlipValue, indpb=0.05)
    toolbox.register("select", tools.selRoulette)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=40,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )
    return


if __name__ == "__main__":
    main()
