#!/usr/bin/env python3

import operator
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import gp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from functools import partial
import multiprocessing
import numpy as np
import pandas as pd


df = pd.read_csv("./merged.csv")
X = df.iloc[:, :-1]
X = X.astype(np.float64)
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y)


def individual_create():
    """
    00 tree_type - ARITHMETIC
    01 pop_size - 100
    02 tree_gen_method - FULL, GROW, HALF_AND_HALF
    03 initial_depth- [2, 15] [2, 8] for decision trees
    04 offspring_depth - [2, 15] [2, 8] for decision trees
    05 selection_method = FITNESS_PROP, TOURNAMENT
    06 selection_size - [2, 10]
    07 reproduction_rate - [0, 100]
    08 mut_type - GROW, SHRINK
    09 max_mut_depth - [2, 6]
    10 control_flow - FIXED, RANDOM
    11 op_combination - [0, 6] CX_MUT, CX, MUT, CX_MUT_PRESET, MUT_RAND_CX, CX_RAND_MUT, CREATE
    12 fitness_type - [0, 3] ACC, F_SCORE, W_ACC, TPR
    13 nr_gen - 50
    """
    tmp = np.empty(14, dtype=int)
    # TODO do decision tree later
    # change this to (0, 2)
    # tmp[0] = random.randint(0, 1)
    tmp[0] = 0
    tmp[1] = 50
    tmp[2] = random.randint(0, 2)
    if tmp[0] == 2:
        tmp[3] = random.randint(2, 8)
        tmp[4] = random.randint(2, 8)
    else:
        tmp[3] = random.randint(2, 15)
        tmp[4] = random.randint(2, 15)
    tmp[5] = random.randint(0, 1)
    tmp[6] = random.randint(2, 10)
    tmp[7] = random.randint(0, 100)
    tmp[8] = random.randint(0, 1)
    tmp[9] = random.randint(2, 6)
    tmp[10] = random.randint(0, 1)
    tmp[11] = random.randint(0, 6)
    # tmp[12] = random.randint(0, 3)
    tmp[12] = 0
    tmp[13] = 50
    return tmp


def mutFlipValue(individual, indpb=0.05):
    tmp = individual_create()
    for idx, _ in enumerate(individual):
        if random.random() < indpb:
            individual[idx] = tmp[idx]
    return (individual,)


def protected_div(left, right):
    if right != 0:
        return left / right
    else:
        return 1


def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


def pset_create():
    pset = gp.PrimitiveSetTyped("main", [type(X[col][0]) for col in X], bool, "IN")
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.not_, [bool], bool)

    pset.addPrimitive(operator.add, [np.float64, np.float64], np.float64)
    pset.addPrimitive(operator.sub, [np.float64, np.float64], np.float64)
    pset.addPrimitive(operator.mul, [np.float64, np.float64], np.float64)
    pset.addPrimitive(protected_div, [np.float64, np.float64], np.float64)

    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)
    # Add this constant to prevent no primitive found error
    # This occurs because there may be an odd number of inputs and all the functions are binary
    pset.addEphemeralConstant(
        name="RAND",
        ephemeral=partial(np.float64, random.random()),
        ret_type=np.float64,
    )
    return pset


def inner_evaluate(individual, tree_type, fitness_type, toolbox):
    func = toolbox.compile(expr=individual)
    y_pred = [func(*data) for _, data in X_train.iterrows()]
    y_labels = np.unique(y)
    # Output of arithmetic trees is float
    if tree_type == 0:
        y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
    elif tree_type == 1:
        y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
    # TODO Implement for decision trees
    else:
        raise ValueError("tree_type should be in [0, 2]")

    if fitness_type == 0:
        ret = accuracy_score(y_true=y_train, y_pred=y_pred)
    elif fitness_type == 1:
        ret = f1_score(y_train, y_pred)
    elif fitness_type == 2:
        ret = (accuracy_score(y_train, y_pred) + f1_score(y_train, y_pred)) / 2
    elif fitness_type == 3:
        ret = recall_score(y_train, y_pred)
    else:
        raise ValueError("fitness_type should be in [0, 3]")

    return (ret,)


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    use_varAnd=True,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print("\t", logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        if use_varAnd:
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        else:
            offspring = algorithms.varOr(
                offspring, toolbox, len(offspring), cxpb, mutpb
            )

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            tmp = logbook.select("max")[-1]
            if tmp > 0.89:
                if halloffame is not None:
                    print("\t", halloffame[0], test(halloffame[0], toolbox))
            print("\t", logbook.stream)

    return population, logbook


# TODO implement accuracy as the fitness measure
def eval_autoga(individual):
    pset = pset_create()
    creator.create("InnerFitnessMax", base.Fitness, weights=(1.0,))
    creator.create("InnerIndividual", gp.PrimitiveTree, fitness=creator.InnerFitnessMax)

    inner_toolbox = base.Toolbox()

    if individual[2] == 0:
        inner_toolbox.register(
            "expr", gp.genFull, pset=pset, min_=2, max_=individual[3]
        )
    elif individual[2] == 1:
        inner_toolbox.register(
            "expr", gp.genGrow, pset=pset, min_=2, max_=individual[3]
        )
    elif individual[2] == 2:
        inner_toolbox.register(
            "expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=individual[3]
        )
    else:
        raise ValueError("tree_gen_method should be from [0, 1, 2]")

    inner_toolbox.register(
        "individual", tools.initIterate, creator.InnerIndividual, inner_toolbox.expr
    )
    inner_toolbox.register(
        "population", tools.initRepeat, list, inner_toolbox.individual
    )
    inner_toolbox.register("compile", gp.compile, pset=pset)

    inner_toolbox.register(
        "evaluate",
        inner_evaluate,
        fitness_type=individual[12],
        toolbox=inner_toolbox,
        tree_type=individual[0],
    )
    if individual[5] == 0:
        inner_toolbox.register("select", tools.selRoulette)
    elif individual[5] == 1:
        inner_toolbox.register("select", tools.selTournament, tournsize=individual[6])
    else:
        raise ValueError("selection_method should be from [0, 1]")

    inner_toolbox.register("mate", gp.cxOnePoint)
    inner_toolbox.register("expr_mut", gp.genFull, min_=2, max_=individual[9])

    if individual[8] == 0:
        inner_toolbox.register(
            "mutate", gp.mutUniform, expr=inner_toolbox.expr_mut, pset=pset
        )
    elif individual[8] == 1:
        inner_toolbox.register("mutate", gp.mutShrink)
    else:
        raise ValueError("mut_type should be from [0, 1]")

    limitHeight = individual[4]
    inner_toolbox.decorate(
        "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitHeight)
    )
    inner_toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=limitHeight),
    )

    use_varAnd = True
    if individual[11] == 0:
        cxpb = individual[7]
        mutpb = 100 - cxpb
        mutpb = float(mutpb / 100)
        cxpb = float(cxpb / 100)
    elif individual[11] == 1:
        cxpb = 1
        mutpb = 0
    elif individual[11] == 2:
        cxpb = 0
        mutpb = 1
    elif individual[11] == 3:
        cxpb = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])
        mutpb = 100 - cxpb
        mutpb = float(mutpb / 100)
        cxpb = float(cxpb / 100)
    elif individual[11] == 4:
        cxpb = random.random()
        mutpb = 1
    elif individual[11] == 5:
        cxpb = 1
        mutpb = random.random()
    elif individual[11] == 6:
        cxpb = 0
        mutpb = 0
        use_varAnd = False
    else:
        raise ValueError("op_combination should be in [0, 6]")

    pop = inner_toolbox.population(n=individual[1])
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, logbook = eaSimple(
        pop,
        inner_toolbox,
        cxpb,
        mutpb,
        individual[13],
        stats,
        halloffame=hof,
        use_varAnd=use_varAnd,
    )
    func = gp.compile(hof[0], pset)
    y_pred = [func(*data) for _, data in X_test.iterrows()]
    y_labels = y_test.unique()
    if individual[0] == 0:
        y_pred = list(map(lambda x: y_labels[0] if x > 0 else y_labels[1], y_pred))
    elif individual[0] == 1:
        y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
    # TODO Implement for decision trees
    else:
        raise ValueError("tree_type should be in [0, 2]")
    ret = accuracy_score(y_test, y_pred)
    if ret > 0.89:
        print(hof[0])
    del creator.InnerIndividual
    del creator.InnerFitnessMax
    return (ret,)


def test(individual, toolbox):
    func = toolbox.compile(expr=individual)
    y_pred = [func(*data) for _, data in X_test.iterrows()]
    y_labels = np.unique(y)
    y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    pool = multiprocessing.Pool()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, individual_create
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_autoga)
    toolbox.register("mate", tools.cxUniform, indpb=0.80)
    toolbox.register("mutate", mutFlipValue, indpb=0.10)
    # TODO Implement elitism
    toolbox.register("select", tools.selRoulette)
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=5,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )
    print(hof[0])
