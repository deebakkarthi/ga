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
import numpy as np
import pandas as pd


df = pd.read_csv("./merged.csv")
X = df.iloc[:, :-1]
X = X.astype(np.float64)
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y)


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

    pset.addPrimitive(operator.lt, [np.float64, np.float64], bool)
    pset.addPrimitive(operator.eq, [np.float64, np.float64], bool)
    pset.addPrimitive(if_then_else, [bool, np.float64, np.float64], float)

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
        print(logbook.stream)

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
            print(logbook.stream)

    return population, logbook


def test(individual, toolbox):
    func = toolbox.compile(expr=individual)
    y_pred = [func(*data) for _, data in X_test.iterrows()]
    y_labels = np.unique(y)
    y_pred = list(map(lambda x: y_labels[0] if x else y_labels[1], y_pred))
    return accuracy_score(y_test, y_pred)


# TODO implement accuracy as the fitness measure
if __name__ == "__main__":
    # individual = individual_create()
    # individual = [0, 100, 2, 3, 2, 1, 5, 28, 0, 2, 0, 4, 3, 50]
    # individual = [0, 50, 0, 10, 6, 1, 10, 90, 0, 6, 0, 6, 0, 50]
    individual = [0, 50, 0, 9, 7, 1, 7, 11, 0, 6, 1, 3, 0, 50]
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
    print(hof[0], test(hof[0], inner_toolbox))
