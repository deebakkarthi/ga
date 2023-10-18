package main

import (
	"fmt"
	"math/rand"
	"sort"
)

type Gene = float32

type DNA struct {
	genes []Gene
}

const mutRate = 0.01
const popSize = 150

var weights []float32 = []float32{10, 20, 30}
var benefits []float32 = []float32{60, 100, 120}

const maxWeight float32 = 50

var numItems int = len(weights)

func geneGen() Gene {
	return Gene(rand.Float32())
}

func populationGen(n int) []DNA {
	ret := make([]DNA, n)
	for i := 0; i < n; i++ {
		var tmp DNA
		for range weights {
			tmp.genes = append(tmp.genes, geneGen())
		}
		ret[i] = tmp
	}
	return ret
}

func fitness(d DNA) float32 {
	var tmp float32
	var totalWeight float32
	for i, v := range d.genes {
		tmp += v * benefits[i]
		totalWeight += v * weights[i]
	}
	// Return -1 if the solution is overweight
	if totalWeight > maxWeight {
		return -1
	}
	return tmp
}

func selection(population *[]DNA, fitness func(DNA) float32) []DNA {
	var matingPool []DNA
	sort.Slice(*population, func(i, j int) bool {
		return fitness((*population)[i]) < fitness((*population)[j])
	})
	for i, v := range *population {
		for j := 0; j < i; j++ {
			matingPool = append(matingPool, v)
		}
	}
	return matingPool
}

func crossover(a, b DNA) DNA {
	var child DNA
	child.genes = make([]float32, numItems)
	midpoint := rand.Intn(len(a.genes))
	for i := 0; i < len(a.genes); i++ {
		if i < midpoint {
			child.genes[i] = a.genes[i]
		} else {
			child.genes[i] = b.genes[i]
		}
	}
	return child
}

func mutate(d *DNA, mutRate float32) {
	for i := range d.genes {
		if rand.Float32() < mutRate {
			d.genes[i] = geneGen()
		}
	}
}

func (d DNA) equals(b DNA) bool {
	for i, v := range d.genes {
		if v != b.genes[i] {
			return false
		}
	}
	return true
}

func reproduction(population *[]DNA, crossover func(a, b DNA) DNA,
	mutate func(d *DNA, mutRate float32)) {
	matingPool := selection(population, fitness)
	for i := range *population {
		var a, b DNA
		for a.equals(b) {
			a = matingPool[rand.Intn(len(matingPool))]
			b = matingPool[rand.Intn(len(matingPool))]
		}
		child := crossover(a, b)
		mutate(&child, mutRate)
		(*population)[i] = child
	}

}

func main() {
	population := populationGen(popSize)
	generations := 0
	for i := 0; i < 300; i++ {
		reproduction(&population, crossover, mutate)
		fmt.Printf("%f %f\n", population[len(population)-1], fitness(population[len(population)-1]))
		generations++
	}
	fmt.Println(generations)
}
