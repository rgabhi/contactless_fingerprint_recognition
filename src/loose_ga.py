import numpy as np
import random

class LooseGeneticAlgorithm:
    def __init__(self, W, assignments, N1, N2, pop_size=200, crossover_rate=0.5, mutation_rate=0.5):
        self.W = W
        self.assignments = assignments  # List of (i, j) tuples corresponding to W indices
        self.N1 = N1
        self.N2 = N2
        self.pop_size = pop_size
        self.cr = crossover_rate
        self.mr = mutation_rate
        
        # Helper: Map (i, j) pair to its index u in W for fast fitness calc
        self.pair_to_idx = {pair: u for u, pair in enumerate(assignments)}

    def vector_to_binary(self, p):
        """
        Convert integer vector p (length N1) to binary vector x (length M).
        p[i] = j means T_i matches Q_{j-1}. p[i] = 0 means unmatched.
        """
        x = np.zeros(len(self.assignments), dtype=np.int8)
        for i, val in enumerate(p):
            if val > 0:
                j = val - 1  # Convert 1-based value to 0-based index
                if (i, j) in self.pair_to_idx:
                    u = self.pair_to_idx[(i, j)]
                    x[u] = 1
        return x

    def calculate_fitness(self, p):
        """Compute x^T W x for an individual p."""
        x = self.vector_to_binary(p)
        idx = np.where(x == 1)[0]
        if len(idx) == 0:
            return 0.0
        # Sum W[u, v] for all active u, v
        return float(np.sum(self.W[np.ix_(idx, idx)]))

    def initialize_population(self, initial_match=None):
        
        population = []
        
        # 1. seeding 
        if initial_match is not None:
            population.append(initial_match)
        
        # 2. random Individuals
        while len(population) < self.pop_size:
            p = np.zeros(self.N1, dtype=np.int32)
            
            # randomly decide how many matches to make (n0)
            n0 = random.randint(1, min(self.N1, self.N2))
            
            # select n0 random indices from T and Q
            indices_T = random.sample(range(self.N1), n0)
            indices_Q = random.sample(range(self.N2), n0)
            
            # assign matches
            for i, j in zip(indices_T, indices_Q):
                p[i] = j + 1  # store as 1-based index
                
            population.append(p)
            
        return np.array(population)

    def crossover(self, p1, p2):
        """
        Set-Based Crossover (Algorithm 1 in Paper / Assignment).
        """
        A = set((i, p1[i]) for i in range(self.N1) if p1[i] != 0)
        B = set((i, p2[i]) for i in range(self.N1) if p2[i] != 0)

        # Intersection and Diff
        C = A.intersection(B)      # Matches in both
        D_A = A.difference(B)      # Matches unique to A
        D_B = B.difference(A)      # Matches unique to B

        # Random Selection
        n1 = random.randint(0, len(C))
        C_n1 = set(random.sample(list(C), n1))
        
        # random subset of D_A
        n2a = random.randint(0, len(D_A))
        D_A_prime = set(random.sample(list(D_A), n2a))
        
        # random subset of D_B
        n2b = random.randint(0, len(D_B))
        D_B_prime = set(random.sample(list(D_B), n2b))

        # construction
        S1 = C_n1.union(D_A_prime)
        S2 = C_n1.union(D_B_prime)

        c1 = np.zeros(self.N1, dtype=np.int32)
        for i, val in S1:
            c1[i] = val
            
        c2 = np.zeros(self.N1, dtype=np.int32)
        for i, val in S2:
            c2[i] = val

        return c1, c2

    def mutation(self, p):
        """
        Mutation Operator (Remove, Add, Swap).
        Returns a NEW mutated individual.
        """
        pm = p.copy()
        
        # 3 types of mutation
        mutation_type = random.choice(['remove', 'add', 'swap'])
        
        if mutation_type == 'remove':
            non_zeros = np.where(pm != 0)[0]
            if len(non_zeros) > 0:
                idx = random.choice(non_zeros)
                pm[idx] = 0
                
        elif mutation_type == 'add':
            zeros = np.where(pm == 0)[0]
            if len(zeros) > 0:
                # find unused values in
                used_vals = set(pm)
                all_vals = set(range(1, self.N2 + 1))
                unused_vals = list(all_vals - used_vals)
                
                if unused_vals:
                    idx = random.choice(zeros)
                    val = random.choice(unused_vals)
                    pm[idx] = val
                    
        elif mutation_type == 'swap':
            # swap two positions (j1, j2)
            j1, j2 = random.sample(range(self.N1), 2)
            pm[j1], pm[j2] = pm[j2], pm[j1]
            
        return pm

    def run(self, max_generations=50, initial_solution=None):
        """Main GA Loop."""
        population = self.initialize_population(initial_solution)
        
        for g in range(max_generations):
            # calculate Fitness
            fitness_scores = [self.calculate_fitness(p) for p in population]
            
            # sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = population[sorted_indices]
            
            next_gen = []
            
            # 2. Crossover
            # Select parents (simple top-half selection or tournament)
            # take random pairs from the current population
            num_crossovers = int(self.pop_size * self.cr)
            for _ in range(num_crossovers):
                p1 = population[random.randint(0, len(population)-1)]
                p2 = population[random.randint(0, len(population)-1)]
                c1, c2 = self.crossover(p1, p2)
                next_gen.extend([c1, c2])
                
            # 3. Mutation
            num_mutations = int(self.pop_size * self.mr)
            for _ in range(num_mutations):
                p = population[random.randint(0, len(population)-1)]
                pm = self.mutation(p)
                next_gen.append(pm)
                
            # Selection (Survivor Selection)
            # combine parents + children + mutants
            combined_pop = np.vstack((population, np.array(next_gen)))
            
            # recalculate fitness for everyone
            final_scores = [self.calculate_fitness(p) for p in combined_pop]
            best_indices = np.argsort(final_scores)[::-1][:self.pop_size]
            
            population = combined_pop[best_indices]
            
            # print(f"Gen {g}: Best Fitness = {final_scores[best_indices[0]]}")
            
        return population[0], self.calculate_fitness(population[0])