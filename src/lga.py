import numpy as np
import random

class LooseGA:
    def __init__(self, W, Nt, Nq, pop_size=50, max_generations=100):
        self.W = W
        self.Nt = Nt
        self.Nq = Nq
        self.pop_size = pop_size
        self.max_generations = max_generations
        
    def fitness(self, solution_vector):
        """
        Eq. 1: f(x) = x.T * W * x
        Solution vector p is length Nt, values in [0, Nq]. 0 means no match.
        We must convert p to binary vector x of length Nt*Nq.
        """
        x = np.zeros(self.Nt * self.Nq)
        for i in range(self.Nt):
            match_idx = solution_vector[i]
            if match_idx > 0: # 1-based index in solution_vector, 0 is null
                # Convert to 0-based index for query
                j = match_idx - 1 
                flat_idx = i * self.Nq + j
                x[flat_idx] = 1
        
        # Calculate Energy
        return x.T @ self.W @ x

    def initialize_population(self, best_greedy_match):
        """
        Initialization[cite: 91]. 
        Includes random individuals and seeded greedy solution.
        """
        population = []
        
        # Seed the best individual (greedy solution passed from Part 2)
        population.append(np.array(best_greedy_match))
        
        # Generate random individuals
        for _ in range(self.pop_size - 1):
            # Randomly select matches. 
            # Note: Need to respect one-to-one roughly, but GA handles convergence.
            ind = np.zeros(self.Nt, dtype=int)
            used_q = set()
            for i in range(self.Nt):
                if random.random() > 0.5: # 50% chance to match
                    choice = random.randint(1, self.Nq)
                    if choice not in used_q:
                        ind[i] = choice
                        used_q.add(choice)
            population.append(ind)
            
        return population

    def crossover(self, p1, p2):
        """
        Set-Based Crossover [cite: 95-104]
        """
        # Step 1: Define Sets A and B (non-zero matches)
        A = {(i, p1[i]) for i in range(self.Nt) if p1[i] != 0}
        B = {(i, p2[i]) for i in range(self.Nt) if p2[i] != 0}
        
        # Step 2: Intersection and Difference
        C = A.intersection(B)
        D_A = A - B
        D_B = B - A
        
        # Step 3: Selection (Random subsets)
        C_n1 = set(random.sample(list(C), k=random.randint(0, len(C))))
        D_A_prime = set(random.sample(list(D_A), k=random.randint(0, len(D_A))))
        D_B_prime = set(random.sample(list(D_B), k=random.randint(0, len(D_B))))
        
        # Step 4: Construction
        S1 = C_n1.union(D_A_prime)
        S2 = C_n1.union(D_B_prime)
        
        # Convert back to vector
        c1 = np.zeros(self.Nt, dtype=int)
        for (i, val) in S1: c1[i] = val
            
        c2 = np.zeros(self.Nt, dtype=int)
        for (i, val) in S2: c2[i] = val
            
        return c1, c2

    def mutation(self, p):
        """
        Mutation Operators: Remove, Add, Swap [cite: 105-113]
        """
        pm = p.copy()
        op = random.choice(['remove', 'add', 'swap'])
        
        non_zeros = [i for i in range(self.Nt) if pm[i] != 0]
        zeros = [i for i in range(self.Nt) if pm[i] == 0]
        
        if op == 'remove' and non_zeros:
            idx = random.choice(non_zeros)
            pm[idx] = 0
            
        elif op == 'add' and zeros:
            idx = random.choice(zeros)
            # Find unused query index
            used_vals = set(pm)
            available = [q for q in range(1, self.Nq + 1) if q not in used_vals]
            if available:
                pm[idx] = random.choice(available)
                
        elif op == 'swap' and len(non_zeros) >= 2:
            idx1, idx2 = random.sample(non_zeros, 2)
            pm[idx1], pm[idx2] = pm[idx2], pm[idx1]
            
        return pm

    def run(self, initial_seed):
        population = self.initialize_population(initial_seed)
        
        for g in range(self.max_generations):
            next_gen = []
            
            # Elitism: Keep best
            scores = [(ind, self.fitness(ind)) for ind in population]
            scores.sort(key=lambda x: x[1], reverse=True)
            next_gen.append(scores[0][0])
            
            # Generate offspring
            while len(next_gen) < self.pop_size:
                # Tournament Selection
                parents = random.sample(population, 2)
                p1, p2 = parents[0], parents[1]
                
                # Crossover
                c1, c2 = self.crossover(p1, p2)
                
                # Mutation
                if random.random() < 0.2: c1 = self.mutation(c1)
                if random.random() < 0.2: c2 = self.mutation(c2)
                
                next_gen.extend([c1, c2])
            
            population = next_gen[:self.pop_size]
            
        # Return best solution
        best_sol = max(population, key=self.fitness)
        return best_sol