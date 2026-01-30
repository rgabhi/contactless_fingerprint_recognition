import argparse
import numpy as np
import random
import sys
import traceback

class LooseGeneticAlgorithm:
    def __init__(self, W, Nt, Nq, pop_size=100, generations=200, mutation_rate=0.1):
        self.W = W
        self.Nt = Nt
        self.Nq = Nq
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Pre-compute diagonal indices for Greedy Seeding
        self.S_aa = np.zeros((Nt, Nq))
        for i in range(Nt):
            for j in range(Nq):
                idx = i * Nq + j
                self.S_aa[i, j] = W[idx, idx]

    def get_fitness(self, p):
        active_indices = []
        for t_idx, val in enumerate(p):
            if val > 0:
                q_idx = val - 1
                global_idx = t_idx * self.Nq + q_idx
                active_indices.append(global_idx)
        
        if not active_indices:
            return 0.0

        sub_W = self.W[np.ix_(active_indices, active_indices)]
        score = np.sum(sub_W)
        return score

    def generate_random_individual(self):
        p = np.zeros(self.Nt, dtype=int)
        max_matches = min(self.Nt, self.Nq)
        if max_matches < 1: return p

        num_matches = random.randint(1, max_matches)
        q_indices = random.sample(range(1, self.Nq + 1), num_matches)
        t_indices = random.sample(range(self.Nt), num_matches)
        
        for t, q in zip(t_indices, q_indices):
            p[t] = q
        return p

    def generate_greedy_individual(self):
        p = np.zeros(self.Nt, dtype=int)
        used_q = set()
        candidates = []
        for i in range(self.Nt):
            for j in range(self.Nq):
                candidates.append((self.S_aa[i, j], i, j))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, t, q in candidates:
            if score <= 0: break
            if p[t] == 0 and (q + 1) not in used_q:
                p[t] = q + 1
                used_q.add(q + 1)
        return p

    def crossover(self, p1, p2):
        A = set((i, p1[i]) for i in range(self.Nt) if p1[i] != 0)
        B = set((i, p2[i]) for i in range(self.Nt) if p2[i] != 0)
        
        C = A.intersection(B)
        Da = A.difference(B)
        Db = B.difference(A)
        
        C_n1 = set(random.sample(list(C), k=random.randint(0, len(C)))) if C else set()
        Da_prime = set(random.sample(list(Da), k=random.randint(0, len(Da)))) if Da else set()
        Db_prime = set(random.sample(list(Db), k=random.randint(0, len(Db)))) if Db else set()
        
        def set_to_vector(match_set):
            child = np.zeros(self.Nt, dtype=int)
            used_q = set()
            for t, q in match_set:
                if child[t] == 0 and q not in used_q:
                    child[t] = q
                    used_q.add(q)
            return child
            
        child1 = set_to_vector(C_n1.union(Da_prime))
        child2 = set_to_vector(C_n1.union(Db_prime))
        return child1, child2

    def mutate(self, p):
        p_new = p.copy()
        
        # Apply 1 to 3 mutations randomly to shake things up more
        num_mutations = random.randint(1, 3) 
        
        for _ in range(num_mutations):
            op = random.choice(['remove', 'add', 'swap'])
            
            active_indices = [i for i in range(self.Nt) if p_new[i] != 0]
            empty_indices = [i for i in range(self.Nt) if p_new[i] == 0]
            
            if op == 'remove' and active_indices:
                idx = random.choice(active_indices)
                p_new[idx] = 0
                
            elif op == 'add' and empty_indices: # Note: Simplified checks
                used_q = set(p_new[i] for i in active_indices)
                available_q = list(set(range(1, self.Nq + 1)) - used_q)
                if available_q:
                    idx = random.choice(empty_indices)
                    val = random.choice(available_q)
                    p_new[idx] = val
                
            elif op == 'swap' and len(active_indices) >= 2:
                idx1, idx2 = random.sample(active_indices, 2)
                p_new[idx1], p_new[idx2] = p_new[idx2], p_new[idx1]
                
        return p_new

    def run(self):
        # 1. Initialize Population
        population = [self.generate_random_individual() for _ in range(self.pop_size - 1)]
        population.append(self.generate_greedy_individual()) # Keep the greedy seed!
        
        best_fitness = -float('inf')
        best_individual = population[0].copy()
        
        # Stagnation tracking variables
        stagnation_counter = 0
        current_mutation_rate = self.mutation_rate
        
        for gen in range(self.generations):
            # Evaluate Fitness
            fitness_scores = [(ind, self.get_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Check for improvement
            current_best_ind, current_best_score = fitness_scores[0]
            
            if current_best_score > best_fitness:
                best_fitness = current_best_score
                best_individual = current_best_ind.copy()
                stagnation_counter = 0  # Reset on improvement
                current_mutation_rate = self.mutation_rate # Reset rate
            else:
                stagnation_counter += 1

            # --- ADAPTIVE TUNING LOGIC ---
            # If stuck for 15 generations, triple the mutation rate
            if stagnation_counter > 15:
                current_mutation_rate = min(0.4, self.mutation_rate * 3)
            # If stuck for 40 generations, introduce "Cataclysm" (Extreme mutation)
            if stagnation_counter > 40:
                current_mutation_rate = 0.6 
            # -----------------------------

            # Selection (Elitism: Keep top 2 parents)
            num_parents = self.pop_size // 2
            parents = [ind for ind, score in fitness_scores[:num_parents]]
            next_gen = [parents[0].copy(), parents[1].copy()]
            
            # Reproduction
            while len(next_gen) < self.pop_size:
                # Random selection from top 50%
                p1, p2 = random.sample(parents, 2)
                c1, c2 = self.crossover(p1, p2)
                
                # Apply mutation with the DYNAMIC rate
                if random.random() < current_mutation_rate: 
                    c1 = self.mutate(c1)
                if random.random() < current_mutation_rate: 
                    c2 = self.mutate(c2)
                    
                next_gen.extend([c1, c2])
                
            population = next_gen[:self.pop_size]
            
        return best_individual

def main(args):
    try:
        # Robust loading: Ensure float32
        W = np.load(args.matrix_path).astype(np.float32)
            
        Nt = args.Nt
        Nq = args.Nq
        expected_dim = Nt * Nq
        
        if W.shape[0] != expected_dim:
            print(f"Error: Matrix dim {W.shape[0]} != expected {expected_dim}", file=sys.stderr)
            sys.exit(1)
            
        ga = LooseGeneticAlgorithm(W, Nt, Nq, 
                                   pop_size=args.pop_size, 
                                   generations=args.generations)
        best_solution = ga.run()
        
        np.save(args.output, best_solution)
        
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix_path', required=True)
    parser.add_argument('--Nt', type=int, required=True)
    parser.add_argument('--Nq', type=int, required=True)
    parser.add_argument('--output', required=True)
    
    # --- TUNING EDITS ---
    # Increased Population: 100 -> 200 (Better diversity)
    parser.add_argument('--pop_size', type=int, default=200) 
    
    # Increased Generations: 100 -> 400 (Allow time for adaptive logic to work)
    parser.add_argument('--generations', type=int, default=400) 
    # --------------------
    
    args = parser.parse_args()
    main(args)