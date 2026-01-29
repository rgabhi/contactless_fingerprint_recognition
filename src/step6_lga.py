import argparse
import numpy as np
import random
import sys

class LooseGeneticAlgorithm:
    def __init__(self, W, Nt, Nq, pop_size=100, generations=100, mutation_rate=0.1):
        self.W = W
        self.Nt = Nt
        self.Nq = Nq
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Pre-compute diagonal indices for Greedy Seeding
        # Diagonal elements of W represent Minutia-Wise Similarity S_aa
        self.S_aa = np.zeros((Nt, Nq))
        for i in range(Nt):
            for j in range(Nq):
                idx = i * Nq + j
                self.S_aa[i, j] = W[idx, idx]

    def get_fitness(self, p):
        """
        Computes Energy Function f(x) = x^T * W * x
        p: Vector of size Nt. p[i] = j+1 (if match to query j), or 0 (no match).
        """
        # Convert p to list of active match indices in W
        # Match (i, j) corresponds to global index k = i * Nq + j
        # Note: p[i] is 1-based index (0 is null), so query index is p[i]-1
        
        active_indices = []
        for t_idx, val in enumerate(p):
            if val > 0:
                q_idx = val - 1
                global_idx = t_idx * self.Nq + q_idx
                active_indices.append(global_idx)
        
        if not active_indices:
            return 0.0

        # Efficient calculation: sum of submatrix of W corresponding to active indices
        # Score = Sum(W[u, v]) for all u, v in active_indices
        score = 0.0
        # Numpy advanced indexing to extract submatrix
        sub_W = self.W[np.ix_(active_indices, active_indices)]
        score = np.sum(sub_W)
        
        return score

    def generate_random_individual(self):
        """Randomly assigns matches respecting one-to-one constraint."""
        p = np.zeros(self.Nt, dtype=int)
        
        # Determine random number of matches
        num_matches = random.randint(1, min(self.Nt, self.Nq))
        
        # Randomly select query indices to use
        q_indices = random.sample(range(1, self.Nq + 1), num_matches)
        
        # Randomly select template indices to fill
        t_indices = random.sample(range(self.Nt), num_matches)
        
        for t, q in zip(t_indices, q_indices):
            p[t] = q
            
        return p

    def generate_greedy_individual(self):
        """Seeding: Creates an individual based on best diagonal scores S_aa."""
        p = np.zeros(self.Nt, dtype=int)
        used_q = set()
        
        # Sort all possible pairs (i, j) by their score S_aa[i, j] descending
        # Flatten S_aa to get list of (score, t_idx, q_idx)
        candidates = []
        for i in range(self.Nt):
            for j in range(self.Nq):
                candidates.append((self.S_aa[i, j], i, j))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, t, q in candidates:
            # Threshold: Only consider positive similarity
            if score <= 0: break
            
            # If t not matched and q not used
            if p[t] == 0 and (q + 1) not in used_q:
                p[t] = q + 1
                used_q.add(q + 1)
                
        return p

    def crossover(self, p1, p2):
        """
        Set-Based Crossover (Algorithm 1 in PDF)
        Handles constraints naturally by operating on sets of matches.
        """
        # Step 1: Define Sets A and B (Non-zero matches)
        # Set format: (template_idx, query_val)
        A = set((i, p1[i]) for i in range(self.Nt) if p1[i] != 0)
        B = set((i, p2[i]) for i in range(self.Nt) if p2[i] != 0)
        
        # Step 2: Intersection and Difference
        C = A.intersection(B)      # Common matches
        Da = A.difference(B)       # Unique to parent 1
        Db = B.difference(A)       # Unique to parent 2
        
        # Step 3: Selection
        # Randomly select subset from C
        if C:
            C_n1 = set(random.sample(list(C), k=random.randint(0, len(C))))
        else:
            C_n1 = set()
            
        # Randomly select subset from Da
        if Da:
            Da_prime = set(random.sample(list(Da), k=random.randint(0, len(Da))))
        else:
            Da_prime = set()

        # Randomly select subset from Db
        if Db:
            Db_prime = set(random.sample(list(Db), k=random.randint(0, len(Db))))
        else:
            Db_prime = set()
            
        # Step 4: Construction (Create 2 children)
        # Child 1: C_n1 U Da_prime
        # Child 2: C_n1 U Db_prime
        
        # NOTE: We need to ensure valid one-to-one mapping in children.
        # Since Da and Db come from valid parents, internal conflicts in Da/Db are impossible.
        # However, mixing them might cause conflict if not careful.
        # The algorithm implies strict separation, but let's be safe.
        
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
        """
        Mutation Operator: Remove, Add, or Swap.
        """
        p_new = p.copy()
        op = random.choice(['remove', 'add', 'swap'])
        
        active_indices = [i for i in range(self.Nt) if p_new[i] != 0]
        empty_indices = [i for i in range(self.Nt) if p_new[i] == 0]
        
        # Get set of used query values
        used_q = set(p_new[i] for i in active_indices)
        available_q = list(set(range(1, self.Nq + 1)) - used_q)
        
        if op == 'remove' and active_indices:
            # Randomly remove a match
            idx = random.choice(active_indices)
            p_new[idx] = 0
            
        elif op == 'add' and empty_indices and available_q:
            # Randomly add a match
            idx = random.choice(empty_indices)
            val = random.choice(available_q)
            p_new[idx] = val
            
        elif op == 'swap' and len(active_indices) >= 2:
            # Swap values of two positions
            idx1, idx2 = random.sample(active_indices, 2)
            p_new[idx1], p_new[idx2] = p_new[idx2], p_new[idx1]
            
        return p_new

    def run(self):
        # 1. Initialization
        population = [self.generate_random_individual() for _ in range(self.pop_size - 1)]
        # Add Seeded Individual
        population.append(self.generate_greedy_individual())
        
        best_fitness = -float('inf')
        best_individual = None
        
        print(f"Starting LGA Evolution over {self.generations} generations...")
        
        for gen in range(self.generations):
            # Calculate Fitness
            fitness_scores = [(ind, self.get_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Track Global Best
            current_best_ind, current_best_score = fitness_scores[0]
            if current_best_score > best_fitness:
                best_fitness = current_best_score
                best_individual = current_best_ind.copy()
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Fitness = {best_fitness:.4f}")
            
            # Selection (Elitism + Top 50%)
            # Keep top 50% as parents
            num_parents = self.pop_size // 2
            parents = [ind for ind, score in fitness_scores[:num_parents]]
            
            # Next Generation
            next_gen = []
            
            # Elitism: Carry over top 2
            next_gen.extend([parents[0].copy(), parents[1].copy()])
            
            while len(next_gen) < self.pop_size:
                # Select 2 random parents
                p1, p2 = random.sample(parents, 2)
                
                # Crossover
                c1, c2 = self.crossover(p1, p2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    c1 = self.mutate(c1)
                if random.random() < self.mutation_rate:
                    c2 = self.mutate(c2)
                    
                next_gen.extend([c1, c2])
                
            population = next_gen[:self.pop_size]
            
        print(f"LGA Finished. Final Best Fitness: {best_fitness:.4f}")
        return best_individual

def main(args):
    # Load Matrix W
    print(f"Loading Matrix W from {args.matrix_path}...")
    try:
        W = np.load(args.matrix_path)
    except FileNotFoundError:
        print("Error: Matrix file not found.")
        sys.exit(1)
        
    Nt = args.Nt
    Nq = args.Nq
    
    # Validation
    expected_dim = Nt * Nq
    if W.shape[0] != expected_dim:
        print(f"Error: Matrix dimension {W.shape} does not match Nt*Nq ({expected_dim}).")
        print("Check if you provided correct Nt/Nq matching the JSON files used to build W.")
        sys.exit(1)
        
    # Run GA
    ga = LooseGeneticAlgorithm(W, Nt, Nq, 
                               pop_size=args.pop_size, 
                               generations=args.generations)
    best_solution = ga.run()
    
    # Save Solution
    np.save(args.output, best_solution)
    print(f"Best solution vector saved to {args.output}.npy")
    
    # Verify Matches
    matches = []
    for t_idx, val in enumerate(best_solution):
        if val > 0:
            matches.append((t_idx, val - 1)) # Convert back to 0-based query index
            
    print(f"Found {len(matches)} matches: {matches}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part 3: Loose Genetic Algorithm')
    parser.add_argument('--matrix_path', required=True, help='Path to W matrix (.npy)')
    parser.add_argument('--Nt', type=int, required=True, help='Number of Minutiae in Template')
    parser.add_argument('--Nq', type=int, required=True, help='Number of Minutiae in Query')
    parser.add_argument('--output', required=True, help='Output file for solution vector')
    parser.add_argument('--pop_size', type=int, default=50, help='Population size')
    parser.add_argument('--generations', type=int, default=100, help='Generations')
    
    args = parser.parse_args()
    main(args)