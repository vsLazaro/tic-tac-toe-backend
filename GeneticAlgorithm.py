import torch
import random
import numpy as np
from NeuralLink import NeuralLink
from Utils import check_winner, minimax_best_move

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate=0.1, convergence_threshold=0.98):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        self.population = [NeuralLink(9, 9, 9) for _ in range(population_size)]
        self.fitness_scores = []

    def evaluate_fitness(self):
        fitness = []
        for nn in self.population:
            score = 0
            for _ in range(5):  # Simula 5 partidas contra o Minimax
                board = [0] * 9  # Tabuleiro vazio
                result = self.simulate_game(nn, board)
                score += result
            fitness.append(score / 5)  # Fitness é a média das partidas
        return fitness

    def simulate_game(self, nn, board):
        player_turn = True
        while check_winner(board) is None:
            if player_turn:
                # Passa o tabuleiro para o método forward da rede neural
                output = nn.forward(torch.tensor(board, dtype=torch.float32))
                # Converte o tensor de saída para lista e obtém o índice do movimento
                output_list = output.tolist()
                move = output_list.index(1) if 1 in output_list else -1
                if move == -1 or board[move] != 0:  # Penalidade para jogada inválida
                    return -100
                board[move] = 1
            else:
                move = minimax_best_move(board)
                board[move] = -1
            player_turn = not player_turn

        winner = check_winner(board)
        return 100 if winner == 1 else 50 if winner == 0 else 0

    def select_parents(self):
        def tournament():
            candidates = random.sample(range(self.population_size), 2)
            if self.fitness_scores[candidates[0]] > self.fitness_scores[candidates[1]]:
                return self.population[candidates[0]]
            else:
                return self.population[candidates[1]]
        return tournament(), tournament()

    def crossover(self, parent1, parent2):
        child = NeuralLink(9, 9, 9)
        crossover_point = random.randint(1, 8)
        child.weights_hidden[:crossover_point, :] = parent1.weights_hidden[:crossover_point, :]
        child.weights_hidden[crossover_point:, :] = parent2.weights_hidden[crossover_point:, :]
        crossover_point = random.randint(1, 8)
        child.weights_output[:crossover_point, :] = parent1.weights_output[:crossover_point, :]
        child.weights_output[crossover_point:, :] = parent2.weights_output[crossover_point:, :]
        child.bias_hidden[:crossover_point] = parent1.bias_hidden[:crossover_point]
        child.bias_hidden[crossover_point:] = parent2.bias_hidden[crossover_point:]
        child.bias_output[:crossover_point] = parent1.bias_output[:crossover_point]
        child.bias_output[crossover_point:] = parent2.bias_output[crossover_point:]
        return child

    def mutate(self, individual):
        for i in range(9):
            if random.random() < self.mutation_rate:
                individual.weights_hidden[i] += torch.empty(9).uniform_(-0.1, 0.1)
            if random.random() < self.mutation_rate:
                individual.weights_output[i] += torch.empty(9).uniform_(-0.1, 0.1)
            if random.random() < self.mutation_rate:
                individual.bias_hidden[i] += random.uniform(-0.1, 0.1)
                individual.bias_output[i] += random.uniform(-0.1, 0.1)

    def run(self):
        for generation in range(self.generations):
            print(f"Generation {generation + 1}")
            self.fitness_scores = self.evaluate_fitness()
            avg_fitness = np.mean(self.fitness_scores)
            print(f"Average fitness: {avg_fitness:.2f}")
            if avg_fitness >= self.convergence_threshold:
                print(f"Convergence reached with average fitness: {avg_fitness:.2f}")
                break
            elite_index = np.argmax(self.fitness_scores)
            elite = self.population[elite_index]
            new_population = [elite]
            for _ in range(self.population_size - 1):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population

if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=20, generations=40, mutation_rate=0.1)
    ga.run()
