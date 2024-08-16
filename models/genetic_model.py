import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import CNNModel
from utils.data import load_data
from sklearn.metrics import accuracy_score

# Define Genetic Algorithm for CNN Hyperparameter Optimization
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, train_loader, test_loader):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                'lr': 10 ** np.random.uniform(-4, -2),  # Learning rate between 0.0001 and 0.01
                'conv1_out_channels': np.random.randint(16, 64),
                'conv2_out_channels': np.random.randint(32, 128),
                'conv3_out_channels': np.random.randint(64, 256),
                'fc1_units': np.random.randint(128, 512),
                'fc2_units': np.random.randint(64, 256),
            }
            population.append(individual)
        return population

    def fitness(self, individual):
        model = CNNModel(
            conv1_out_channels=individual['conv1_out_channels'],
            conv2_out_channels=individual['conv2_out_channels'],
            conv3_out_channels=individual['conv3_out_channels'],
            fc1_units=individual['fc1_units'],
            fc2_units=individual['fc2_units'],
        )
        optimizer = optim.Adam(model.parameters(), lr=individual['lr'])
        criterion = nn.CrossEntropyLoss()

        # Train the model
        model.train()
        for images, labels in self.train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = model(images)
                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        _, predicted = torch.max(all_outputs, 1)
        accuracy = accuracy_score(all_labels.numpy(), predicted.numpy())
        return accuracy

    def selection(self):
        # Roulette Wheel Selection
        fitness_scores = np.array([self.fitness(ind) for ind in self.population])
        probabilities = fitness_scores / fitness_scores.sum()
        selected_indices = np.random.choice(len(self.population), size=self.population_size, p=probabilities)
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            if np.random.rand() > 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def mutation(self, individual):
        for key in individual:
            if np.random.rand() < self.mutation_rate:
                if key == 'lr':
                    individual[key] = 10 ** np.random.uniform(-4, -2)
                else:
                    individual[key] = np.random.randint(16, 256) if 'conv' in key else np.random.randint(64, 512)
        return individual

    def evolve(self):
        for generation in range(self.generations):
            print(f'Generation {generation + 1}/{self.generations}')
            selected_population = self.selection()
            new_population = []

            for i in range(0, self.population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % self.population_size]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))

            self.population = new_population

        # Get the best individual from the final generation
        best_individual = max(self.population, key=lambda ind: self.fitness(ind))
        return best_individual

# Example usage:
if __name__ == "__main__":
    train_loader, test_loader = load_data('C:/Kuliah/Penelitian Padi_1/dataset2/train_images', 'C:/Kuliah/Penelitian Padi_1/dataset2/test_images')
    ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1, generations=5, train_loader=train_loader, test_loader=test_loader)
    best_hyperparameters = ga.evolve()
    print("Best Hyperparameters:", best_hyperparameters)

    # Build and save the best CNN model with these hyperparameters
    best_cnn_model = CNNModel(
        conv1_out_channels=best_hyperparameters['conv1_out_channels'],
        conv2_out_channels=best_hyperparameters['conv2_out_channels'],
        conv3_out_channels=best_hyperparameters['conv3_out_channels'],
        fc1_units=best_hyperparameters['fc1_units'],
        fc2_units=best_hyperparameters['fc2_units'],
    )
    torch.save(best_cnn_model.state_dict(), 'best_cnn_model.pth')
