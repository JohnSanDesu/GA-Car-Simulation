import numpy as np
import matplotlib.pyplot as plt
import os

from braitenberg import Braitenberg

def plot_trajectories(trajectories, light_source, title):
    """
    Plots the trajectories of agents on a 2D plane.
    Each trajectory is labeled, and the light source is marked.
    """
    plt.figure()
    for trajectory, label in trajectories:
        plt.plot(trajectory[0], trajectory[1], label=label)
        # Mark start (green) and end (red) points of the trajectory
        plt.scatter(trajectory[0][0], trajectory[1][0], color='green', zorder=5)
        plt.scatter(trajectory[0][-1], trajectory[1][-1], color='red', zorder=5)
    plt.scatter(light_source[0], light_source[1], color='yellow', s=100, label='Light Source', edgecolors='black')
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='upper left', framealpha=0.7)
    plt.grid(True)
    plt.show()

def plot_fitness_trends(fitness_trends, title):
    """
    Plots the fitness trends over generations for multiple runs.
    Uses moving average smoothing for clarity.
    """
    num_plots = len(fitness_trends)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    for ax, (fitness, label) in zip(axes, fitness_trends):
        window_size = 5
        moving_avg = np.convolve(fitness, np.ones(window_size) / window_size, mode='valid')
        ax.plot(moving_avg, label=label)
        ax.set_ylim([0.0, 1.0])
        ax.set_title(label)
        ax.set_ylabel('Fitness')
        ax.grid(True)
    plt.xlabel('Generation')
    plt.suptitle(title)
    plt.show()

def save_results(filename, genotype, fitness):
    """
    Saves the best genotype and fitness score to a text file.
    """
    with open(filename, 'w') as file:
        file.write(f'Best Genotype: {genotype}\n')
        file.write(f'Best Fitness: {fitness}\n')

def run_ga_algorithm(ga_algorithm, utils, encoder, starting_position, starting_bearing, env, runtime, noise_stddev, encoding_method):
    """
    Runs the GA algorithm, encodes the best genotype and returns the trajectory,
    best fitness, fitness trend, and the best genotype.
    """
    best_genotype, best_fitness, fitness_trend = ga_algorithm.run(starting_position, starting_bearing, env, runtime=runtime, track_fitness=True)
    encoded_genotype = encoder.encode(best_genotype)
    # Create an agent with the encoded genotype
    agent = Braitenberg(starting_position, starting_bearing, encoded_genotype)
    trajectory, _ = env.run(runtime, agent, show=False)
    return trajectory, best_fitness, fitness_trend, best_genotype
