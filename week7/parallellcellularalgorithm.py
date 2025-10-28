import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time

# -------------------------------
# Parallel Cellular Algorithm (PCA)
# -------------------------------
def parallel_cellular_denoise(noisy_img, iterations=50, alpha=0.4):
    """
    Parallel Cellular Algorithm for image denoising.
    Prints iteration-wise convergence values.
    """
    img = noisy_img.astype(float) / 255.0  # normalize 0-1
    denoised = img.copy()
    rows, cols = img.shape

    print("\n🔹 Running Parallel Cellular Algorithm (PCA)...")
    prev_img = denoised.copy()

    for it in range(iterations):
        new_img = denoised.copy()
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # 8-neighbor average
                neighborhood = denoised[i - 1:i + 2, j - 1:j + 2]
                avg_neighbor = np.mean(neighborhood)
                new_img[i, j] = (1 - alpha) * denoised[i, j] + alpha * avg_neighbor

        denoised = np.clip(new_img, 0, 1)

        # Calculate average pixel change to track convergence
        avg_change = np.mean(np.abs(denoised - prev_img))
        prev_img = denoised.copy()

        print(f"   → Iteration {it + 1}/{iterations} | Avg Pixel Change: {avg_change:.6f}")

    return (denoised * 255).astype(np.uint8)


# -------------------------------
# Genetic Algorithm (GA)
# -------------------------------
def genetic_denoise(noisy_img, population_size=10, generations=50, mutation_rate=0.1):
    """
    Genetic Algorithm for image denoising.
    Prints fitness values for each generation.
    """
    noisy = noisy_img.astype(float) / 255.0
    rows, cols = noisy.shape

    print("\n🔹 Running Genetic Algorithm (GA)...")

    # Initialize random population near the noisy image
    population = [np.clip(noisy + np.random.normal(0, 0.05, size=noisy.shape), 0, 1)
                  for _ in range(population_size)]

    def fitness(candidate):
        diff_term = np.mean((candidate - noisy) ** 2)
        gx, gy = np.gradient(candidate)
        smoothness = np.mean(np.abs(gx)) + np.mean(np.abs(gy))
        return diff_term + 0.5 * smoothness

    for g in range(generations):
        fitness_values = [fitness(p) for p in population]
        ranked = np.argsort(fitness_values)
        best_fitness = fitness_values[ranked[0]]
        best = population[ranked[0]]

        # Print best fitness value for tracking progress
        print(f"   → Generation {g + 1}/{generations} | Best Fitness: {best_fitness:.6f}")

        # Selection: top 2 parents
        parents = [population[ranked[0]], population[ranked[1]]]

        # Crossover & Mutation
        new_population = []
        for _ in range(population_size):
            p1, p2 = random.choices(parents, k=2)
            mask = np.random.rand(rows, cols) < 0.5
            child = np.where(mask, p1, p2)

            # Mutation
            mutation = np.random.rand(rows, cols) < mutation_rate
            child[mutation] += np.random.normal(0, 0.05, size=np.sum(mutation))
            child = np.clip(child, 0, 1)
            new_population.append(child)

        population = new_population

    return (best * 255).astype(np.uint8)


# -------------------------------
# Load and Prepare Image
# -------------------------------
img = cv2.imread("imagee.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image file not found. Please check the filename and path.")

# Add Gaussian noise
noisy = img + np.random.normal(0, 25, img.shape)
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

# -------------------------------
# Run Algorithms
# -------------------------------
start_pca = time.time()
pca_result = parallel_cellular_denoise(noisy, iterations=50)
end_pca = time.time()

start_ga = time.time()
ga_result = genetic_denoise(noisy, generations=50)
end_ga = time.time()

# -------------------------------
# Display Results
# -------------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy Image")

plt.subplot(1, 3, 2)
plt.imshow(pca_result, cmap='gray')
plt.title(f"PCA Denoised\nTime={end_pca - start_pca:.2f}s")

plt.subplot(1, 3, 3)
plt.imshow(ga_result, cmap='gray')
plt.title(f"GA Denoised\nTime={end_ga - start_ga:.2f}s")

plt.tight_layout()
plt.show()
