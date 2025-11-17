import cv2
import numpy as np
import random

def fitness(img_noisy, img_original, sigma):
    if sigma <= 0:
        return -1e9
    blurred = cv2.GaussianBlur(img_noisy, (0, 0), sigma)
    mse = np.mean((img_original - blurred) ** 2)
    if mse == 0:
        return 1e9
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr

def pso_optimize(img_original, img_noisy, n_particles=15, iterations=30):
    particles = np.random.uniform(0.1, 5, n_particles)
    velocities = np.random.uniform(-1, 1, n_particles)
    pbest = particles.copy()
    pbest_values = np.array([fitness(img_noisy, img_original, s) for s in particles])
    gbest = pbest[np.argmax(pbest_values)]
    w = 0.7
    c1 = 1.4
    c2 = 1.4

    for _ in range(iterations):
        for i in range(n_particles):
            r1 = random.random()
            r2 = random.random()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )
            particles[i] += velocities[i]
            if particles[i] < 0.1:
                particles[i] = 0.1
            value = fitness(img_noisy, img_original, particles[i])
            if value > pbest_values[i]:
                pbest_values[i] = value
                pbest[i] = particles[i]
        gbest = pbest[np.argmax(pbest_values)]

    return gbest, max(pbest_values)

img_original = cv2.imread("original.png", 0)
img_noisy = cv2.imread("noisy.png", 0)

best_sigma, best_psnr = pso_optimize(img_original, img_noisy)
denoised = cv2.GaussianBlur(img_noisy, (0, 0), best_sigma)

cv2.imwrite("denoised.png", denoised)

print("Best Sigma:", best_sigma)
print("Best PSNR:", best_psnr)
