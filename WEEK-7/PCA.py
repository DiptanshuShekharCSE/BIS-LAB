import cv2
import numpy as np
import random

def otsu_fitness(threshold, img):
    t = int(threshold)
    ret, th = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    total = img.size
    P = hist / total
    w0 = np.sum(P[:t])
    w1 = np.sum(P[t:])
    if w0 == 0 or w1 == 0:
        return 0
    m0 = np.sum(np.arange(0, t) * P[:t]) / w0
    m1 = np.sum(np.arange(t, 256) * P[t:]) / w1
    sigma = w0 * w1 * (m0 - m1) ** 2
    return sigma

def GWO(img, max_iter=30, pack_size=20):
    wolves = np.random.randint(0, 256, pack_size)
    alpha = beta = delta = 0
    alpha_score = beta_score = delta_score = -np.inf

    for t in range(max_iter):
        for i in range(pack_size):
            fitness = otsu_fitness(wolves[i], img)
            if fitness > alpha_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = alpha_score, alpha
                alpha_score, alpha = fitness, wolves[i]
            elif fitness > beta_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = fitness, wolves[i]
            elif fitness > delta_score:
                delta_score, delta = fitness, wolves[i]

        a = 2 - t * (2 / max_iter)

        for i in range(pack_size):
            r1, r2 = random.random(), random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta

            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta

            wolves[i] = int((X1 + X2 + X3) / 3)
            wolves[i] = np.clip(wolves[i], 0, 255)

    return alpha

img = cv2.imread("image.jpg", 0)
best_threshold = GWO(img)
ret, output = cv2.threshold(img, best_threshold, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", img)
cv2.imshow("GWO Thresholded", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
