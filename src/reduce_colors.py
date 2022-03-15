import numpy as np
import random
import numpy.matlib


def homogenize_image(img):
    return np.reshape(img / 255.0, (img.shape[0] * img.shape[1], 3))


def grab_initial_centroids(X, K):
    return random.sample(list(X),K)


def find_nearest_centroids(X,K,c):
    arr = np.empty((np.size(X,0),1))
    for i in range(0,K):
        y = c[i]
        temp = np.ones((np.size(X,0),1))*y
        b = np.power(np.subtract(X,temp),2)
        a = np.sum(b,axis = 1)
        a = np.asarray(a)
        a.resize((np.size(X,0),1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr, 0, axis=1)
    return np.argmin(arr, axis=1)


def compute_centroids(X,idx,K):
    n = np.size(X,1)
    centroids = np.zeros((K,n))
    for i in range(0,K):
        ci = idx==i
        ci = ci.astype(int)
        total_number = sum(ci)
        ci.resize((np.size(X,0),1))
        total_matrix = np.matlib.repmat(ci,1,n)
        ci = np.transpose(ci)
        total = np.multiply(X,total_matrix)
        centroids[i] = ( (np.sum(total,axis=0) / total_number) if total_number != 0 else 0) 
    return centroids


def compute_knn(X, K, initial_centroids, threshold, max_iters):
    centroids = initial_centroids
    previous_centroids = centroids
    previous_delta = 0
    for i in range(1,max_iters):
        idx = find_nearest_centroids(X,K,centroids)
        centroids = compute_centroids(X,idx,K)
        delta = np.linalg.norm(centroids - previous_centroids)
        if i > 1 and np.linalg.norm(delta - previous_delta) < threshold:
            break
        previous_delta = delta
    return centroids, idx


def reduce_colors_knn(img, K=24, threshold=0.05, max_iters=25):
    X = homogenize_image(img)
    initial_centroids = grab_initial_centroids(X, K)
    centroids, idx = compute_knn(X, K, initial_centroids, threshold, max_iters)
    idx = find_nearest_centroids(X,K,centroids)
    X_recovered = centroids[idx]
    X_recovered = np.reshape(X_recovered, (img.shape[0], img.shape[1], 3)) * 255
    return X_recovered