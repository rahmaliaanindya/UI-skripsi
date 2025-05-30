# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from scipy.stats import zscore
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from collections import Counter
from joblib import Parallel, delayed
import numba
from scipy.sparse import csr_matrix
import warnings
import random
import os

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
warnings.filterwarnings("ignore")

# Memuat data
df = pd.read_excel('/content/dataset indikator kemiskinan.xlsx')

# =============================================
# OPTIMASI FUNGSI UTAMA DENGAN NUMBA JIT
# =============================================

@numba.jit(nopython=True, fastmath=True)
def rbf_kernel_fast(X, gamma):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = X[i] - X[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    return K

def evaluate_gamma_parallel(gamma_val, X_scaled, n_clusters=2, n_runs=3):
    sil_list, dbi_list = [], []
    
    for _ in range(n_runs):
        try:
            # Gunakan versi fast RBF kernel
            W = rbf_kernel_fast(X_scaled, gamma_val)
            W[W < 0.01] = 0  # Thresholding
            
            # Gunakan sparse matrix untuk efisiensi
            W_sparse = csr_matrix(W)
            L = laplacian(W_sparse, normed=True)
            
            # Gunakan sparse eigensolver dengan toleransi lebih longgar
            eigvals, eigvecs = eigsh(L, k=n_clusters, which='SM', tol=1e-4)
            U = normalize(eigvecs, norm='l2')
            
            # KMeans dengan n_init lebih kecil untuk efisiensi
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=SEED)
            labels = kmeans.fit_predict(U)
            
            if len(np.unique(labels)) < 2:
                raise ValueError("Single cluster")
                
            sil = silhouette_score(U, labels)
            dbi = davies_bouldin_score(U, labels)
            
            sil_list.append(sil)
            dbi_list.append(dbi)
            
        except Exception as e:
            sil_list.append(0.0)
            dbi_list.append(10.0)
    
    mean_sil = np.mean(sil_list)
    mean_dbi = np.mean(dbi_list)
    return -mean_sil + mean_dbi  # Fitness score (lower is better)

def parallel_evaluation(gamma_array, X_scaled):
    # Evaluasi partikel secara paralel
    scores = Parallel(n_jobs=-1)(
        delayed(evaluate_gamma_parallel)(gamma[0], X_scaled)
        for gamma in gamma_array
    )
    return np.array(scores)

# =============================================
# PREPROCESSING DATA (SAMA DENGAN SEBELUMNYA)
# =============================================

X = df.drop(columns=['Kabupaten/Kota'])
scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
X_scaled = scaler.fit_transform(X)

# =============================================
# EVALUASI JUMLAH CLUSTER OPTIMAL (TIDAK BERUBAH)
# =============================================

silhouette_scores = []
db_scores = []
k_range = range(2, 11)

for k in k_range:
    model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
    labels = model.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Jumlah cluster optimal: {optimal_k}")

# =============================================
# SPECTRAL CLUSTERING MANUAL SEBELUM PSO
# =============================================

gamma_initial = 0.1
W = rbf_kernel_fast(X_scaled, gamma_initial)
W[W < 0.01] = 0

D = np.diag(W.sum(axis=1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

eigvals, eigvecs = eigh(L_sym)
U = eigvecs[:, :optimal_k]
U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

kmeans = KMeans(n_clusters=optimal_k, random_state=SEED)
labels_before = kmeans.fit_predict(U_norm)

# =============================================
# OPTIMASI PSO DENGAN VERSI PARALEL
# =============================================

# Setup PSO
options = {
    'c1': 1.5,
    'c2': 1.7,  # Lebih eksploratif
    'w': 0.72,
    'k': 10,
    'p': 2
}

optimizer = GlobalBestPSO(
    n_particles=20,
    dimensions=1,
    options=options,
    bounds=(np.array([0.001]), np.array([5.0]))
)

# History tracking
history = {
    'iteration': [],
    'g_best': [],
    'p_best': [],
    'best_gamma': [],
    'silhouette': [],
    'dbi': []
}

def callback(optimizer):
    current_iter = optimizer.it
    best_pos = optimizer.swarm.best_pos
    best_cost = optimizer.swarm.best_cost
    
    # Hitung rata-rata p_best
    avg_pbest = np.mean([p.best_cost for p in optimizer.swarm.particles])
    
    # Simpan history
    history['iteration'].append(current_iter)
    history['g_best'].append(best_cost)
    history['p_best'].append(avg_pbest)
    history['best_gamma'].append(best_pos[0][0])
    
    # Evaluasi clustering dengan gamma terbaik
    try:
        gamma_val = best_pos[0][0]
        W_opt = rbf_kernel_fast(X_scaled, gamma_val)
        W_opt[W_opt < 0.01] = 0
        L_opt = laplacian(csr_matrix(W_opt), normed=True)
        eigvals_opt, eigvecs_opt = eigsh(L_opt, k=optimal_k, which='SM', tol=1e-4)
        U_opt = normalize(eigvecs_opt, norm='l2')
        kmeans_opt = KMeans(n_clusters=optimal_k, n_init=5, random_state=SEED)
        labels_opt = kmeans_opt.fit_predict(U_opt)
        
        sil = silhouette_score(U_opt, labels_opt)
        dbi = davies_bouldin_score(U_opt, labels_opt)
        
        history['silhouette'].append(sil)
        history['dbi'].append(dbi)
    except:
        history['silhouette'].append(0.0)
        history['dbi'].append(10.0)
    
    # Print progress
    print(f"Iter {current_iter}: γ={best_pos[0][0]:.4f} | G-best={best_cost:.4f} | P-best avg={avg_pbest:.4f}")

# Jalankan optimasi
print("Memulai optimasi PSO...")
best_cost, best_pos = optimizer.optimize(
    parallel_evaluation,  # Gunakan fungsi evaluasi paralel
    iters=50,
    X_scaled=X_scaled,    # Pass data sebagai argumen tambahan
    verbose=False,
    callback=callback
)

best_gamma = best_pos[0][0]
print(f"\nOptimasi selesai! Gamma optimal: {best_gamma:.4f}")

# =============================================
# EVALUASI HASIL OPTIMAL
# =============================================

# Hitung embedding dengan gamma optimal
W_opt = rbf_kernel_fast(X_scaled, best_gamma)
W_opt[W_opt < 0.01] = 0
L_opt = laplacian(csr_matrix(W_opt), normed=True)
eigvals_opt, eigvecs_opt = eigsh(L_opt, k=optimal_k, which='SM', tol=1e-4)
U_opt = normalize(eigvecs_opt, norm='l2')
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=10)
labels_opt = kmeans_opt.fit_predict(U_opt)

# Simpan hasil ke dataframe
df['Cluster'] = labels_opt

# =============================================
# VISUALISASI DAN ANALISIS (SAMA DENGAN SEBELUMNYA)
# =============================================

# Plot konvergensi PSO
plt.figure(figsize=(12, 6))
plt.plot(history['iteration'], history['g_best'], 'b-', label='Global Best')
plt.plot(history['iteration'], history['p_best'], 'r--', label='Rata-rata Personal Best')
plt.xlabel('Iterasi')
plt.ylabel('Nilai Fitness')
plt.title('Konvergensi PSO')
plt.legend()
plt.grid(True)
plt.show()

# Perbandingan sebelum dan sesudah PSO
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(U_norm[:,0], U_norm[:,1], c=labels_before, cmap='viridis')
ax1.set_title(f'Sebelum PSO (γ=0.1)\nSilhouette: {silhouette_score(U_norm, labels_before):.4f}')
ax2.scatter(U_opt[:,0], U_opt[:,1], c=labels_opt, cmap='viridis')
ax2.set_title(f'Sesudah PSO (γ={best_gamma:.4f})\nSilhouette: {silhouette_score(U_opt, labels_opt):.4f}')
plt.tight_layout()
plt.show()

# Feature importance
X = df.drop(columns=['Cluster', 'Kabupaten/Kota'])
y = df['Cluster']
rf = RandomForestClassifier(random_state=SEED)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Output hasil clustering
print("\nHasil Clustering:")
print(df[['Kabupaten/Kota', 'Cluster']].sort_values('Cluster'))
