import matplotlib.pyplot as plt
import numpy as np

# ── Strong scaling data ──────────────────────────────────────────────
omp_threads      = [1, 2, 4, 8, 16, 20]
omp_strong_times = [5.02, 2.74, 1.48, 0.94, 0.58, 0.65]

mpi_procs        = [1, 2, 4, 8, 16]
mpi_strong_times = [5.64, 3.15, 1.84, 1.28, 1.04]

cuda_sms         = [1, 2, 4, 8, 16, 32, 66]
cuda_strong_times= [0.2943, 0.1381, 0.0712, 0.0359, 0.0181, 0.0092, 0.0048]

# ── Weak scaling data ────────────────────────────────────────────────
omp_weak_threads = [1, 2, 4, 8, 16, 20]
omp_weak_times   = [0.30, 0.34, 0.53, 0.81, 0.78, 1.21]

mpi_weak_procs   = [1, 2, 4, 8, 16]
mpi_weak_times   = [0.41, 0.59, 0.70, 1.13, 2.06]

cuda_weak_sms    = [1, 2, 4, 8, 16, 32, 66]
cuda_weak_times  = [0.2985, 0.3004, 0.3151, 0.3543, 0.5179, 0.5603, 0.5737]

# ── Speedup calculation ──────────────────────────────────────────────
omp_speedup  = [omp_strong_times[0] / t for t in omp_strong_times]
mpi_speedup  = [mpi_strong_times[0] / t for t in mpi_strong_times]
cuda_speedup = [cuda_strong_times[0] / t for t in cuda_strong_times]

# ════════════════════════════════════════════════════════════════════
# Plot 1: Strong scaling — OpenMP & MPI
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(omp_threads, omp_speedup,  'o-', color='royalblue',  label='OpenMP')
ax.plot(mpi_procs,   mpi_speedup,  's-', color='tomato',     label='MPI')
ax.plot(omp_threads, omp_threads,  '--', color='gray',        label='Ideal')

ax.set_xlabel('Threads / Processes')
ax.set_ylabel('Speedup')
ax.set_title('Strong Scaling — OpenMP & MPI (N=2048)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('strong_scaling_omp_mpi.png', dpi=150)
plt.close()
print("Saved: strong_scaling_omp_mpi.png")

# ════════════════════════════════════════════════════════════════════
# Plot 2: Strong scaling — CUDA
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(cuda_sms, cuda_speedup, 'o-', color='mediumseagreen', label='CUDA')
ax.plot(cuda_sms, cuda_sms,     '--', color='gray',            label='Ideal')

ax.set_xlabel('Streaming Multiprocessors (SMs)')
ax.set_ylabel('Speedup')
ax.set_title('Strong Scaling — CUDA (N=1024, 128 threads/block)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('strong_scaling_cuda.png', dpi=150)
plt.close()
print("Saved: strong_scaling_cuda.png")

# ════════════════════════════════════════════════════════════════════
# Plot 3: Weak scaling — OpenMP & MPI
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(omp_weak_threads, omp_weak_times, 'o-', color='royalblue', label='OpenMP')
ax.plot(mpi_weak_procs,   mpi_weak_times, 's-', color='tomato',    label='MPI')
ax.axhline(y=omp_weak_times[0], linestyle='--', color='gray',       label='Ideal')

ax.set_xlabel('Threads / Processes')
ax.set_ylabel('Time (s)')
ax.set_title('Weak Scaling — OpenMP & MPI (N scales as p^(1/3))')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('weak_scaling_omp_mpi.png', dpi=150)
plt.close()
print("Saved: weak_scaling_omp_mpi.png")

# ════════════════════════════════════════════════════════════════════
# Plot 4: Weak scaling — CUDA
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(cuda_weak_sms, cuda_weak_times, 'o-', color='mediumseagreen', label='CUDA')
ax.axhline(y=cuda_weak_times[0], linestyle='--', color='gray',         label='Ideal')

ax.set_xlabel('Streaming Multiprocessors (SMs)')
ax.set_ylabel('Time (s)')
ax.set_title('Weak Scaling — CUDA (N scales as SMs^(1/3))')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('weak_scaling_cuda.png', dpi=150)
plt.close()
print("Saved: weak_scaling_cuda.png")

# ════════════════════════════════════════════════════════════════════
# Plot 5: Strong scaling — CUDA at N=8096
# ════════════════════════════════════════════════════════════════════
cuda_8096_sms     = [16, 32, 66]
cuda_8096_times   = [17.5366, 8.8457, 4.5634]
cuda_8096_speedup = [cuda_8096_times[0] / t for t in cuda_8096_times]
cuda_8096_ideal   = [1.0, 2.0, 66/16]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(cuda_8096_sms, cuda_8096_speedup, 'o-', color='mediumseagreen', label='CUDA')
ax.plot(cuda_8096_sms, cuda_8096_ideal,   '--', color='gray',            label='Ideal')

ax.set_xlabel('Streaming Multiprocessors (SMs)')
ax.set_ylabel('Speedup (relative to 16 SMs)')
ax.set_title('Strong Scaling — CUDA (N=8096, 128 threads/block)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('strong_scaling_cuda_8096.png', dpi=150)
plt.close()
print("Saved: strong_scaling_cuda_8096.png")

# ════════════════════════════════════════════════════════════════════
# Plot 6: Weak scaling — CUDA at N=8096 baseline
# ════════════════════════════════════════════════════════════════════
cuda_8096_weak_sms   = [16, 32, 66]
cuda_8096_weak_times = [17.5366, 18.3056, 19.5906]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(cuda_8096_weak_sms, cuda_8096_weak_times, 'o-', color='mediumseagreen', label='CUDA')
ax.axhline(y=cuda_8096_weak_times[0], linestyle='--', color='gray', label='Ideal')

ax.set_xlabel('Streaming Multiprocessors (SMs)')
ax.set_ylabel('Time (s)')
ax.set_title('Weak Scaling — CUDA (N=8096 baseline, N scales as SMs^(1/3))')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('weak_scaling_cuda_8096.png', dpi=150)
plt.close()
print("Saved: weak_scaling_cuda_8096.png")
