N=2048

echo "=== Compiling ==="
gcc -O2 -o task0 Task0.c
gcc -O2 -fopenmp -o task1 Task1.c
mpicc -O2 -o task2 Task2.c

echo ""
echo "=== Serial baseline (Task0, N=$N) ==="
./task0 $N

echo ""
echo "=== Strong scaling: OpenMP (N=$N) ==="
printf "%-10s %10s\n" "Threads" "Time (s)"
for threads in 1 2 4 8 16 20; do
    t=$(OMP_NUM_THREADS=$threads ./task1 $N | grep "IKJ  schedule(guided)" | awk '{print $3}')
    printf "%-10s %10s\n" "$threads" "$t"
done

echo ""
echo "=== Strong scaling: MPI (N=$N) ==="
printf "%-10s %10s\n" "Processes" "Time (s)"
for procs in 1 2 4 8 16; do
    t=$(mpirun --oversubscribe -n $procs ./task2 $N | grep "Time" | awk '{print $3}')
    printf "%-10s %10s\n" "$procs" "$t"
done
