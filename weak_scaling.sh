echo "=== Compiling ==="
gcc -O2 -o task0 Task0.c
gcc -O2 -fopenmp -o task1 Task1.c
mpicc -O2 -o task2 Task2.c

echo ""
echo "=== Weak scaling: OpenMP ==="
printf "%-10s %-8s %10s\n" "Threads" "N" "Time (s)"
for threads in 1 2 4 8 16 20; do
    N=$(awk "BEGIN {printf \"%d\", int(1024 * ($threads^(1.0/3.0)) + 0.5)}")
    t=$(OMP_NUM_THREADS=$threads ./task1 $N | grep "IKJ  schedule(guided)" | awk '{print $3}')
    printf "%-10s %-8s %10s\n" "$threads" "$N" "$t"
done

echo ""
echo "=== Weak scaling: MPI ==="
printf "%-10s %-8s %10s\n" "Processes" "N" "Time (s)"
for procs in 1 2 4 8 16; do
    N=$(awk "BEGIN {printf \"%d\", int(1024 * ($procs^(1.0/3.0)) + 0.5)}")
    t=$(mpirun --oversubscribe -n $procs ./task2 $N | grep "Time" | awk '{print $3}')
    printf "%-10s %-8s %10s\n" "$procs" "$N" "$t"
done
