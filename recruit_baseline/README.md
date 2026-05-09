To replicate the experimental results,

1) Download the data folder ICPSR_22140 and put in same directory

2) Run the experiments over all the parameters, e.g. via the Bash script below

ALL_B=(100 150 200)
ALL_GAMMA=(0.5 0.7 0.9)
ALL_N=(5 10 15)
ALL_EPS=(0.0 0.2 0.4)
NUM_TIMES=30
for B in "${ALL_B[@]}"; do
  for GAMMA in "${ALL_GAMMA[@]}"; do
    for N in "${ALL_N[@]}"; do
      python3 run_experiments.py 0 $NUM_TIMES $B $GAMMA $N 0.0
      python3 run_experiments.py 1 $NUM_TIMES $B $GAMMA $N 0.0
      python3 run_experiments.py 4 $NUM_TIMES $B $GAMMA $N 0.0
      python3 run_experiments.py 5 $NUM_TIMES $B $GAMMA $N 0.0
      for EPS in "${ALL_EPS[@]}"; do
        python3 run_experiments.py 2 $NUM_TIMES $B $GAMMA $N $EPS
        python3 run_experiments.py 3 $NUM_TIMES $B $GAMMA $N $EPS
      done
    done
  done
done

3) Run python3 plot_figures.py
