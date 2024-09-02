RESULTS_DIR=$1
# CUM_REWARD="$(grep -hoP '\Q"cum_reward": \E\K\d+(?:\.\d+)?' $(find $RESULTS_DIR -type f -name "summary_info.json") | tee /dev/tty | paste -sd+ | bc -l)"
CUM_REWARD="$(grep -hoP '\Q"cum_reward": \E\K\d+(?:\.\d+)?' $(find $RESULTS_DIR -type f -name "summary_info.json") | paste -sd+ | bc -l)"
NUM_EXPERIMENTS="$(find $RESULTS_DIR -type f -name "summary_info.json" | wc -w)"
SUCCESS_RATE="$(echo "$CUM_REWARD/$NUM_EXPERIMENTS" | bc -l)"
echo "Cumulative reward: $CUM_REWARD
Num experiments: $NUM_EXPERIMENTS
Success rate: $SUCCESS_RATE
Successful runs:"
grep -oP '\Q"cum_reward": \E1' $(find $RESULTS_DIR -type f -name "summary_info.json") 