function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}
for i in $(seq 102 920);
do
python run_demo.py \
    --task_name webarena.$i \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --slow_mo 500 \
    --headless True \
    --use_history False \
    --use_html False \
    --multi_actions False \
    --use_screenshot False &
    pwait 10
done