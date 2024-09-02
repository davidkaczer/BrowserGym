for i in $(seq 0 0);
do
python run_demo.py \
    --task_name webarena.$i \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --headless True \
    --use_history False \
    --use_html False \
    --multi_actions False \
    --use_screenshot False \

done