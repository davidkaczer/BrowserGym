for i in $(seq 50 100);
do
python run_demo.py \
    --task_name webarena.$i \
    --model_name HuggingFaceM4/Idefics3-8B-Llama3 \
    --slow_mo 500 \
    --headless True \
    --use_history False \
    --use_html False \
    --multi_actions False \
    --use_screenshot True \

done