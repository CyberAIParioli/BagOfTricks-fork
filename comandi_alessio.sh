python3 main.py \
    --attack AutoDAN \
    --target_model_path lmsys/vicuna-7b-v1.5 \
    --instructions_path data/harmful_bench_50.csv \
    --save_result_path ./exp_results/autodan_attacks/ \
    --exp_name my_autodan_run --device_id -1