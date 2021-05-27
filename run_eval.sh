model_path="./models/UM_eval"
output_path="./outputs/UM_eval"
log_path="./logs/UM_eval"
model_file='./models/UM/model_seed_6.pkl'

CUDA_VISIBLE_DEVICES=7 python -W ignore ./main_eval.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file}
