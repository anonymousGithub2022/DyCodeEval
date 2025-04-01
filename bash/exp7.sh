#python prompt_generation.py --llm_id=6


####################

N=10
gpu_id=7
partial_id=0
lora_data_id=0

for temp in 0 0.8; do
  for data_id in 1 3; do
    for model_id in {3..8}; do
      CUDA_VISIBLE_DEVICES=${gpu_id} python generate_code.py \
          --n=$N \
          --model_id="${model_id}" \
          --partial_id=$partial_id \
          --lora_data_id=$lora_data_id \
          --data_id=$data_id \
          --temperature=$temp
      done
  done
done