
# python prompt_generation.py --llm_id=1
###############################



N=10
model_id=0
gpu_id=1

CUDA_VISIBLE_DEVICES="${gpu_id}" python overfit_train.py \
        --model_id=$model_id \
        --data_id=1

sleep 3000

for temp in 0 0.8; do
  for lora_data_id in 0 1; do
    for partial_id in {0..4}; do
      for data_id in 1 3; do
        # Set CUDA_VISIBLE_DEVICES and run the python script
        CUDA_VISIBLE_DEVICES=${gpu_id} python generate_code.py \
          --n=$N \
          --model_id=$model_id \
          --partial_id=$partial_id \
          --lora_data_id=$lora_data_id \
          --data_id=$data_id \
          --temperature=$temp
      done
    done
  done
done
