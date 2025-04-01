from vllm import LLM
from utils import load_my_dataset, load_benchmark_model
from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen2.5-7B-Instruct" #
data = load_my_dataset(0)
llm = LLM(model=model_name,
          trust_remote_code=True,
          max_model_len=2048)

sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.95,
            max_tokens=1024,)


for d in data:
    outputs = llm.generate(
        prompts=d.prefix,
        sampling_params = sampling_params
    )
    print(outputs[0].outputs[0].text)
    exit(0)

        # print(f"PREFIX: \n{d.prefix}")
        # print('---------------------------------')
        # print(f"SOLUTION: \n{d.solution}" )
        # print('---------------------------------')

        # print("DOCSTRING: \n", d.docstring)
        # print('---------------------------------')
        # print("CLEAN CODE: \n", d.clean_code)
        # print('---------------------------------')
        # print("INST :", d.instruction)
        # print('---------------------------------')
        # print("DEMO :", d.demo_str)
        
    print('*' * 50)



