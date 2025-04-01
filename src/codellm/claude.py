import threading
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Tuple
from copy import deepcopy
import json
import boto3
import re
from .io_types import CodeTask, CodeLLMOutput
from .abst_llm import AbstLLM, vLLM




class Claude3_5LLM(AbstLLM):
    def __init__(self, model_name, is_lora):
        super().__init__(model_name, is_lora)

        self.prefix_sym = "<｜fim▁begin｜>"
        self.suffix_sym = "<｜fim▁end｜>"
        self.mid_sym = "<｜fim▁hole｜>"
        self.stop = [
            "\n>>>", "\n$", '\nclass',
            '\ndef', '\n#', '\nprint',
            '\n}\n', "\n@",
            "\nif __name__ == '__main__':"
        ]

        self.client = boto3.client('bedrock-runtime', region_name='us-west-2')

    import re

    def extract_code(self, text):
        """
        Extracts the first code block from a given text, whether or not a language is specified.

        Args:
            text (str): The input text containing a code block.

        Returns:
            str: The extracted code, or an empty string if no code block is found.
        """
        match = re.search(r"```(?:\w+)?\n(.*?)\n```", text, re.DOTALL)
        return match.group(1) if match else ""

    def _task2prompt(self, task: CodeTask) -> str:
        # return self._lm_task2prompt(task)
        prompt_template = \
'''You are a helpful coding assistant producing high-quality code. Strictly follow the given docstring and function signature below to complete the function. Your code should always gracefully return. Your response should include all dependencies, headers and function declaration to be directly usable (even for the ones seen in the given part). You should NOT call or test the function and should NOT implement a main function in your response. You should implement the function in Python. You should output your complete implementation in a single code block wrapped by triple backticks.

```python
{code_prompt}
```

You should output your complete implementation in a single code block.
'''
        prompt = prompt_template.format(code_prompt = task.prefix)

        return prompt


    def _prediction2output(self, prompt, task: CodeTask, model_prediction, cost_time) -> CodeLLMOutput:

        code_blocks = self.extract_code(model_prediction)
        logits = None
        output = CodeLLMOutput(
            prompt_input=prompt,
            original_task=task,
            original_output=model_prediction,
            text=model_prediction,
            logits=logits,
            final_code=code_blocks,
            cost_time=cost_time
        )
        return output

    def _invoke_model(self, payload, result_list, index):
        try:
            response = self.client.invoke_model(
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
                body=json.dumps(payload),
            )
            stream = response['body'].read().decode('utf-8')
            res = json.loads(stream)['content'][0]['text']
            result_list[index] = res
        except self.client.exceptions.ServiceQuotaExceededException as e:
            time.sleep(60)
            self._invoke_model(payload, result_list, index)
        except self.client.exceptions.ThrottlingException as e:
            time.sleep(60)
            self._invoke_model(payload, result_list, index)
        except Exception as e:
            print(e)
            result_list[index] = None

    def _prompt2output_batch(self, prompts: list[str]) -> list[str]:
        payload_list = [{
            "messages": [
                {"content": f"{pt}", "role": "user"}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "anthropic_version": "bedrock-2023-05-31"  # Required key
        } for pt in prompts]

        all_res = [None] * len(payload_list)  # Initialize result list with None
        threads = []

        for i, payload in enumerate(payload_list):
            thread = threading.Thread(target=self._invoke_model, args=(payload, all_res, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        all_res = [d for d in all_res if d is not None]
        return all_res


    def code_gen_batch(self, tasks: List[CodeTask]) -> List[CodeLLMOutput]:
        if not self.is_init:
            raise NotImplementedError
        prompts = [self._task2prompt(task) for task in tasks]
        t1 = time.time()
        outputs = self._prompt2output_batch(prompts)
        t2 = time.time()
        cost_time = t2 - t1
        res = [self._prediction2output(p, t, o, cost_time) for p, t, o in zip(prompts, tasks, outputs)]
        return res