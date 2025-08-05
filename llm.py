from vllm import LLM
from vllm import SamplingParams

# initialize LLM with AWQ quantization
llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    quantization="awq",
    device="cuda",   # auto device mapping
    dtype="auto"     # let vLLM infer fp16/int8 appropriately
)

prompt = "Tell me a D&D story about a dragon and a young parent."
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

outputs = llm.generate([prompt], params)
print(outputs[0].outputs[0].text)
