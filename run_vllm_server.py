import subprocess
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Pick one model: AWQ or GPTQ
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
QUANTIZATION = "awq"
DTYPE = "auto"

# vLLM command to run as OpenAI-compatible API server
cmd = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_ID,
    "--quantization", QUANTIZATION,
    "--dtype", DTYPE,
    "--max-model-len", "1024",  # lower memory footprint
    "--gpu-memory-utilization", "0.8",  # Be gentle to avoid crash!
]


def main():
    print(f"Launching vLLM server with model: {MODEL_ID}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
