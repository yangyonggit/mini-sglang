from minisgl.llm import LLM
from minisgl.core import SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", attention_backend="triton", cuda_graph_max_bs=0)
out = llm.generate(["Hello world"], SamplingParams(max_tokens=10))
print(out)
