from llama_cpp import Llama
import os
print("Testing Llama-cpp Vulkan Initialization...")
llm = Llama(model_path="models/text/deepseek/deepseek-r1-distill-qwen-7b-q4_k_m.gguf", n_gpu_layers=-1, verbose=True)
print("Finished!")
