from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel
from torch import nn
import torch
from typing import Optional, Union, Tuple, List, Unpack
# from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import Cache
from transformers.generation.utils import CausalLMOutputWithPast

from generate import GenerationMixin
from modeling_llama import LlamaForCausalLM

import lovely_tensors as lt
lt.monkey_patch()



from transformers import AutoTokenizer
import time
def main():
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).cuda()
    
    print(f"{model=}")
    
    time.sleep(2)
    
    LONG_CONTENT = "Hello, how are you?" * 100 + " The capital of France is"
    
    
    inputs = tokenizer([LONG_CONTENT for _ in range(128)], return_tensors="pt").to(model.device)
    print(inputs.input_ids.shape)
    torch.cuda.reset_peak_memory_stats()
    
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=1,
        use_cache=False,
        logits_to_keep=1,
    )
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    print(f"{outputs=}")
    
    time.sleep(1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    del outputs
    torch.cuda.empty_cache()
    
    torch.cuda.reset_peak_memory_stats()
    
    time.sleep(2)
    
    # Build position ids
    batch_size = inputs.input_ids.shape[0]
    seq_len = inputs.input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(model.device)
    cache_position = torch.arange(seq_len).to(model.device) # (seq_len,)
    logits_to_keep = 1
    
    model.eval()
    with torch.no_grad():
        print(f"Memory allocated before forward: {torch.cuda.max_memory_allocated() / 1024**2} MB")
        outputs = model.forward(
            **inputs, 
            position_ids=position_ids, 
            cache_position=cache_position,
            return_dict=True,
            logits_to_keep=logits_to_keep,
            use_cache=False,
        )
        print(f"Memory allocated after forward: {torch.cuda.max_memory_allocated() / 1024**2} MB")
        print(f"{outputs=}")
        print(f"{outputs.logits.shape=}")
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB")
        time.sleep(1)
        del outputs
        del logits
        torch.cuda.empty_cache()
        print(tokenizer.decode(next_token, skip_special_tokens=True))
        time.sleep(2)

if __name__ == "__main__":
    main()