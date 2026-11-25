import os
from nbconvert import export
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. WORKER FUNCTION (Runs on each GPU)
# ==========================================
def process_shard_on_gpu(shard_df, gpu_id):
    # 1. Isolate the GPU
    # CHOSEN_GPUS = [2] 
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(CHOSEN_GPUS[gpu_id])
    
    # 2. Add your HF Token if using the script method
    # os.environ["HF_TOKEN"] = "hf_your_token_here"
    
    # Import inside the worker
    from vllm import LLM, SamplingParams
    
    print(f"[GPU {gpu_id}] Starting... processing {len(shard_df)} rows.")
    model_id = "meta-llama/Llama-3.1-8B-Instruct" #"Qwen/Qwen2.5-14B-Instruct"#
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_seqs=2560,
    )
    tokenizer=llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=500,
        repetition_penalty=1.1,  # Strictly penalizes repeating the same sentences      # Forces generation to stop when the assistant finishes
    )
    sys_prompt = (
    " You must generate the complete, fully written final document or answer based on the user's query. "
    "Do NOT provide outlines, templates, placeholders, or instructions for the user to fill in. "
)
    prompts = []
    for _, row in tqdm(shard_df.iterrows(),total=len(shard_df)):
        messages = [
                {"role": "system", "content": "You are a helpful, accurate assistant, who responds to user properly. Do not generate templates just answer"},
                {"role": "user", "content": row['query_with_presupposition']}
            ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    # Save results to the shard
    shard_df = shard_df.copy()
    shard_df['model_response'] = [out.outputs[0].text.strip() for out in outputs]
    
    # CRITICAL FIX: Save directly to disk to avoid multiprocessing memory issues
    temp_file = "temp_shard_gpu_{gpu_id}.csv"
    shard_df.to_csv(temp_file, index=False)
    
    print(f"[GPU {gpu_id}] Finished and saved to {temp_file.name}")

# ==========================================
# 2. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Must use 'spawn' for CUDA
    # try:
    #     mp.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass

    # --- Setup Paths ---
    # work_path = Path(os.environ.get('WORK', './')).resolve()
    # temp_path = Path("/path/to/temp").resolve()
    # os.environ['HF_HOME'] = str(work_path / "thesis_storage/cache/huggingface")
    # os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    
    # --- Load Your Data ---
    print("Loading data...")
    # df = pd.read_csv(...) # Load your actual dataframe here
    df=pd.read_csv(  "data/queries.csv") # Example loading, replace with your actual path
    # --- Safe Data Splitting ---
    import torch

    # Returns an integer of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    NUM_GPUS = 1#gpu_count if gpu_count > 0 else 1
    shards = [df.iloc[i::NUM_GPUS].copy() for i in range(NUM_GPUS)]
    
    # --- Launch Explicit Non-Daemonic Processes ---
    print(f" Launching independent processes across {NUM_GPUS} GPUs...")
    processes = []
    
    for i in range(NUM_GPUS):
        # We use mp.Process instead of mp.Pool to avoid the daemon error
        p = mp.Process(target=process_shard_on_gpu, args=(shards[i], i))
        processes.append(p)
        p.start()
        
    # Wait for all GPUs to finish their work
    for p in processes:
        p.join()
        
    # --- Recombine the Temporary Files ---
    print("All GPUs finished. Combining temporary files...")
    
    # Read the 3 temporary CSVs back into Pandas
    temp_dfs = []
    for i in range(NUM_GPUS):
        temp_file =f"temp_shard_gpu_{i}.csv"
        if temp_file.exists():
            temp_dfs.append(pd.read_csv(temp_file))
            # Clean up the temp file after reading
            os.remove(temp_file)
            
    # Combine and sort by original index
    final_df = pd.concat(temp_dfs).sort_index()
    
    # Save the final
    save_path = "parallel_greedy_results_llama_v4.csv"
    final_df.to_csv(save_path, index=False)
    
    print(f" Success! Data completely processed and saved to {save_path}")