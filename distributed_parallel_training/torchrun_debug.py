import os
import torch
import torch.distributed as dist

def init_process():
    print("Initializing process group...")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK not set in environment")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Local rank: {local_rank}")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"CUDA available. Using device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Try different backends
    backends = ['nccl', 'gloo']
    for backend in backends:
        try:
            print(f"Attempting to initialize with {backend} backend...")
            dist.init_process_group(backend=backend)
            print(f"Process group initialized with {backend} backend.")
            return
        except Exception as e:
            print(f"Failed to initialize with {backend} backend: {str(e)}")
    
    raise RuntimeError("Failed to initialize process group with any backend.")

def main():
    print("Starting main function.")
    init_process()
    print("Distributed environment initialized.")
    dist.destroy_process_group()
    print("Process group destroyed.")

if __name__ == "__main__":
    main()
