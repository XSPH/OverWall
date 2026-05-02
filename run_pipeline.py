import os
import glob
import subprocess
import re

def modify_ppo(student_reinforcing_val):
    ppo_path = "rsl_rl/rsl_rl/algorithms/ppo.py"
    with open(ppo_path, "r") as f:
        content = f.read()
    
    # regex substitution
    new_content = re.sub(
        r'student_reinforcing\s*=\s*(True|False)',
        f'student_reinforcing = {student_reinforcing_val}',
        content
    )
    with open(ppo_path, "w") as f:
        f.write(new_content)

def modify_train(new_model_path):
    train_path = "legged_gym/legged_gym/scripts/train.py"
    with open(train_path, "r") as f:
        content = f.read()
    
    new_content = re.sub(
        r'path_1\s*=\s*".*?"',
        f'path_1 = "{new_model_path}"',
        content
    )
    with open(train_path, "w") as f:
        f.write(new_content)

def get_latest_model(task_name):
    # Assuming logs format: legged_gym/logs/<task_name>_<something>/...
    # Since in train.py it's hard to guess the exact subfolder, we can just find the newest .pt file in logs/
    log_dir = "legged_gym/logs"
    list_of_files = glob.glob(f"{log_dir}/*/*/*.pt") # e.g. logs/go2w_flat_turn/May01_.../model_...pt
    if not list_of_files:
        raise Exception("No model found!")
    latest_file = max(list_of_files, key=os.path.getmtime)
    return os.path.abspath(latest_file)

if __name__ == "__main__":
    print("=== Starting Phase 1 Training ===")
    modify_ppo("False")
    subprocess.run(["python", "legged_gym/legged_gym/scripts/train.py", "--task=go2w", "--headless"], check=True)
    
    print("=== Phase 1 Complete. Finding latest model... ===")
    latest_pt = get_latest_model("go2w")
    print(f"Latest model found: {latest_pt}")
    
    print("=== Preparing Phase 2 (Student Reinforcing) ===")
    modify_ppo("True")
    modify_train(latest_pt)
    
    print("=== Starting Phase 2 Training ===")
    subprocess.run([
        "python", "legged_gym/legged_gym/scripts/train.py", 
        "--task=go2w", "--resume", "--student_reinforcing", 
        "--max_iterations=5000", "--headless"
    ], check=True)
