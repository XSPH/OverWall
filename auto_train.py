import os
import glob
import subprocess
import re

def modify_ppo(student_reinforcing_val):
    ppo_path = "rsl_rl/rsl_rl/algorithms/ppo.py"
    with open(ppo_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 查找 def __init__ 里面的 student_reinforcing = False
    new_content = re.sub(
        r'student_reinforcing\s*=\s*(True|False)',
        f'student_reinforcing = {student_reinforcing_val}',
        content
    )
    with open(ppo_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Set student_reinforcing to {student_reinforcing_val} in ppo.py")

def modify_train(new_model_path):
    train_path = "legged_gym/legged_gym/scripts/train.py"
    with open(train_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 替换 path_1
    new_content = re.sub(
        r'path_1\s*=\s*".*?"',
        f'path_1 = "{new_model_path}"',
        content
    )
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Updated path_1 in train.py to:\n{new_model_path}")

def get_latest_model():
    log_dir = "legged_gym/logs"
    # 获取最新的包含模型文件的完整目录路径
    list_of_files = glob.glob(f"{log_dir}/*/*/*.pt") 
    if not list_of_files:
        raise Exception("No model found!")
    # 取最新的 pt 文件
    latest_file = max(list_of_files, key=os.path.getmtime)
    return os.path.abspath(latest_file)

if __name__ == "__main__":
    print("-" * 50)
    print("1. 准备第一阶段训练...")
    modify_ppo("False") # 确保第一阶段为 False
    
    print("-" * 50)
    print("2. 开始第一阶段训练: train.py --task=go2w --headless")
    subprocess.run(["python", "legged_gym/legged_gym/scripts/train.py", "--task=go2w", "--headless"], check=True)
    
    print("-" * 50)
    print("3. 第一阶段完毕，查找最新模型...")
    latest_pt = get_latest_model()
    print(f"找到最新模型:\n{latest_pt}")
    
    print("-" * 50)
    print("4. 准备第二阶段: 修改 ppo.py = True 并替换 train.py 中的 path_1")
    modify_ppo("True")
    modify_train(latest_pt)
    
    print("-" * 50)
    print("5. 开始第二阶段训练: train.py --task=go2w --resume ...")
    subprocess.run([
        "python", "legged_gym/legged_gym/scripts/train.py", 
        "--task=go2w", "--resume", "--student_reinforcing", 
        "--max_iterations=5000", "--headless"
    ], check=True)
    print("全部训练任务完成！")
