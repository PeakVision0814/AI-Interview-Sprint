# src/config.py
import os
from pathlib import Path

# 1. 定位 Project Root (基于当前文件位置向上回溯)
# current: project/src/config.py -> parent: project/src -> parent: project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. 数据目录
DATA_DIR = PROJECT_ROOT / "data"
PRETRAINED_MODEL_DIR = DATA_DIR / "pretrained_models/bert-base-chinese"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"

# 3. 输出目录 (自动创建不存在的文件夹)
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_DIR = OUTPUT_DIR / "train"
CHECKPOINT_DIR = TRAIN_DIR / "checkpoints"
LOG_DIR = TRAIN_DIR / "logs"
MODEL_SAVE_DIR = TRAIN_DIR / "saved_model"

TEST_DIR = OUTPUT_DIR / "test"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# 4. 自动初始化目录结构的函数
def init_directories():
    """在代码运行前调用，确保目录存在"""
    for path in [CHECKPOINT_DIR, LOG_DIR, MODEL_SAVE_DIR, TEST_DIR, ANALYSIS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {path}")

# 5. 全局超参数 (可选，也可以放在 TrainingArgs 里)
MAX_LEN = 128
BATCH_SIZE = 32