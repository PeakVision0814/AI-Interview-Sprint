import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from safetensors.torch import load_file
from src.models.model_bert import BertClassifier
from src.config import PRETRAINED_MODEL_DIR

class IntentPredictor:
    def __init__(self, checkpoint_path, num_labels=3):
        """
        初始化推理引擎
        :param checkpoint_path: 训练好的模型权重路径 (包含 model.safetensors)
        :param num_labels: 类别数量
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {0: "MOVE", 1: "STOP", 2: "GRAB"} 

        print(f"正在加载推理资源... (Device: {self.device})")

        self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
        self.model = BertClassifier(model_path=PRETRAINED_MODEL_DIR, num_labels=num_labels)

        weight_path = f"{checkpoint_path}/model.safetensors"
        try:
            state_dict = load_file(weight_path)
            
            # ★★★ 核心修改在这里：strict=False ★★★
            # 作用：忽略 checkpoint 中多余的 'loss_fct.weight' 参数，因为推理不需要计算 Loss
            self.model.load_state_dict(state_dict, strict=False)
            
            print("✅ 权重加载成功！(已忽略多余的 Loss 权重)")
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            raise e
        
        self.model.to(self.device)
        self.model.eval() 

    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs)
            
        probs = F.softmax(logits, dim=-1) 
        confidence, pred_id = torch.max(probs, dim=-1)
        
        pred_id = pred_id.item()
        confidence = confidence.item()
        
        return {
            "text": text,
            "intent": self.label_map[pred_id],
            "confidence": f"{confidence:.4f}",
            "intent_id": pred_id
        }