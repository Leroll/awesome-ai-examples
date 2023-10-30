"""
计算 ceval 的 acc

单文件可独立运行
需要配合本地版的 ceval-exam 数据集
支持chatglm模型 及 lora微调后的 chatglm模型
基于 baichuan eval 代码改写
"""

import argparse
import json
import os
 
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import pathlib
import gc
import ctypes
import torch
import time 
import codecs
from peft import PeftModel


def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
 


class CEval:
    DATA_PATH = "ceval-exam"
    TASK2DESC = {
        "high_school_physics": "高中物理",
        "fire_engineer": "注册消防工程师",
        "computer_network": "计算机网络",
        "advanced_mathematics": "高等数学",
        "logic": "逻辑学",
        "middle_school_physics": "初中物理",
        "clinical_medicine": "临床医学",
        "probability_and_statistics": "概率统计",
        "ideological_and_moral_cultivation": "思想道德修养与法律基础",
        "operating_system": "操作系统",
        "middle_school_mathematics": "初中数学",
        "chinese_language_and_literature": "中国语言文学",
        "electrical_engineer": "注册电气工程师",
        "business_administration": "工商管理",
        "high_school_geography": "高中地理",
        "modern_chinese_history": "近代史纲要",
        "legal_professional": "法律职业资格",
        "middle_school_geography": "初中地理",
        "middle_school_chemistry": "初中化学",
        "high_school_biology": "高中生物",
        "high_school_chemistry": "高中化学",
        "physician": "医师资格",
        "high_school_chinese": "高中语文",
        "tax_accountant": "税务师",
        "high_school_history": "高中历史",
        "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
        "high_school_mathematics": "高中数学",
        "professional_tour_guide": "导游资格",
        "veterinary_medicine": "兽医学",
        "environmental_impact_assessment_engineer": "环境影响评价工程师",
        "basic_medicine": "基础医学",
        "education_science": "教育学",
        "urban_and_rural_planner": "注册城乡规划师",
        "middle_school_biology": "初中生物",
        "plant_protection": "植物保护",
        "middle_school_history": "初中历史",
        "high_school_politics": "高中政治",
        "metrology_engineer": "注册计量师",
        "art_studies": "艺术学",
        "college_economics": "大学经济学",
        "college_chemistry": "大学化学",
        "law": "法学",
        "sports_science": "体育学",
        "civil_servant": "公务员",
        "college_programming": "大学编程",
        "middle_school_politics": "初中政治",
        "teacher_qualification": "教师资格",
        "computer_architecture": "计算机组成",
        "college_physics": "大学物理",
        "discrete_mathematics": "离散数学",
        "marxism": "马克思主义基本原理",
        "accountant": "注册会计师",
   }
 
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
    

    def load_local_dataset(self, task_name):
        """读取离线下载的ceval文件，形成与hf下载相同的格式
        """
        file_path = pathlib.Path(self.DATA_PATH).glob(f'*/{task_name}*')
        sets = ['dev', 'test', 'val']
        dataset = DatasetDict({sets[idx]:load_dataset('csv', data_files=str(f))['train'] for idx, f in enumerate(file_path)})  
        return dataset
    

    def build_example(self, data, with_answer: bool = True):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\n答案：{answer}"

    @torch.no_grad()
    def run_single_task(self, task_name: str, shot: int, split: str):
        t1 = time.time()
        # dataset = load_dataset(self.DATA_PATH, task_name)  
        dataset = self.load_local_dataset(task_name)

        results = []
        acc = 0
        for data in tqdm(dataset[split]):
            prompt = f"以下是中国关于{self.TASK2DESC[task_name]}考试的单项选择题，请选出其中的正确答案。\n"
            if shot != 0:
                shuffled = dataset["dev"].shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += "\n" + self.build_example(shuffled[i], with_answer=True)
            prompt += "\n" + self.build_example(data, with_answer=False)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)


            logits = self.model(
                    input_ids=input_ids,
                ).logits[:,-1].flatten()

            candidate_logits = [logits[self.tokenizer(label, add_special_tokens=False).input_ids[-1]] for label in ["A", "B", "C", "D"]]
            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            probs = (
                torch.nn.functional.softmax(
                    candidate_logits,
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]
 
            results.append(
                    {
                        "prompt": prompt,
                        "correct": answer == data["answer"].strip().upper(),
                        "answer": answer,
                    }
                )
            acc += answer == data["answer"].strip().upper()

            clean_memory()  # 控制显存

        acc /= len(dataset[split])
        cost = time.time() - t1
        results.append(
                {
                    task_name: self.TASK2DESC[task_name], 
                    'cost_time': f"{cost:.02f}",
                    'acc': f"{acc:.02%}"
                }
            )  # 增加单组的 acc
        return results, acc
 

    def run(self, shot: int, split: str):
        t1 = time.time()
        results, accs = {}, {}
 
        # run all task
        for task_name in self.TASK2DESC:
            print("=" * 100)
            print(f"run task: {task_name}")
            result, acc = self.run_single_task(task_name, shot, split)
            results[task_name] = result
            accs[task_name] = acc
            result_path = os.path.join(self.output_dir, f"{task_name}.json")
            with codecs.open(result_path, "w", 'utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"save result to {result_path}")
 
        # results
        average_acc = sum(accs.values()) / len(accs)  
        print(f"average acc: {average_acc}")
        cost = time.time() - t1

        accs["average acc"] = f"{average_acc:.02%}"
        accs["cost_time"] = f"{cost:.02f}"
        acc_path = os.path.join(self.output_dir, "acc.json")
        with codecs.open(acc_path, "w", 'utf-8') as f:
            json.dump(accs, f, indent=4, ensure_ascii=False)


def load_model(model_name_or_path:str, device:str, use_lora:bool = False, lora_path:str = None):
    """加载模型
    """
    if 'chatglm' in model_name_or_path and not use_lora:
        print('load Chatglm Model...')
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    if ('chatglm' in model_name_or_path) and use_lora:
        print('load Chatglm Model with lora...')
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, lora_path, device_map='auto')
        model.half().cuda(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float32
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=False,
            add_eos_token=False,
            padding_side="left",
        )

    return model, tokenizer
        

def main(model_name_or_path, shot, split, output_dir, device, use_lora, lora_path):
    model, tokenizer = load_model(model_name_or_path, device, use_lora, lora_path)
    model.eval()
    ceval = CEval(model, tokenizer, output_dir)
    ceval.run(shot, split)
 

if __name__ == "__main__":
    model_name_or_path = '/path/to/chatglm-6b'

    use_lora = True
    lora_path = '/path/to/lora'

    shot = 5 # number of shot for few-shot learning
    split = 'val'  # split of dataset to evaluate
    output_dir = 'ceval_output'  # output directory
    device = 'cuda:1'

    main(model_name_or_path, shot, split, output_dir, device, use_lora, lora_path)