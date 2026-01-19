"""
本地 LoRA 训练器

使用用户自己的对话数据微调个性化 adapter
"""

import json
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from hivemind.config import model_config, lora_config, training_config


class LoRATrainer:
    """
    LoRA 微调训练器

    支持使用 QLoRA 在消费级 GPU 上训练
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        use_4bit: bool = True,
    ):
        self.model_name = model_name or model_config.name
        self.output_dir = output_dir or training_config.output_dir
        self.use_4bit = use_4bit

        self.tokenizer = None
        self.model = None
        self.peft_model = None

    def load_model(self):
        """加载基础模型和 tokenizer"""
        print(f"Loading model: {self.model_name}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=model_config.trust_remote_code,
            padding_side="right",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 量化配置 (QLoRA)
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=training_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, training_config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.float16 if not self.use_4bit else None,
        )

        # 准备模型进行训练
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # 配置 LoRA
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

        return self

    def load_dataset(self, data_path: Path) -> Dataset:
        """
        加载训练数据

        支持 Alpaca 格式:
        [{"instruction": "...", "input": "...", "output": "..."}]

        或 ShareGPT 格式:
        [{"conversations": [{"from": "human", "value": "..."}, {"from": "assistant", "value": "..."}]}]
        """
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检测格式并转换
        if isinstance(data, list) and len(data) > 0:
            if "conversations" in data[0]:
                # ShareGPT 格式
                processed = self._process_sharegpt(data)
            else:
                # Alpaca 格式
                processed = self._process_alpaca(data)
        else:
            raise ValueError("Unsupported data format")

        return Dataset.from_list(processed)

    def _process_alpaca(self, data: list) -> list:
        """处理 Alpaca 格式数据"""
        processed = []
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            full_text = prompt + output

            processed.append({"text": full_text, "prompt": prompt, "response": output})

        return processed

    def _process_sharegpt(self, data: list) -> list:
        """处理 ShareGPT 格式数据"""
        processed = []
        for item in data:
            conversations = item.get("conversations", [])
            text_parts = []

            for conv in conversations:
                role = conv.get("from", "")
                value = conv.get("value", "")

                if role == "human":
                    text_parts.append(f"### Human:\n{value}")
                elif role in ["assistant", "gpt"]:
                    text_parts.append(f"### Assistant:\n{value}")

            if text_parts:
                full_text = "\n\n".join(text_parts)
                processed.append({"text": full_text})

        return processed

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """tokenize 数据集"""

        def tokenize_fn(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=training_config.max_seq_length,
                padding=False,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    def train(self, data_path: Path) -> Path:
        """
        执行训练

        Args:
            data_path: 训练数据路径

        Returns:
            训练后的 adapter 路径
        """
        if self.peft_model is None:
            self.load_model()

        # 加载数据
        print(f"Loading dataset from: {data_path}")
        dataset = self.load_dataset(data_path)
        print(f"Dataset size: {len(dataset)}")

        # Tokenize
        tokenized_dataset = self.tokenize_dataset(dataset)

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            warmup_ratio=training_config.warmup_ratio,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            save_total_limit=2,
            fp16=True,
            optim="paged_adamw_32bit" if self.use_4bit else "adamw_torch",
            report_to="none",
        )

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )

        # 创建 Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # 开始训练
        print("Starting training...")
        trainer.train()

        # 保存 adapter
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Adapter saved to: {self.output_dir}")
        return self.output_dir
