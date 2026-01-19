"""
本地推理模块

加载基础模型 + LoRA adapter 进行对话
"""

from pathlib import Path
from typing import Optional, Generator

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

from hivemind.config import model_config


class HiveMindChat:
    """
    HiveMind 对话接口

    支持加载基础模型和个性化 LoRA adapter
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        adapter_path: Optional[Path] = None,
        use_4bit: bool = True,
    ):
        self.model_name = model_name or model_config.name
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit

        self.tokenizer = None
        self.model = None
        self.conversation_history = []

    def load(self):
        """加载模型"""
        print(f"Loading model: {self.model_name}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=model_config.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 量化配置
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.float16 if not self.use_4bit else None,
        )

        # 加载 LoRA adapter (如果有)
        if self.adapter_path and Path(self.adapter_path).exists():
            print(f"Loading adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()
        print("Model loaded successfully!")

        return self

    def format_prompt(self, message: str, system_prompt: Optional[str] = None) -> str:
        """格式化对话提示"""
        # Qwen2 格式
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # 添加历史对话
        for turn in self.conversation_history:
            prompt_parts.append(f"<|im_start|>user\n{turn['user']}<|im_end|>")
            prompt_parts.append(f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>")

        # 添加当前用户输入
        prompt_parts.append(f"<|im_start|>user\n{message}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def generate(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        生成回复

        Args:
            message: 用户消息
            system_prompt: 系统提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样
            stream: 是否流式输出

        Returns:
            生成的回复 (流式时返回生成器)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = self.format_prompt(message, system_prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if stream:
            return self._stream_generate(inputs, generation_config)
        else:
            return self._generate(inputs, generation_config, message)

    def _generate(self, inputs, generation_config, message: str) -> str:
        """非流式生成"""
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)

        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # 清理结束标记
        response = response.replace("<|im_end|>", "").strip()

        # 保存到历史
        self.conversation_history.append({"user": message, "assistant": response})

        return response

    def _stream_generate(self, inputs, generation_config) -> Generator[str, None, None]:
        """流式生成"""
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_config["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs={**inputs, **generation_config})
        thread.start()

        generated_text = ""
        for text in streamer:
            text = text.replace("<|im_end|>", "")
            generated_text += text
            yield text

        thread.join()

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []

    def chat(self, system_prompt: Optional[str] = None):
        """
        交互式对话

        Args:
            system_prompt: 系统提示
        """
        if self.model is None:
            self.load()

        print("\n" + "=" * 50)
        print("HiveMind Chat")
        print("Type 'exit' to quit, 'clear' to clear history")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "clear":
                    self.clear_history()
                    print("History cleared.")
                    continue

                print("AI: ", end="", flush=True)

                # 流式输出
                for chunk in self.generate(
                    user_input,
                    system_prompt=system_prompt,
                    stream=True,
                ):
                    print(chunk, end="", flush=True)

                print("\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
