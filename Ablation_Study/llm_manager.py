# -*- coding: utf-8 -*-
"""
简单的LLM管理器
提供基本的API调用功能
"""
import requests
import time
from typing import List, Dict, Any, Optional


# LLM配置
LLM_CONFIG = {
    "model": "gpt-4o",
    "base_url": "https://xh.v1api.cc/v1",
    "api_key": "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm",
    "temperature": 0.1,
    "max_tokens": 1000
}


class SimpleLLMManager:
    """简单的LLM管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or LLM_CONFIG
    
    def call_llm(self, messages: List[Dict[str, str]], 
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> Optional[str]:
        """调用LLM API"""
        try:
            temp = temperature if temperature is not None else self.config["temperature"]
            max_tok = max_tokens if max_tokens is not None else self.config["max_tokens"]
            
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.config["model"],
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok
            }
            
            response = requests.post(
                f"{self.config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result_data = response.json()
                return result_data["choices"][0]["message"]["content"].strip()
            else:
                print(f"LLM API调用失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"LLM调用异常: {e}")
            return None
