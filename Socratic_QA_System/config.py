# -*- coding: utf-8 -*-
"""
智能苏格拉底教学问答系统配置文件
"""
import os
from typing import Dict, Any

class Config:
    """系统配置类"""
    
    # LLM API配置
    LLM_CONFIG = {
        "model": "gpt-4o",  # 使用指定的模型
        "base_url": "https://xh.v1api.cc/v1",  # 使用指定的API地址
        "api_key": "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm",  # 使用指定的API密钥
        "timeout": 120,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    # 系统配置
    SYSTEM_CONFIG = {
        "max_conversation_rounds": 50,  # 最大对话轮数
        "conversation_timeout": 300,    # 对话超时时间（秒）
        "enable_monitoring": True,      # 是否启用监控智能体
        "enable_knowledge_analysis": True,  # 是否启用知识状态分析
        "response_delay": 1.0,         # 回复延迟时间（秒）
    }
    
    # 教学配置
    TEACHING_CONFIG = {
        "socratic_style_weight": 0.3,   # 苏格拉底式教学风格权重
        "emotional_support_weight": 0.2, # 情感支持权重
        "clarity_weight": 0.2,          # 语言清晰度权重
        "appropriateness_weight": 0.15,  # 回复适当性权重
        "engagement_weight": 0.15,      # 学生参与度权重
        "min_quality_score": 7.0,       # 最低质量分数
    }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """获取LLM配置"""
        return cls.LLM_CONFIG.copy()
    
    @classmethod
    def get_system_config(cls) -> Dict[str, Any]:
        """获取系统配置"""
        return cls.SYSTEM_CONFIG.copy()
    
    @classmethod
    def get_teaching_config(cls) -> Dict[str, Any]:
        """获取教学配置"""
        return cls.TEACHING_CONFIG.copy()
    
    @classmethod
    def update_llm_config(cls, **kwargs):
        """更新LLM配置"""
        cls.LLM_CONFIG.update(kwargs)
    
    @classmethod
    def update_system_config(cls, **kwargs):
        """更新系统配置"""
        cls.SYSTEM_CONFIG.update(kwargs)
    
    @classmethod
    def update_teaching_config(cls, **kwargs):
        """更新教学配置"""
        cls.TEACHING_CONFIG.update(kwargs)
    
    @classmethod
    def load_from_env(cls):
        """从环境变量加载配置"""
        # LLM配置
        if os.getenv("LLM_API_KEY"):
            cls.LLM_CONFIG["api_key"] = os.getenv("LLM_API_KEY")
        if os.getenv("LLM_BASE_URL"):
            cls.LLM_CONFIG["base_url"] = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_MODEL"):
            cls.LLM_CONFIG["model"] = os.getenv("LLM_MODEL")
        
        # 系统配置
        if os.getenv("MAX_CONVERSATION_ROUNDS"):
            cls.SYSTEM_CONFIG["max_conversation_rounds"] = int(os.getenv("MAX_CONVERSATION_ROUNDS"))
        if os.getenv("ENABLE_MONITORING"):
            cls.SYSTEM_CONFIG["enable_monitoring"] = os.getenv("ENABLE_MONITORING").lower() == "true"
        if os.getenv("ENABLE_KNOWLEDGE_ANALYSIS"):
            cls.SYSTEM_CONFIG["enable_knowledge_analysis"] = os.getenv("ENABLE_KNOWLEDGE_ANALYSIS").lower() == "true"
