# -*- coding: utf-8 -*-
"""
对比实验配置文件
定义8种方法的基本配置和对话生成参数
"""

# LLM配置
LLM_CONFIG = {
    "model": "gpt-3.5-turbo",
    "base_url": "https://xh.v1api.cc/v1",
    "api_key": "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm",
    "temperature": 0.7,
    "max_tokens": 1000
}

# 8种方法的基本配置 - 统一配置
METHOD_CONFIGS = {
    "ToT": {
        "temperature": 0.7,
        "max_tokens": 1000
    },
    
    "Self_Consistency": {
        "temperature": 0.7,
        "max_tokens": 1000
    },
    
    "Best_of_N": {
        "temperature": 0.7,
        "max_tokens": 1000
    },
    
    "Zero_shot": {
        "temperature": 0.7,
        "max_tokens": 1000
    },
    
    "ICL": {
        "temperature": 0.7,
        "max_tokens": 1000
    },
    
    "CoT": {
        "temperature": 0.7,
        "max_tokens": 1000
    },
    
    "ICL_CoT": {
        "temperature": 0.7,
        "max_tokens": 1000
    },
    
    "Socratic_Induction": {
        "temperature": 0.7,
        "max_tokens": 1000
    }
}

# 对话生成配置
DIALOGUE_CONFIG = {
    "conversations_per_method": 50,  # 每种方法生成的对话数量
    "student_data_path": "../Emotional_Quantification/data/Student_Record.csv"  # 学生数据路径
}

# 基础教师角色提示词
BASE_TEACHER_PROMPT = """你是一名经验丰富、富有耐心的老师。

你的教学理念：
- 采用苏格拉底式教学方法，通过提问引导学生自己发现答案
- 关注学生的情绪状态，给予适当的情感支持
- 根据学生的知识掌握情况和理解程度调整教学策略
- 使用启发式教学，引导学生主动思考
- 语言温和友善，避免让学生感到压力
- 善于将复杂问题分解为易懂的步骤
- 基于学生的知识薄弱点进行针对性教学

苏格拉底式教学的核心原则：
- 不直接给出答案，而是通过精心设计的问题引导学生思考
- 从学生已知的知识出发，逐步引导到未知领域
- 鼓励学生表达自己的想法，即使想法不完整或错误
- 通过反问和追问帮助学生发现逻辑漏洞
- 让学生通过自己的思考得出结论，增强学习成就感
- 培养学生的批判性思维和独立思考能力"""
