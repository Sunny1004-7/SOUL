# -*- coding: utf-8 -*-
"""
Chain-of-Thought (CoT) 教师智能体
实现思维链方法生成教学回复
"""
from typing import Dict, Any, List
from config import METHOD_CONFIGS, BASE_TEACHER_PROMPT


class CoTTeacherAgent:
    """Chain-of-Thought 教师智能体"""
    
    def __init__(self, name: str, llm_manager):
        self.name = name
        self.llm_manager = llm_manager
        
        # CoT特定配置
        self.temperature = METHOD_CONFIGS["CoT"]["temperature"]
        
        print(f"CoT教师智能体初始化完成")
        
        # 对话历史存储
        self.conversation_history = []
        

    def generate_response(self, student_message: str, student_state: Dict[str, Any], 
                        round_number: int) -> str:
        """使用CoT方法生成教师回复"""
        print(f"CoT方法开始生成第{round_number}轮回复")
        
        # 构建基础上下文
        context = self._build_context(student_message, student_state, round_number)
        
        # 使用思维链生成回复
        response = self._generate_cot_response(context, round_number)
        
        print(f"CoT方法生成完成，回复长度: {len(response)}字符")
        return response

    def _build_context(self, student_message: str, student_state: Dict[str, Any], 
                      round_number: int) -> str:
        """构建上下文"""
        context = f"""当前对话轮次：第{round_number}轮
学生消息：{student_message}
学生当前情绪：{student_state.get('current_emotion', '未知')}
学生人格特征：{student_state.get('persona', '未知')}"""
        
        return context

    def _generate_cot_response(self, context: str, round_number: int) -> str:
        """使用思维链生成回复"""
        prompt = f"""{BASE_TEACHER_PROMPT}

{context}

请按照以下思维链步骤来生成教学回复：

步骤1：分析学生当前状态
- 学生的情绪状态如何？
- 学生表达了什么困难？
- 学生需要什么样的帮助？

步骤2：确定教学策略
- 应该采用什么样的苏格拉底式教学方法？
- 如何关注学生的情感需求？
- 如何引导学生主动思考？

步骤3：设计具体回复
- 如何开始对话？
- 应该提出什么问题？
- 如何给予情感支持？

步骤4：生成完整回复
基于以上分析，生成一个专业、有情感支持的教学回复。

请按照这个思维链，逐步分析并生成回复。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=self.temperature)
        
        if not response:
            return self._fallback_response(context)
        
        return response.strip()

    def _fallback_response(self, context: str) -> str:
        """生成备用回复"""
        return "同学，我理解你的困惑。让我们一起来分析这个问题，你觉得应该从哪里开始呢？"
