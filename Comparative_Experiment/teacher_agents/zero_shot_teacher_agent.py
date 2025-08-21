# -*- coding: utf-8 -*-
"""
Zero-shot Learning 教师智能体
实现零样本学习方法生成教学回复
"""
from typing import Dict, Any, List
from config import METHOD_CONFIGS, BASE_TEACHER_PROMPT


class ZeroShotTeacherAgent:
    """Zero-shot Learning 教师智能体"""
    
    def __init__(self, name: str, llm_manager):
        self.name = name
        self.llm_manager = llm_manager
        
        # Zero-shot特定配置
        self.temperature = METHOD_CONFIGS["Zero_shot"]["temperature"]
        
        # 对话历史存储
        self.conversation_history = []
        
        print(f"Zero-shot教师智能体初始化完成")

    def generate_response(self, student_message: str, student_state: Dict[str, Any], 
                        round_number: int) -> str:
        """使用Zero-shot方法生成教师回复"""
        print(f"Zero-shot方法开始生成第{round_number}轮回复")
        
        # 构建基础上下文
        context = self._build_context(student_message, student_state, round_number)
        
        # 直接生成回复
        response = self._generate_zero_shot_response(context, round_number)
        
        # 记录对话历史
        self._add_to_conversation_history("teacher", response, round_number)
        
        print(f"Zero-shot方法生成完成，回复长度: {len(response)}字符")
        return response

    def _build_context(self, student_message: str, student_state: Dict[str, Any], 
                      round_number: int) -> str:
        """构建上下文"""
        context = f"""当前对话轮次：第{round_number}轮
学生消息：{student_message}
学生当前情绪：{student_state.get('current_emotion', '未知')}
学生人格特征：{student_state.get('persona', '未知')}"""
        
        return context

    def _generate_zero_shot_response(self, context: str, round_number: int) -> str:
        """使用零样本方法生成回复"""
        prompt = f"""{BASE_TEACHER_PROMPT}

{context}

请基于你的教学经验和苏格拉底式教学原则，生成一个专业、有情感支持的教学回复。

要求：
1. 体现苏格拉底式教学方法
2. 关注学生的情绪状态
3. 语言温和友善
4. 引导学生思考而不是直接给出答案
5. 长度适中（100-200字）"""

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

    def _add_to_conversation_history(self, sender: str, content: str, round_number: int):
        """添加消息到对话历史"""
        message = {
            "sender": sender,
            "content": content,
            "type": "message",
            "round": round_number
        }
        self.conversation_history.append(message)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.conversation_history.copy()

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
