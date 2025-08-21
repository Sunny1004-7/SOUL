# -*- coding: utf-8 -*-
"""
学生智能体：用于对比实验的师生对话模拟
基于Teaching_Dialogue中的student_agent.py，简化并适配对比实验系统
"""
import random
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os

# 添加路径以导入必要的模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Emotional_Quantification'))
from persona_loader import STUDENT_PERSONAS


class StudentAgent:
    """学生智能体：用于对比实验的师生对话模拟"""
    
    def __init__(self, llm_manager, problem_content: str, user_id: str = None):
        self.llm_manager = llm_manager
        self.problem_content = problem_content
        self.user_id = user_id
        
        # 学生状态
        self.current_emotion = "困惑"
        self.conversation_id = None
        
        # 对话历史存储
        self.conversation_history = []
        
        # 加载学生历史记录
        self.student_history = []
        self._load_student_history()
        
        # 随机选择学生人格
        self.persona = self._select_random_persona()
        
        print(f"学生智能体初始化完成 - 学生ID: {user_id}, 人格: {self.persona}")

    def _load_student_history(self):
        """加载学生历史习题作答记录"""
        try:
            from student_data_loader import StudentDataLoader
            loader = StudentDataLoader()
            
            if self.user_id is None:
                # 如果没有指定user_id，使用第一个学生
                self.user_id = loader.get_first_student_id()
                if self.user_id is None:
                    print("警告：无法获取学生ID，使用默认设置")
                    return
            
            # 获取学生历史记录（排除最后一条）
            self.student_history = loader.get_student_history_except_last(self.user_id)
            print(f"成功加载学生历史记录，题目数: {len(self.student_history)}")
                
        except Exception as e:
            print(f"历史记录加载失败: {e}")
            self.student_history = []

    def _select_random_persona(self) -> str:
        """随机选择学生人格"""
        try:
            persona_keys = list(STUDENT_PERSONAS.keys())
            selected_key = random.choice(persona_keys)
            return STUDENT_PERSONAS[selected_key]
        except:
            return "谨慎型学习者"

    def generate_first_message(self) -> str:
        """生成第一轮发言"""
        print("开始生成第一轮发言")
        
        # 构建历史记录上下文
        history_context = self._build_history_context()
        
        messages = [
            {
                "role": "system", 
                "content": f"你是一名{self.persona}，正在与老师进行教学对话。请自然地扮演学生角色。"
            },
            {
                "role": "user",
                "content": f"""你需要解决这个题目：{self.problem_content}

{history_context}

请生成你的第一轮发言，要求：
1. 表达你对这个题目的困惑
2. 说明你希望得到什么样的帮助
3. 语言要自然，符合中学生特点
4. 体现你的性格特点
"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.8)
        
        # 处理LLM调用失败的情况
        if not response:
            response = "老师，这道数学题我完全不知道从哪里开始，感觉很困惑。"
            print("LLM调用失败，使用默认回复")
        
        # 将第一轮发言添加到对话历史
        self._add_to_conversation_history("student", response)
        
        print(f"初始发言生成完成，长度: {len(response)}字符")
        return response

    def generate_response(self, teacher_message: str, round_number: int) -> str:
        """基于教师回复生成学生回复"""
        print(f"开始生成第{round_number}轮回复")
        
        # 1. 先用LLM分析学生当前情绪
        new_emotion = self._analyze_emotion(teacher_message, round_number)
        self.current_emotion = new_emotion
        print(f"情绪分析完成，新情绪: {self.current_emotion}")
        
        # 2. 再用新情绪生成学生回复
        student_response = self._generate_student_response(teacher_message, round_number)
        
        if student_response:
            # 将学生回复添加到对话历史
            self._add_to_conversation_history("student", student_response)
            print(f"第{round_number}轮回复生成完成，长度: {len(student_response)}字符")
        
        return student_response

    def _build_history_context(self) -> str:
        """构建历史记录上下文"""
        if not self.student_history:
            return "这是你第一次做题，没有历史记录。"
        
        total_problems = len(self.student_history)
        
        context = f"""你的历史做题情况：
- 总共做过 {total_problems} 道题
"""
        
        # 显示最近几道题的基本信息
        if self.student_history:
            context += "\n最近做题情况：\n"
            for i, record in enumerate(self.student_history[-5:], 1):  # 只显示最近5道题
                content = record.get('content', '')[:50] + '...' if len(record.get('content', '')) > 50 else record.get('content', '')
                context += f"{i}. {content}\n"
        
        return context

    def _analyze_emotion(self, teacher_message: str, round_number: int) -> str:
        """用LLM分析学生当前情绪"""
        # 获取学生上一轮发言
        last_student_message = self._get_last_student_message()
        
        prompt = f"""你是一名{self.persona}，正在与老师进行教学对话。

请根据你上一轮的发言和老师最新的回复，判断你现在的情绪状态。

- 只输出一个词，表示情绪，例如：困惑、理解、满意、紧张、开心、无聊、激动、失落等。
- 不要输出解释或其他内容。

学生上一轮发言：{last_student_message}
老师最新回复：{teacher_message}

你的当前情绪是："""
        
        response = self.llm_manager.call_llm([
            {"role": "system", "content": prompt}
        ], temperature=0.3)
        
        return response.strip().split()[0] if response else self.current_emotion

    def _generate_student_response(self, teacher_message: str, round_number: int) -> str:
        """基于角色扮演和当前情绪生成学生回复"""
        # 获取对话历史
        conversation_history = self._get_conversation_history()
        
        # 构建对话上下文
        context = ""
        if conversation_history:
            context = "对话历史：\n"
            for msg in conversation_history[-6:]:  # 只取最近6轮对话
                sender = "老师" if msg.get("sender") == "teacher" else "学生"
                content = msg.get("content", "")
                context += f"{sender}：{content}\n"
        
        # 角色扮演prompt - 基于当前情绪状态和对话历史
        messages = [
            {"role": "system", "content": f"你是一名学生，当前情绪：{self.current_emotion}。请基于你的情绪状态、对话历史和老师的回复，自然地生成学生回复。"},
            {"role": "user", "content": f"{context}老师最新回复：{teacher_message}\n当前情绪：{self.current_emotion}\n请以学生身份自然回复老师。"}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.8)
        
        # 处理LLM调用失败的情况
        if not response:
            response = "老师，我还在思考您刚才说的话，让我再想想..."
            print("LLM调用失败，使用默认回复")
        
        return response

    def _get_last_student_message(self) -> str:
        """获取学生上一轮发言"""
        # 从本地对话历史中获取学生上一轮发言
        for message in reversed(self.conversation_history):
            if message.get("sender") == "student":
                return message.get("content", "")
        return "初始问题"

    def _get_conversation_history(self) -> list:
        """获取对话历史"""
        return self.conversation_history

    def _add_to_conversation_history(self, sender: str, content: str, message_type: str = "message"):
        """添加消息到对话历史"""
        message = {
            "sender": sender,
            "content": content,
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "round": len(self.conversation_history) // 2 + 1  # 每轮包含学生和老师两条消息
        }
        self.conversation_history.append(message)

    def get_student_state(self) -> Dict[str, Any]:
        """获取学生当前状态"""
        return {
            "current_emotion": self.current_emotion,
            "persona": self.persona,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取完整对话历史"""
        return self.conversation_history.copy()
