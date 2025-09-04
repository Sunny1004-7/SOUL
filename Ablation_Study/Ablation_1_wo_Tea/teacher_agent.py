# -*- coding: utf-8 -*-
"""
消融实验1: w/o.Tea 变体模型
移除教师智能体的复杂ICECoT思维链逻辑，仅保留对话监管智能体与反思智能体
通过直接提示生成响应，不进行复杂的情绪分析、意图推断等步骤
"""
from core import BaseAgent, MessageType, Message
from typing import Dict, Any, Optional
import json


class TeacherAgent(BaseAgent):
    """简化版教师智能体 - 移除ICECoT思维链"""
    
    def __init__(self, name: str, llm_manager, logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        self.conversation_id = None
        self.conversation_history = []
        
        # 简化的教师角色基础prompt
        self.base_prompt = """你是一名经验丰富、富有耐心的老师。

你的教学理念：
- 采用苏格拉底式教学方法，通过提问引导学生自己发现答案
- 语言温和友善，避免让学生感到压力
- 善于将复杂问题分解为易懂的步骤

请直接根据学生的问题生成教学回复，不需要复杂的分析过程。"""

        # 注册消息处理器
        self._register_teacher_handlers()
        
        if self.logger:
            self.logger.log_agent_work("TEACHER", "初始化完成", "简化版教师智能体已就绪")

    def initialize(self):
        """初始化教师智能体"""
        self.update_state("ready", True)
        self.update_state("simplified_mode", True)

    def _register_teacher_handlers(self):
        """注册教师特定的消息处理器"""
        self.message_handlers[MessageType.TASK_REQUEST] = self._handle_student_message
        self.message_handlers[MessageType.REVIEW_RESPONSE] = self._handle_monitor_feedback
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control

    def _handle_student_message(self, message: Message):
        """处理学生消息 - 简化版本"""
        content = message.content
        self.conversation_id = content.get("conversation_id")
        student_message = content.get("student_message", "")
        round_number = content.get("round_number", 1)
        student_state = content.get("student_state", {})
        
        if self.logger:
            self.logger.log_agent_work("TEACHER", f"收到学生消息", f"第{round_number}轮: {student_message[:50]}...")
        
        # 更新对话历史
        self.conversation_history.append({
            "role": "student",
            "content": student_message,
            "round": round_number,
            "state": student_state
        })
        
        # 直接生成教学回复（无复杂分析）
        teacher_response = self._generate_simple_teaching_response(student_message, round_number)
        
        if teacher_response:
            # 发送给监控智能体审核
            self.send_message(
                recipient="monitor",
                message_type=MessageType.REVIEW_REQUEST,
                content={
                    "conversation_id": self.conversation_id,
                    "teacher_response": teacher_response,
                    "student_message": student_message,
                    "round_number": round_number,
                    "conversation_history": self.conversation_history.copy()
                },
                correlation_id=self.conversation_id
            )

    def _handle_monitor_feedback(self, message: Message):
        """处理监控反馈"""
        content = message.content
        approved = content.get("approved", False)
        teacher_response = content.get("teacher_response", "")
        round_number = content.get("round_number", 1)
        
        if approved:
            # 审核通过，发送回复给学生
            self._send_approved_response(teacher_response, round_number)
        else:
            # 审核未通过，重新生成
            feedback = content.get("feedback", "")
            student_message = content.get("student_message", "")
            self._regenerate_response(student_message, feedback, round_number)

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        action = content.get("action")
        
        if action == "conversation_ended":
            # 清理对话状态
            self.conversation_id = None
            self.conversation_history = []
            if self.logger:
                self.logger.log_agent_work("TEACHER", "对话结束", "清理状态完成")

    def _generate_simple_teaching_response(self, student_message: str, round_number: int) -> str:
        """生成简化的教学回复（无复杂分析）"""
        if self.logger:
            self.logger.log_agent_work("TEACHER", "开始生成简化回复", f"第{round_number}轮")
        
        messages = [
            {
                "role": "system",
                "content": f"""{self.base_prompt}

请根据学生的发言直接生成教学回复。回复要求：
1. 采用苏格拉底式教学方法
2. 语言温和友善
3. 长度适中，直接针对学生问题回答
4. 不直接给出答案，而是通过问题引导学生思考"""
            },
            {
                "role": "user",
                "content": f"""学生发言：{student_message}

这是第{round_number}轮对话。请生成教学回复。"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.7, max_tokens=300)
        result = response if response else "我理解你的困惑，让我们一起来解决这个问题。"
        
        if self.logger:
            if response:
                self.logger.log_agent_work("TEACHER", "简化回复生成成功", f"回复长度: {len(result)}字符")
            else:
                self.logger.log_agent_work("TEACHER", "简化回复生成失败", "使用默认回复")
        
        return result

    def _send_approved_response(self, teacher_response: str, round_number: int):
        """发送审核通过的回复给学生"""
        # 添加到对话历史
        self.conversation_history.append({
            "role": "teacher",
            "content": teacher_response,
            "round": round_number
        })
        
        # 发送给学生
        self.send_message(
            recipient="student",
            message_type=MessageType.TASK_RESPONSE,
            content={
                "conversation_id": self.conversation_id,
                "teacher_response": teacher_response,
                "round_number": round_number
            },
            correlation_id=self.conversation_id
        )
        
        # 通知对话协调器记录消息
        self.send_message(
            recipient="orchestrator",
            message_type=MessageType.SYSTEM_CONTROL,
            content={
                "action": "add_message",
                "conversation_id": self.conversation_id,
                "sender": "teacher",
                "content": teacher_response,
                "message_type": "message"
            }
        )
        
        if self.logger:
            self.logger.log_agent_work("TEACHER", "回复已发送", f"第{round_number}轮，长度: {len(teacher_response)}字符")

    def _regenerate_response(self, student_message: str, feedback: str, round_number: int):
        """根据监控反馈重新生成回复"""
        if self.logger:
            self.logger.log_agent_work("TEACHER", "重新生成回复", f"反馈: {feedback}")
        
        messages = [
            {
                "role": "system",
                "content": f"""{self.base_prompt}

你刚才的回复被监控系统发现问题，需要重新生成。

监控反馈：{feedback}

请注意：
1. 避免之前回复中的问题
2. 确保语调温和友善
3. 确保内容与学生问题相关
4. 保持专业性和准确性"""
            },
            {
                "role": "user",
                "content": f"学生发言：{student_message}\n\n请重新生成一个更好的教学回复。"
            }
        ]
        
        new_response = self.llm_manager.call_llm(messages, temperature=0.8, max_tokens=300)
        
        if new_response:
            # 重新发送给监控智能体
            self.send_message(
                recipient="monitor",
                message_type=MessageType.REVIEW_REQUEST,
                content={
                    "conversation_id": self.conversation_id,
                    "teacher_response": new_response,
                    "student_message": student_message,
                    "round_number": round_number,
                    "conversation_history": self.conversation_history.copy(),
                    "is_regenerated": True
                },
                correlation_id=self.conversation_id
            )
            
            if self.logger:
                self.logger.log_agent_work("TEACHER", "重新生成完成", f"新回复长度: {len(new_response)}字符")
        else:
            if self.logger:
                self.logger.log_agent_work("TEACHER", "重新生成失败", "使用默认回复")
            self._send_approved_response("抱歉，让我重新为你解释一下。", round_number)