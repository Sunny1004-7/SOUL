# -*- coding: utf-8 -*-
"""
基于AutoGen架构的学生智能体：实现事件驱动的学生行为模拟
保留原有的学生角色特征和行为逻辑，但使用异步消息传递和Actor模型
"""
from core import BaseAgent, MessageType, Message
from typing import Dict, Any, Optional
from datetime import datetime


class StudentAgent(BaseAgent):
    """基于AutoGen架构的学生智能体"""
    
    def __init__(self, name: str, llm_manager, problem_content: str, 
                 persona: str = "耐心但容易紧张的中学生", 
                 initial_emotion: str = "困惑", 
                 logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        self.problem_content = problem_content
        self.persona = persona
        self.initial_emotion = initial_emotion
        
        # 学生状态
        self.current_emotion = initial_emotion
        self.conversation_id = None
        
        # 对话历史存储
        self.conversation_history = []
        
        # 导入对话分析器
        from conversation_analyzer import ConversationAnalyzer
        self.conversation_analyzer = ConversationAnalyzer(llm_manager, logger)
        
        # 注册消息处理器
        self._register_student_handlers()
        
        if self.logger:
            self.logger.log_agent_work("STUDENT", "初始化完成", f"学生角色: {persona}")

    def initialize(self):
        """初始化学生智能体"""
        self.update_state("ready", True)
        self.update_state("conversation_started", False)

    def _register_student_handlers(self):
        """注册学生特定的消息处理器"""
        self.message_handlers[MessageType.TASK_REQUEST] = self._handle_task_request
        self.message_handlers[MessageType.TASK_RESPONSE] = self._handle_teacher_response
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control

    def _handle_task_request(self, message: Message):
        """处理任务请求"""
        content = message.content
        
        # 设置对话ID
        if content.get("conversation_id"):
            self.conversation_id = content.get("conversation_id")
            if self.logger:
                self.logger.log_agent_work("STUDENT", "设置对话ID", f"ID: {self.conversation_id}")
        
        if content.get("instruction") == "start_conversation":
            # 生成第一轮学生发言
            student_message = self._generate_first_message()
            if student_message:
                # 发送给教师智能体
                self.send_message(
                    recipient="teacher",
                    message_type=MessageType.TASK_REQUEST,
                    content={
                        "conversation_id": self.conversation_id,
                        "student_message": student_message,
                        "round_number": 1,
                        "student_state": self.get_student_state()
                    },
                    correlation_id=self.conversation_id
                )
                
                if self.logger:
                    self.logger.log_agent_work("STUDENT", "第一轮发言", f"内容: {student_message[:50]}...")

    def _handle_teacher_response(self, message: Message):
        """处理教师回复"""
        content = message.content
        teacher_message = content.get("teacher_message", "")
        round_number = content.get("round_number", 1)
        
        if self.logger:
            self.logger.log_agent_work("STUDENT", f"收到教师回复", f"第{round_number}轮: {teacher_message[:50]}...")
        
        # 将老师回复添加到对话历史
        self._add_to_conversation_history("teacher", teacher_message)
        
        # 1. 先用LLM分析学生当前情绪
        new_emotion = self._analyze_emotion(teacher_message, round_number)
        self.current_emotion = new_emotion
        if self.logger:
            self.logger.log_agent_work("STUDENT", f"情绪分析", f"第{round_number}轮, 新情绪: {self.current_emotion}")
        
        # 2. 再用新情绪生成学生回复
        student_response = self._generate_student_response(teacher_message, round_number)
        
        if student_response:
            # 将学生回复添加到对话历史
            self._add_to_conversation_history("student", student_response)
            
            # 使用对话分析器智能判断是否应该结束对话
            conversation_history = self._get_conversation_history()
            analysis_result = self.conversation_analyzer.analyze_conversation_end(
                student_response, conversation_history, round_number + 1, self.problem_content
            )
            should_end = analysis_result.get("should_end", False)
            if should_end:
                # 通知协调器对话结束
                self.send_message(
                    recipient="orchestrator",
                    message_type=MessageType.SYSTEM_CONTROL,
                    content={
                        "action": "end_conversation",
                        "conversation_id": self.conversation_id,
                        "final_message": student_response,
                        "reason": analysis_result.get("reason", "student_understood"),
                        "analysis_result": analysis_result
                    },
                    correlation_id=self.conversation_id
                )
            else:
                # 继续对话，发送给教师
                self.send_message(
                    recipient="teacher",
                    message_type=MessageType.TASK_REQUEST,
                    content={
                        "conversation_id": self.conversation_id,
                        "student_message": student_response,
                        "round_number": round_number + 1,
                        "student_state": self.get_student_state()
                    },
                    correlation_id=self.conversation_id
                )

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        action = content.get("action")
        
        if action == "conversation_ended":
            if self.logger:
                self.logger.log_agent_work("STUDENT", "对话结束", f"对话ID: {content.get('conversation_id')}")
            self.conversation_id = None

    def _get_conversation_history(self) -> list:
        """获取对话历史"""
        # 直接返回本地存储的对话历史
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
        
        if self.logger:
            self.logger.log_agent_work("STUDENT", "添加对话历史", f"{sender}: {content[:30]}...")

    def _generate_first_message(self) -> str:
        """生成第一轮发言"""
        if self.logger:
            self.logger.log_agent_work("STUDENT", "开始生成第一轮发言", "基于题目内容生成初始问题")
        
        messages = [
            {
                "role": "system", 
                "content": f"你是一名{self.persona}，正在与老师进行教学对话。请自然地扮演学生角色。"
            },
            {
                "role": "user",
                "content": f"""你需要解决这个题目：{self.problem_content}

请生成你的第一轮发言，要求：
1. 表达你对这个题目的困惑
2. 说明你希望得到什么样的帮助
3. 语言要自然，符合中学生特点
4. 体现你的性格特点"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.8)
        
        if self.logger:
            self.logger.log_agent_work("STUDENT", "初始发言生成", f"长度: {len(response)}字符")
        
        # 将第一轮发言添加到对话历史
        self._add_to_conversation_history("student", response)
        
        # 通知对话协调器记录消息
        if self.event_bus:
            from core import MessageType
            self.send_message(
                recipient="orchestrator",
                message_type=MessageType.SYSTEM_CONTROL,
                content={
                    "action": "add_message",
                    "conversation_id": self.conversation_id,
                    "sender": "student",
                    "content": response,
                    "message_type": "message"
                }
            )
        
        return response

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
        """基于角色扮演和当前情绪生成学生回复（包含对话历史）"""
        if self.logger:
            self.logger.log_agent_work("STUDENT", f"开始生成第{round_number}轮回复", f"基于教师回复: {teacher_message[:50]}... 当前情绪: {self.current_emotion}")
        
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
        if self.logger:
            self.logger.log_agent_work("STUDENT", f"第{round_number}轮回复生成", f"长度: {len(response)}字符, 新情绪: {self.current_emotion}")
        # 通知对话协调器记录消息
        if self.event_bus:
            from core import MessageType
            self.send_message(
                recipient="orchestrator",
                message_type=MessageType.SYSTEM_CONTROL,
                content={
                    "action": "add_message",
                    "conversation_id": self.conversation_id,
                    "sender": "student",
                    "content": response,
                    "message_type": "message"
                }
            )
        return response

    def _get_last_student_message(self) -> str:
        """获取学生上一轮发言"""
        # 从本地对话历史中获取学生上一轮发言
        for message in reversed(self.conversation_history):
            if message.get("sender") == "student":
                return message.get("content", "")
        return "初始问题"

    def get_student_state(self) -> Dict[str, Any]:
        """获取学生当前状态"""
        return {
            "current_emotion": self.current_emotion,
            "persona": self.persona,
            "conversation_id": self.conversation_id
        } 