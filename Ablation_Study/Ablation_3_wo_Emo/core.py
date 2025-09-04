# -*- coding: utf-8 -*-
"""
AutoGen风格核心架构模块：实现Actor模型、异步消息传递和事件驱动的智能体架构
基于AutoGen v0.4的设计理念，提供更好的模块化、可扩展性和调试支持
"""
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import threading
import queue


class MessageType(Enum):
    """消息类型枚举"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    REVIEW_REQUEST = "review_request"
    REVIEW_RESPONSE = "review_response"
    REFLECTION_REQUEST = "reflection_request"
    REFLECTION_RESPONSE = "reflection_response"
    SYSTEM_CONTROL = "system_control"
    ERROR = "error"
    TERMINATION = "termination"


@dataclass
class Message:
    """消息类，支持Agent间的异步通信"""
    id: str
    sender: str
    recipient: str
    type: MessageType
    content: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        return cls(**data)


class EventBus:
    """事件总线，负责Agent间的消息路由"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        
    def register_agent(self, agent: 'BaseAgent'):
        """注册Agent到事件总线"""
        self.agents[agent.name] = agent
        agent.set_event_bus(self)
        if self.logger:
            self.logger.log("EVENT_BUS", f"Agent注册: {agent.name}")
    
    def unregister_agent(self, agent_name: str):
        """注销Agent"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            if self.logger:
                self.logger.log("EVENT_BUS", f"Agent注销: {agent_name}")
    
    def send_message(self, message: Message):
        """发送消息"""
        if self.logger:
            self.logger.log("EVENT_BUS", f"消息发送: {message.sender} -> {message.recipient} ({message.type.value})")
        self.message_queue.put(message)
    
    def start(self):
        """启动事件总线"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_messages, daemon=True)
            self.worker_thread.start()
            if self.logger:
                self.logger.log("EVENT_BUS", "事件总线启动")
    
    def stop(self):
        """停止事件总线"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        if self.logger:
            self.logger.log("EVENT_BUS", "事件总线停止")
    
    def _process_messages(self):
        """处理消息队列"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                self._deliver_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                if self.logger:
                    self.logger.log("EVENT_BUS", f"消息处理错误: {e}")
    
    def _deliver_message(self, message: Message):
        """投递消息到目标Agent"""
        recipient = self.agents.get(message.recipient)
        if recipient:
            try:
                recipient.receive_message(message)
            except Exception as e:
                if self.logger:
                    self.logger.log("EVENT_BUS", f"消息投递失败: {message.id}, 错误: {e}")
                # 发送错误消息给发送者
                error_msg = Message(
                    id="",
                    sender="event_bus",
                    recipient=message.sender,
                    type=MessageType.ERROR,
                    content={"error": str(e), "original_message_id": message.id},
                    timestamp=datetime.now().isoformat()
                )
                sender = self.agents.get(message.sender)
                if sender:
                    sender.receive_message(error_msg)
        else:
            if self.logger:
                self.logger.log("EVENT_BUS", f"未找到接收者: {message.recipient}")


class BaseAgent(ABC):
    """Agent基类，实现Actor模型"""
    
    def __init__(self, name: str, logger=None):
        self.name = name
        self.logger = logger
        self.event_bus: Optional[EventBus] = None
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.state: Dict[str, Any] = {}
        self.inbox = queue.Queue()
        self.running = False
        self.worker_thread = None
        
        # 注册默认消息处理器
        self._register_handlers()
    
    def set_event_bus(self, event_bus: EventBus):
        """设置事件总线"""
        self.event_bus = event_bus
    
    def _register_handlers(self):
        """注册消息处理器"""
        self.message_handlers[MessageType.ERROR] = self._handle_error
        self.message_handlers[MessageType.TERMINATION] = self._handle_termination
    
    def start(self):
        """启动Agent"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_inbox, daemon=True)
            self.worker_thread.start()
            if self.logger:
                self.logger.log_agent_work(self.name.upper(), "启动", "Agent已启动")
    
    def stop(self):
        """停止Agent"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        if self.logger:
            self.logger.log_agent_work(self.name.upper(), "停止", "Agent已停止")
    
    def receive_message(self, message: Message):
        """接收消息"""
        self.inbox.put(message)
    
    def _process_inbox(self):
        """处理收件箱"""
        while self.running:
            try:
                message = self.inbox.get(timeout=1)
                self._handle_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                if self.logger:
                    self.logger.log_agent_work(self.name.upper(), "消息处理错误", str(e))
    
    def _handle_message(self, message: Message):
        """处理消息"""
        if self.logger:
            self.logger.log_agent_work(self.name.upper(), "收到消息", f"类型: {message.type.value}, 发送者: {message.sender}")
        
        handler = self.message_handlers.get(message.type)
        if handler:
            try:
                handler(message)
            except Exception as e:
                if self.logger:
                    self.logger.log_agent_work(self.name.upper(), "处理器错误", str(e))
        else:
            if self.logger:
                self.logger.log_agent_work(self.name.upper(), "未知消息类型", message.type.value)
    
    def send_message(self, recipient: str, message_type: MessageType, content: Dict[str, Any], correlation_id: Optional[str] = None):
        """发送消息"""
        if not self.event_bus:
            if self.logger:
                self.logger.log_agent_work(self.name.upper(), "发送失败", "事件总线未设置")
            return
        
        message = Message(
            id="",
            sender=self.name,
            recipient=recipient,
            type=message_type,
            content=content,
            timestamp="",
            correlation_id=correlation_id
        )
        
        self.event_bus.send_message(message)
    
    def _handle_error(self, message: Message):
        """处理错误消息"""
        if self.logger:
            error_info = message.content.get("error", "未知错误")
            self.logger.log_agent_work(self.name.upper(), "收到错误", error_info)
    
    def _handle_termination(self, message: Message):
        """处理终止消息"""
        if self.logger:
            self.logger.log_agent_work(self.name.upper(), "收到终止指令", "开始停止")
        self.stop()
    
    @abstractmethod
    def initialize(self):
        """初始化Agent"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return self.state.copy()
    
    def update_state(self, key: str, value: Any):
        """更新Agent状态"""
        self.state[key] = value
        if self.logger:
            self.logger.log_agent_work(self.name.upper(), "状态更新", f"{key}: {value}")


class ConversationOrchestrator:
    """对话协调器，实现AutoGen风格的conversation protocols"""
    
    def __init__(self, event_bus: EventBus, logger=None):
        self.event_bus = event_bus
        self.logger = logger
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.max_rounds = 7
        self.termination_keywords = ["理解了", "明白了", "谢谢", "结束"]
    
    def start_conversation(self, conversation_id: str, participants: List[str], initial_message: str, problem_content: str) -> str:
        """启动对话"""
        conversation = {
            "id": conversation_id,
            "participants": participants,
            "problem_content": problem_content,
            "history": [],
            "current_round": 0,
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        
        self.conversations[conversation_id] = conversation
        
        if self.logger:
            self.logger.log("ORCHESTRATOR", f"对话启动: {conversation_id}, 参与者: {participants}")
        
        # 发送初始消息给学生Agent
        student_agent = participants[0] if participants else ""
        if student_agent:
            self.event_bus.send_message(Message(
                id="",
                sender="orchestrator",
                recipient=student_agent,
                type=MessageType.TASK_REQUEST,
                content={
                    "conversation_id": conversation_id,
                    "problem_content": problem_content,
                    "instruction": "start_conversation"
                },
                timestamp="",
                correlation_id=conversation_id
            ))
        
        return conversation_id
    
    def should_terminate_conversation(self, conversation_id: str, last_message: str) -> bool:
        """判断是否应该终止对话"""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return True
        
        # 检查轮数限制
        if conversation["current_round"] >= self.max_rounds:
            if self.logger:
                self.logger.log("ORCHESTRATOR", f"对话{conversation_id}达到最大轮数限制")
            return True
        
        # 检查终止关键词
        for keyword in self.termination_keywords:
            if keyword in last_message:
                if self.logger:
                    self.logger.log("ORCHESTRATOR", f"对话{conversation_id}检测到终止关键词: {keyword}")
                return True
        
        return False
    
    def add_message_to_conversation(self, conversation_id: str, sender: str, content: str, message_type: str = "message"):
        """添加消息到对话历史"""
        conversation = self.conversations.get(conversation_id)
        if conversation:
            # 如果是学生发言，增加轮次计数
            if sender == "student":
                conversation["current_round"] += 1
            
            conversation["history"].append({
                "sender": sender,
                "content": content,
                "type": message_type,
                "timestamp": datetime.now().isoformat(),
                "round": conversation["current_round"]
            })
            
            if self.logger:
                self.logger.log("ORCHESTRATOR", f"对话{conversation_id}添加消息: {sender} - 第{conversation['current_round']}轮 - {content[:50]}...")
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """获取对话历史"""
        conversation = self.conversations.get(conversation_id)
        return conversation["history"] if conversation else []
    
    def end_conversation(self, conversation_id: str):
        """结束对话"""
        conversation = self.conversations.get(conversation_id)
        if conversation:
            conversation["status"] = "completed"
            conversation["ended_at"] = datetime.now().isoformat()
            
            if self.logger:
                rounds = conversation["current_round"]
                self.logger.log("ORCHESTRATOR", f"对话{conversation_id}结束，共{rounds}轮")
            
            # 通知所有参与者对话结束
            for participant in conversation["participants"]:
                self.event_bus.send_message(Message(
                    id="",
                    sender="orchestrator",
                    recipient=participant,
                    type=MessageType.SYSTEM_CONTROL,
                    content={
                        "action": "conversation_ended",
                        "conversation_id": conversation_id
                    },
                    timestamp=""
                )) 