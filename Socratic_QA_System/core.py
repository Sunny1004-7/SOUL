# -*- coding: utf-8 -*-
"""
核心架构组件：消息传递、事件总线、智能体基类等
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
import threading
import time
import requests
import json
from datetime import datetime


class MessageType(Enum):
    """消息类型枚举"""
    TASK_REQUEST = "task_request"           # 任务请求
    TASK_RESPONSE = "task_response"         # 任务响应
    REVIEW_REQUEST = "review_request"       # 审查请求
    REVIEW_RESPONSE = "review_response"     # 审查响应
    ANALYSIS_REQUEST = "analysis_request"   # 分析请求
    DATA_REQUEST = "data_request"           # 数据请求
    SYSTEM_CONTROL = "system_control"       # 系统控制
    HEARTBEAT = "heartbeat"                 # 心跳消息


@dataclass
class Message:
    """消息数据结构"""
    id: str
    sender: str
    recipient: str
    type: MessageType
    content: Dict[str, Any]
    timestamp: str


class EventBus:
    """事件总线：负责智能体间的消息路由"""
    
    def __init__(self, logger=None):
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.message_queue: List[Message] = []
        self.running = False
        self.logger = logger
        self.lock = threading.Lock()
        
    def register_agent(self, agent: 'BaseAgent'):
        """注册智能体"""
        self.agents[agent.name] = agent
        if self.logger:
            self.logger.log("EVENT_BUS", f"智能体 {agent.name} 已注册")
    
    def unregister_agent(self, agent_name: str):
        """注销智能体"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            if self.logger:
                self.logger.log("EVENT_BUS", f"智能体 {agent_name} 已注销")
    
    def send_message(self, message: Message):
        """发送消息"""
        with self.lock:
            self.message_queue.append(message)
            if self.logger:
                self.logger.log("EVENT_BUS", f"消息已加入队列: {message.type.value} -> {message.recipient}")
    
    def _process_messages(self):
        """处理消息队列"""
        while self.running:
            with self.lock:
                if self.message_queue:
                    message = self.message_queue.pop(0)
                else:
                    message = None
            
            if message:
                self._route_message(message)
            
            time.sleep(0.1)  # 避免过度占用CPU
    
    def _route_message(self, message: Message):
        """路由消息到目标智能体"""
        recipient = message.recipient
        
        if recipient in self.agents:
            try:
                self.agents[recipient].receive_message(message)
                if self.logger:
                    self.logger.log("EVENT_BUS", f"消息已路由到 {recipient}")
            except Exception as e:
                if self.logger:
                    self.logger.log_error("EVENT_BUS", f"消息路由失败: {e}", f"目标: {recipient}")
        else:
            if self.logger:
                self.logger.log("EVENT_BUS", f"目标智能体不存在: {recipient}")
    
    def start(self):
        """启动事件总线"""
        self.running = True
        self.message_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.message_thread.start()
        if self.logger:
            self.logger.log("EVENT_BUS", "事件总线已启动")
    
    def stop(self):
        """停止事件总线"""
        self.running = False
        if hasattr(self, 'message_thread'):
            self.message_thread.join(timeout=1)
        if self.logger:
            self.logger.log("EVENT_BUS", "事件总线已停止")


class BaseAgent:
    """智能体基类"""
    
    def __init__(self, name: str, logger=None):
        self.name = name
        self.logger = logger
        self.running = False
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.state: Dict[str, Any] = {}
        self.message_thread = None
        
        # 注册默认消息处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认消息处理器"""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control
    
    def initialize(self):
        """初始化智能体"""
        self.update_state("initialized", True)
        if self.logger:
            self.logger.log_agent_work(self.name, "初始化完成", "")
    
    def start(self):
        """启动智能体"""
        self.running = True
        self.message_thread = threading.Thread(target=self._message_loop, daemon=True)
        self.message_thread.start()
        if self.logger:
            self.logger.log_agent_work(self.name, "已启动", "")
    
    def stop(self):
        """停止智能体"""
        self.running = False
        if self.message_thread:
            self.message_thread.join(timeout=1)
        if self.logger:
            self.logger.log_agent_work(self.name, "已停止", "")
    
    def _message_loop(self):
        """消息处理循环"""
        while self.running:
            # 子类应该实现具体的消息处理逻辑
            time.sleep(0.1)
    
    def receive_message(self, message: Message):
        """接收消息"""
        if message.type in self.message_handlers:
            try:
                self.message_handlers[message.type](message)
                if self.logger:
                    self.logger.log_agent_work(self.name, "消息处理完成", f"类型: {message.type.value}")
            except Exception as e:
                if self.logger:
                    self.logger.log_error(self.name, f"消息处理失败: {e}", f"消息类型: {message.type.value}")
        else:
            if self.logger:
                self.logger.log(self.name, f"未找到消息处理器: {message.type.value}")
    
    def send_message(self, recipient: str, message_type: MessageType, content: Dict[str, Any]):
        """发送消息"""
        message = Message(
            id=str(time.time()),
            sender=self.name,
            recipient=recipient,
            type=message_type,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        # 这里需要访问EventBus，子类应该重写此方法
        if self.logger:
            self.logger.log_agent_work(self.name, "消息发送", f"目标: {recipient}, 类型: {message_type.value}")
    
    def update_state(self, key: str, value: Any):
        """更新智能体状态"""
        self.state[key] = value
    
    def get_state(self, key: str, default=None):
        """获取智能体状态"""
        return self.state.get(key, default)
    
    def _handle_heartbeat(self, message: Message):
        """处理心跳消息"""
        # 默认心跳处理逻辑
        pass
    
    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        # 默认系统控制处理逻辑
        pass


class ConversationOrchestrator:
    """对话协调器：管理对话状态和流程"""
    
    def __init__(self, event_bus: EventBus, logger=None):
        self.event_bus = event_bus
        self.logger = logger
        self.conversations: Dict[str, Dict[str, Any]] = {}
    
    def start_conversation(self, conversation_id: str, initial_data: Dict[str, Any] = None):
        """开始新对话"""
        self.conversations[conversation_id] = {
            "start_time": datetime.now().isoformat(),
            "rounds": 0,
            "status": "active",
            "data": initial_data or {}
        }
        if self.logger:
            self.logger.log("ORCHESTRATOR", f"对话开始: {conversation_id}")
    
    def add_round(self, conversation_id: str, round_data: Dict[str, Any]):
        """添加对话轮次"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["rounds"] += 1
            self.conversations[conversation_id]["data"][f"round_{self.conversations[conversation_id]['rounds']}"] = round_data
            if self.logger:
                self.logger.log("ORCHESTRATOR", f"对话轮次添加: {conversation_id} - 第{self.conversations[conversation_id]['rounds']}轮")
    
    def end_conversation(self, conversation_id: str):
        """结束对话"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["status"] = "ended"
            self.conversations[conversation_id]["end_time"] = datetime.now().isoformat()
            if self.logger:
                self.logger.log("ORCHESTRATOR", f"对话结束: {conversation_id}")
    
    def get_conversation_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """获取对话信息"""
        return self.conversations.get(conversation_id)


class LLMManager:
    """统一的LLM管理器，避免重复实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def call_llm(self, messages, temperature=None, max_tokens=None):
        """调用LLM API"""
        # 使用配置中的参数，如果没有提供则使用默认值
        temp = temperature if temperature is not None else self.config.get("temperature", 0.7)
        max_t = max_tokens if max_tokens is not None else self.config.get("max_tokens", 1000)
        
        # 构建完整的API端点URL
        base_url = self.config.get("base_url", "https://api.openai.com/v1")
        # 如果base_url已经包含完整的端点，直接使用；否则添加默认端点
        if base_url.endswith("/chat/completions"):
            api_url = base_url
        elif base_url.endswith("/v1"):
            api_url = f"{base_url}/chat/completions"
        else:
            api_url = f"{base_url}/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.get('api_key', '')}"
        }
        
        data = {
            "model": self.config.get("model", "gpt-3.5-turbo"),
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_t
        }
        
        print(f"正在调用LLM API: {api_url}")
        print(f"使用模型: {data['model']}")
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"LLM API调用失败: {response.status_code} - {response.text}"
                print(f"错误详情: {error_msg}")
                print(f"请求URL: {api_url}")
                print(f"请求头: {headers}")
                print(f"请求数据: {data}")
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求异常: {e}"
            print(f"网络错误: {error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"LLM调用异常: {e}"
            print(f"其他错误: {error_msg}")
            raise Exception(error_msg)
    



class Logger:
    """统一的日志记录器，提供完整的日志功能"""
    
    def __init__(self, enable_file_logging=False, log_file_path=None):
        self.enable_file_logging = enable_file_logging
        self.log_file_path = log_file_path or "system.log"
    
    def log(self, component, message):
        """记录一般日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{component}] {message}"
        print(log_entry)
        
        if self.enable_file_logging:
            self._write_to_file(log_entry)
    
    def log_agent_work(self, agent_name, action, details):
        """记录智能体工作日志"""
        self.log(f"AGENT_{agent_name}", f"{action}: {details}")
    
    def log_error(self, component, error, context=""):
        """记录错误日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [ERROR] [{component}] {error}"
        if context:
            log_entry += f" | 上下文: {context}"
        print(log_entry)
        
        if self.enable_file_logging:
            self._write_to_file(log_entry)
    
    def log_system(self, message):
        """记录系统日志"""
        self.log("SYSTEM", message)
    
    def _write_to_file(self, log_entry):
        """写入日志文件"""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"写入日志文件失败: {e}")


class SampleDataManager:
    """示例数据管理器：统一管理所有示例数据，避免重复定义"""
    
    @staticmethod
    def get_sample_exercise_records() -> List[Dict[str, Any]]:
        """获取示例习题记录（统一格式）"""
        return [
            {
                "problem_id": "P001",
                "problem_content": "求解一元二次方程 x² + 5x + 6 = 0",
                "student_answer": "x = -2 或 x = -3",
                "correct_answer": "x = -2 或 x = -3",
                "is_correct": True,
                "difficulty": "medium",
                "concept": "一元二次方程",
                "knowledge_points": ["一元二次方程", "因式分解", "求根公式"],
                "timestamp": "2024-01-15T10:30:00",
                "time_spent": 120,
                "attempts": 1
            },
            {
                "problem_id": "P002", 
                "problem_content": "计算函数 f(x) = 2x + 3 在 x = 4 处的值",
                "student_answer": "11",
                "correct_answer": "11",
                "is_correct": True,
                "difficulty": "easy",
                "concept": "函数求值",
                "knowledge_points": ["函数", "代数运算", "函数求值"],
                "timestamp": "2024-01-16T14:20:00",
                "time_spent": 45,
                "attempts": 1
            },
            {
                "problem_id": "P003",
                "problem_content": "求函数 y = x² - 4x + 3 的顶点坐标",
                "student_answer": "(2, -1)",
                "correct_answer": "(2, -1)", 
                "is_correct": True,
                "difficulty": "medium",
                "concept": "二次函数",
                "knowledge_points": ["二次函数", "顶点公式", "坐标计算"],
                "timestamp": "2024-01-17T09:15:00",
                "time_spent": 180,
                "attempts": 2
            },
            {
                "problem_id": "P004",
                "problem_content": "解不等式 2x - 5 > 3",
                "student_answer": "x > 4",
                "correct_answer": "x > 4",
                "is_correct": True,
                "difficulty": "easy",
                "concept": "一元一次不等式",
                "knowledge_points": ["不等式", "代数运算", "解集"],
                "timestamp": "2024-01-18T16:45:00",
                "time_spent": 60,
                "attempts": 1
            },
            {
                "problem_id": "P005",
                "problem_content": "求等差数列 3, 7, 11, 15... 的第10项",
                "student_answer": "39",
                "correct_answer": "39",
                "is_correct": True,
                "difficulty": "medium",
                "concept": "等差数列",
                "knowledge_points": ["等差数列", "通项公式", "数列计算"],
                "timestamp": "2024-01-19T11:30:00",
                "time_spent": 150,
                "attempts": 1
            }
        ]
    
    @staticmethod
    def get_simple_exercise_records() -> List[Dict[str, Any]]:
        """获取简化版习题记录（用于知识状态分析）"""
        records = SampleDataManager.get_sample_exercise_records()
        return [
            {
                "question": record["problem_content"],
                "knowledge_points": record["knowledge_points"],
                "is_correct": record["is_correct"]
            }
            for record in records
        ]

