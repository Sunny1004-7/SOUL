# -*- coding: utf-8 -*-
"""
消融实验4: w/o.Cog 变体模型主程序
移除认知驱动机制，教师智能体仅基于学习者情绪状态生成辅导响应
"""
import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

# 导入核心组件
from core import EventBus, ConversationOrchestrator, MessageType
from teacher_agent import TeacherAgent
from student_agent import StudentAgent
from monitor_agent import MonitorAgent
# 注意：不导入knowledge_state_agent
from reflection_agent import ReflectionAgent
from conversation_analyzer import ConversationAnalyzer
from core import BaseAgent, Message


class SimpleLLMManager:
    """简单的LLM管理器"""
    def __init__(self):
        self.config = {
            "model": "gpt-3.5-turbo",
            "base_url": "https://xh.v1api.cc/v1",
            "api_key": "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm"
        }
    
    def call_llm(self, messages, temperature=0.7, max_tokens=1000):
        """调用LLM"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.config["model"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
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
            print(f"LLM调用失败: {e}")
            return None


class SimpleLogger:
    """简单的日志记录器"""
    def __init__(self):
        self.log_content = []
        self.start_time = datetime.now()
    
    def log(self, component, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{component}] {message}"
        self.log_content.append(log_entry)
        print(log_entry)
    
    def log_agent_work(self, agent_name, action, details):
        """记录Agent工作过程"""
        self.log(f"{agent_name}_AGENT", f"{action}: {details}")
    
    def log_analysis_result(self, component, analysis_type, result):
        """记录分析结果"""
        if isinstance(result, dict):
            result_str = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            result_str = str(result)
        self.log(f"{component}_ANALYSIS", f"详细{analysis_type}结果:\n{result_str}")
    
    def get_log_content(self):
        """获取日志内容"""
        return "\n".join(self.log_content)


class ConversationOrchestrator(BaseAgent):
    """对话协调器：管理多智能体之间的对话流程"""
    
    def __init__(self, llm_manager, logger=None):
        super().__init__("orchestrator", logger)
        self.llm_manager = llm_manager
        
        # 对话数据存储
        self.conversation_data = {}
        
        # 反思状态跟踪
        self.reflection_status = {}
        
        # 对话协调器
        self.conversation_orchestrator = None
        
        if self.logger:
            self.logger.log_agent_work("ORCHESTRATOR_AGENT", "初始化完成", "对话协调器已就绪（无认知状态模式）")

    def initialize(self):
        """初始化对话协调器"""
        # 先调用父类的处理器注册
        super()._register_handlers()
        
        # 注册系统控制消息处理器
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control_message

    def _handle_system_control_message(self, message: Message):
        """处理系统控制消息"""
        self._handle_system_control(message.content)

    def start_conversation(self, conversation_id: str, problem_content: str, user_id: str = None):
        """启动对话"""
        if self.logger:
            self.logger.log_agent_work("ORCHESTRATOR_AGENT", "启动对话", f"对话ID: {conversation_id}（无认知状态模式）")
        
        # 初始化对话数据
        self.conversation_data[conversation_id] = {
            "status": "active",
            "start_time": datetime.now().isoformat(),
            "problem_content": problem_content,
            "user_id": user_id,
            "rounds_completed": 0,
            "messages": []
        }
        
        # 发送启动消息给学生智能体
        start_message = Message(
            id="",
            sender="orchestrator",
            recipient="student",
            type=MessageType.TASK_REQUEST,
            content={
                "instruction": "start_conversation",
                "conversation_id": conversation_id,
                "problem_content": problem_content
            },
            timestamp=datetime.now().isoformat(),
            correlation_id=conversation_id
        )
        self.event_bus.send_message(start_message)

    def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话状态"""
        if conversation_id in self.conversation_data:
            return self.conversation_data[conversation_id]
        return {"status": "not_found"}

    def get_conversation_history(self, conversation_id: str) -> list:
        """获取对话历史"""
        if conversation_id in self.conversation_data:
            return self.conversation_data[conversation_id].get("messages", [])
        return []

    def get_reflection_status(self, conversation_id: str) -> Dict[str, Any]:
        """获取反思状态"""
        return self.reflection_status.get(conversation_id, {"completed": False, "started": False})

    def _handle_system_control(self, message: Dict[str, Any]):
        """处理系统控制消息"""
        action = message.get("action")
        conversation_id = message.get("conversation_id")
        
        if action == "add_message":
            # 添加消息到对话历史
            if conversation_id in self.conversation_data:
                self.conversation_data[conversation_id]["messages"].append({
                    "sender": message.get("sender"),
                    "content": message.get("content"),
                    "type": message.get("message_type", "message"),
                    "timestamp": message.get("timestamp", datetime.now().isoformat())
                })
                
        elif action == "reflection_response":
            # 处理反思响应
            if self.logger:
                self.logger.log_agent_work("ORCHESTRATOR_AGENT", "收到反思响应", f"对话ID: {conversation_id}")
            
            # 更新反思状态为完成
            if conversation_id in self.reflection_status:
                self.reflection_status[conversation_id]["completed"] = True
                self.reflection_status[conversation_id]["result"] = message.get("reflection_result", {})
            
        elif action == "conversation_ended":
            # 对话结束
            if conversation_id in self.conversation_data:
                self.conversation_data[conversation_id]["status"] = "completed"
        
        elif action == "end_conversation":
            # 结束对话
            if self.logger:
                self.logger.log_agent_work("ORCHESTRATOR_AGENT", "处理对话结束", f"对话ID: {conversation_id}")
            self._end_conversation(conversation_id, message)
        
        elif action == "conversation_history_response":
            # 接收对话历史响应
            if conversation_id in self.conversation_data:
                conversation_history = message.get("conversation_history", [])
                self.conversation_data[conversation_id]["messages"] = conversation_history
                if self.logger:
                    self.logger.log_agent_work("ORCHESTRATOR_AGENT", "收到对话历史", f"对话ID: {conversation_id}, 消息数: {len(conversation_history)}")

    def _end_conversation(self, conversation_id: str, message: Dict[str, Any]):
        """结束对话"""
        if conversation_id in self.conversation_data:
            self.conversation_data[conversation_id]["status"] = "ending"
            self.conversation_data[conversation_id]["end_time"] = datetime.now().isoformat()
            self.conversation_data[conversation_id]["end_reason"] = message.get("reason", "unknown")
            
            # 从对话协调器获取实际轮数
            actual_rounds = 0
            if self.conversation_orchestrator:
                conversation_history = self.conversation_orchestrator.get_conversation_history(conversation_id)
                actual_rounds = len([msg for msg in conversation_history if msg.get("sender") == "student"])
            
            # 更新轮数
            self.conversation_data[conversation_id]["rounds_completed"] = actual_rounds
            
            if self.logger:
                self.logger.log_agent_work("ORCHESTRATOR_AGENT", "对话结束", 
                                         f"总轮数: {actual_rounds}, 原因: {message.get('reason', 'unknown')}")
            
            # 发送反思请求给反思Agent
            self._request_reflection(conversation_id, message)
            
            # 标记对话为已完成
            self.conversation_data[conversation_id]["status"] = "completed"

    def _request_reflection(self, conversation_id: str, message: Dict[str, Any]):
        """请求反思Agent进行对话总结反思"""
        try:
            # 获取对话历史
            conversation_history = self.get_conversation_history(conversation_id)
            
            # 发送反思请求
            reflection_message = Message(
                id="",
                sender="orchestrator",
                recipient="reflection",
                type=MessageType.REFLECTION_REQUEST,
                content={
                    "conversation_id": conversation_id,
                    "conversation_history": conversation_history,
                    "is_conversation_end": True,
                    "analysis_result": message.get("analysis_result", {})
                },
                timestamp=datetime.now().isoformat(),
                correlation_id=conversation_id
            )
            self.event_bus.send_message(reflection_message)
            
            if self.logger:
                self.logger.log_agent_work("ORCHESTRATOR_AGENT", "反思请求已发送", f"对话ID: {conversation_id}")
                
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("ORCHESTRATOR_AGENT", "反思请求失败", f"错误: {e}")

    def _cleanup_conversation(self, conversation_id: str):
        """清理对话数据"""
        if conversation_id in self.conversation_data:
            self.conversation_data[conversation_id]["status"] = "completed"
            if self.logger:
                self.logger.log_agent_work("ORCHESTRATOR_AGENT", "对话清理完成", f"对话ID: {conversation_id}")

    def _force_reflection_on_timeout(self, conversation_id: str):
        """超时时强制执行反思"""
        if self.logger:
            self.logger.log_agent_work("ORCHESTRATOR_AGENT", "超时强制反思", f"对话ID: {conversation_id}")
        
        # 更新反思状态
        self.reflection_status[conversation_id] = {"completed": False, "started": True}
        
        try:
            # 获取对话历史
            conversation_history = self.get_conversation_history(conversation_id)
            
            # 发送反思请求
            reflection_message = Message(
                id="",
                sender="orchestrator",
                recipient="reflection",
                type=MessageType.REFLECTION_REQUEST,
                content={
                    "conversation_id": conversation_id,
                    "conversation_history": conversation_history,
                    "is_conversation_end": True,
                    "is_timeout": True,
                    "analysis_result": {
                        "should_end": True,
                        "reason": "对话超时，强制结束",
                        "round_number": len([msg for msg in conversation_history if msg.get("sender") == "student"])
                    }
                },
                timestamp=datetime.now().isoformat(),
                correlation_id=conversation_id
            )
            self.event_bus.send_message(reflection_message)
            
            if self.logger:
                self.logger.log_agent_work("ORCHESTRATOR_AGENT", "超时反思请求已发送", f"对话ID: {conversation_id}")
                
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("ORCHESTRATOR_AGENT", "超时反思请求失败", f"错误: {e}")
            # 反思失败，标记为完成
            self.reflection_status[conversation_id] = {"completed": True, "started": True, "error": str(e)}

    # 添加BaseAgent需要的方法
    def set_event_bus(self, event_bus):
        """设置事件总线"""
        self.event_bus = event_bus
    
    def initialize(self):
        """初始化"""
        pass
    
    def start(self):
        """启动"""
        pass
    
    def stop(self):
        """停止"""
        pass
    
    def receive_message(self, message):
        """接收消息"""
        if message.type == MessageType.SYSTEM_CONTROL:
            self._handle_system_control(message.content)
        elif self.logger:
            self.logger.log_agent_work("ORCHESTRATOR_AGENT", "收到未知消息", f"类型: {message.type.value}")


def load_problem_content(user_id: str = None) -> str:
    """加载问题内容"""
    try:
        from student_data_loader import StudentDataLoader
        loader = StudentDataLoader()
        
        if user_id is None:
            user_id = loader.get_first_student_id()
            if user_id is None:
                return "请解释一下什么是函数？"
        
        # 获取最后一道题目作为问题内容
        problem_content = loader.get_last_problem_content(user_id)
        if problem_content:
            return problem_content
        else:
            return "请解释一下什么是函数？"
            
    except Exception as e:
        print(f"加载问题内容失败: {e}")
        return "请解释一下什么是函数？"


def main():
    """主函数"""
    print("=== 消融实验4: w/o.Cog 教学对话系统启动 ===")
    
    # 1. 初始化核心组件
    llm_manager = SimpleLLMManager()
    logger = SimpleLogger()
    
    # 2. 加载问题内容
    problem_content = load_problem_content()
    print(f"问题内容: {problem_content[:100]}...")
    
    # 3. 创建智能体（注意：不创建knowledge_state_agent）
    teacher_agent = TeacherAgent("teacher", llm_manager, logger)
    student_agent = StudentAgent("student", llm_manager, problem_content, logger=logger)
    monitor_agent = MonitorAgent("monitor", llm_manager, logger)
    # knowledge_state_agent 被移除
    reflection_agent = ReflectionAgent("reflection", llm_manager, "experience_bank.json", logger)
    
    # 4. 创建对话协调器
    orchestrator_agent = ConversationOrchestrator(llm_manager, logger)
    
    # 5. 创建对话分析器
    conversation_analyzer = ConversationAnalyzer(llm_manager, logger)
    
    # 6. 设置事件总线
    event_bus = EventBus(logger)
    
    # 注册智能体到事件总线（注意：不注册knowledge_state_agent）
    event_bus.register_agent(teacher_agent)
    event_bus.register_agent(student_agent)
    event_bus.register_agent(monitor_agent)
    # event_bus.register_agent(knowledge_state_agent)  # 被移除
    event_bus.register_agent(reflection_agent)
    event_bus.register_agent(orchestrator_agent)
    
    # 设置智能体的事件总线
    teacher_agent.set_event_bus(event_bus)
    student_agent.set_event_bus(event_bus)
    monitor_agent.set_event_bus(event_bus)
    # knowledge_state_agent.set_event_bus(event_bus)  # 被移除
    reflection_agent.set_event_bus(event_bus)
    
    # 设置协调器的事件总线
    orchestrator_agent.event_bus = event_bus
    orchestrator_agent.conversation_orchestrator = orchestrator_agent
    
    # 7. 初始化智能体
    teacher_agent.initialize()
    student_agent.initialize()
    monitor_agent.initialize()
    # knowledge_state_agent.initialize()  # 被移除
    reflection_agent.initialize()
    
    # 8. 启动事件总线（必须在所有Agent注册后立即启动）
    event_bus.start()
    
    # 9. 启动智能体
    teacher_agent.start()
    student_agent.start()
    monitor_agent.start()
    # knowledge_state_agent.start()  # 被移除
    reflection_agent.start()
    
    # 添加反思Agent启动确认
    logger.log("SYSTEM", f"反思Agent启动状态: {reflection_agent.name} - {'已启动' if reflection_agent.running else '未启动'}")
    
    # 10. 生成对话ID
    conversation_id = str(uuid.uuid4())
    
    # 11. 启动对话
    orchestrator_agent.start_conversation(conversation_id, problem_content)
    
    # 12. 等待对话完成
    max_wait_time = 600  # 最大等待10分钟
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status = orchestrator_agent.get_conversation_status(conversation_id)
        if status.get("status") == "completed":
            logger.log("SYSTEM", "对话正常完成")
            break
        time.sleep(2)
    else:
        logger.log("SYSTEM", "对话超时，强制结束")
    
    # 13. 等待反思Agent处理经验（最多再等10秒）
    for _ in range(10):
        if os.path.exists("experience_bank.json") and os.path.getsize("experience_bank.json") > 0:
            logger.log("SYSTEM", "经验已成功写入 experience_bank.json")
            break
        time.sleep(1)
    else:
        logger.log("SYSTEM", "未检测到经验写入 experience_bank.json")

    # 14. 保存对话记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("run_output", f"output_wo_cog_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存系统日志
    system_log_file = os.path.join(output_dir, f"system_log_{timestamp}.txt")
    try:
        with open(system_log_file, "w", encoding="utf-8") as f:
            f.write(logger.get_log_content())
    except Exception as e:
        print(f"写入日志文件失败: {e}")
    
    # 保存完整对话记录到输出目录
    conversation_json_file = os.path.join(output_dir, f"conversation_log_{timestamp}.json")
    
    try:
        conversation_history = orchestrator_agent.get_conversation_history(conversation_id)
        if conversation_history:
            # 保存JSON格式对话记录
            with open(conversation_json_file, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_type": "w/o.Cog (无认知状态)",
                    "conversation_id": conversation_id,
                    "problem_content": problem_content,
                    "start_time": datetime.now().isoformat(),
                    "total_rounds": len([msg for msg in conversation_history if msg.get("sender") == "student"]),
                    "conversation_history": conversation_history
                }, f, ensure_ascii=False, indent=2)
        else:
            print("警告：对话历史为空")
    except Exception as e:
        print(f"保存对话记录失败: {e}")
    
    # 15. 输出统计信息
    final_status = orchestrator_agent.get_conversation_status(conversation_id)
    total_rounds = final_status.get("rounds_completed", 0)
    
    print("\n=== 消融实验4对话完成 ===")
    print(f"实验类型: w/o.Cog (无认知状态)")
    print(f"对话ID: {conversation_id}")
    print(f"总轮数: {total_rounds}")
    print(f"系统日志: {system_log_file}")
    print(f"对话记录: {conversation_json_file}")
    print("=== 系统结束 ===")


if __name__ == "__main__":
    main()