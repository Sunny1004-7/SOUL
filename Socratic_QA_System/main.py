# -*- coding: utf-8 -*-
"""
智能苏格拉底教学范式问答系统主程序
基于AutoGen架构的简化版教学问答系统
系统启动时生成一次知识状态摘要，供后续所有教学对话使用
"""
import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

# 导入核心组件
from core import EventBus, MessageType, LLMManager, Logger, ConversationOrchestrator, SampleDataManager
from teacher_agent import TeacherAgent
from monitor_agent import MonitorAgent
from knowledge_state_agent import KnowledgeStateAgent
from config import Config


class SocraticQASystem:
    """苏格拉底问答系统主控制器"""
    
    def __init__(self):
        # 加载配置
        Config.load_from_env()
        self.system_config = Config.get_system_config()
        
        # 使用统一的LLM管理器和日志记录器
        self.llm_manager = LLMManager(Config.get_llm_config())
        self.logger = Logger(enable_file_logging=False)
        self.event_bus = EventBus(self.logger)
        
        # 初始化对话协调器
        self.conversation_orchestrator = ConversationOrchestrator(self.event_bus, self.logger)
        
        # 初始化智能体
        self.teacher_agent = TeacherAgent("teacher", self.llm_manager, self.logger)
        
        # 根据配置决定是否启用监控和知识分析智能体
        if self.system_config.get("enable_monitoring", True):
            self.monitor_agent = MonitorAgent("monitor", self.llm_manager, self.logger)
        else:
            self.monitor_agent = None
            
        if self.system_config.get("enable_knowledge_analysis", True):
            self.knowledge_state_agent = KnowledgeStateAgent("knowledge_state", self.llm_manager, self.logger)
        else:
            self.knowledge_state_agent = None
        
        # 注册智能体到事件总线
        self.event_bus.register_agent(self.teacher_agent)
        if self.monitor_agent:
            self.event_bus.register_agent(self.monitor_agent)
        if self.knowledge_state_agent:
            self.event_bus.register_agent(self.knowledge_state_agent)
        
        # 启动事件总线
        self.event_bus.start()
        
        # 初始化所有智能体
        self.teacher_agent.initialize()
        if self.monitor_agent:
            self.monitor_agent.initialize()
        if self.knowledge_state_agent:
            self.knowledge_state_agent.initialize()
        
        # 启动所有智能体（关键：启动消息处理线程）
        self.teacher_agent.start()
        if self.monitor_agent:
            self.monitor_agent.start()
        if self.knowledge_state_agent:
            self.knowledge_state_agent.start()
        
        # 等待知识状态智能体初始化完成
        if self.knowledge_state_agent:
            self._wait_for_knowledge_agent_ready()
        
        # 对话相关变量
        self.conversation_id = None
        self.round_number = 0
        self.conversation_history = []
        self.max_rounds = self.system_config.get("max_conversation_rounds", 10)
        
        # 知识状态摘要（系统启动时生成一次）
        self.knowledge_summary = None
        
        # 使用统一的示例数据管理器获取习题记录
        self.exercise_records = SampleDataManager.get_sample_exercise_records()
        
        # 获取知识状态摘要
        self._get_knowledge_summary()
        
        # 调用知识状态追踪智能体分析默认记录
        if self.knowledge_state_agent:
            self._analyze_default_exercise_records()
    
    def _wait_for_knowledge_agent_ready(self):
        """等待知识状态智能体准备就绪"""
        max_wait_time = 30  # 最大等待时间（秒）
        wait_interval = 0.5  # 检查间隔（秒）
        elapsed_time = 0
        
        self.logger.log_system("等待知识状态智能体初始化...")
        
        while elapsed_time < max_wait_time:
            if (self.knowledge_state_agent and 
                self.knowledge_state_agent.analysis_ready):
                self.logger.log_system("知识状态智能体已就绪")
                return
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        self.logger.log_system("知识状态智能体初始化超时，继续启动系统")
    
    def _get_knowledge_summary(self):
        """获取知识状态摘要"""
        if not self.knowledge_state_agent:
            self.knowledge_summary = "知识状态分析功能已禁用"
            return
        
        try:
            # 等待知识状态智能体准备就绪
            if not self.knowledge_state_agent.analysis_ready:
                self.logger.log_system("知识状态智能体尚未就绪，使用默认摘要")
                self.knowledge_summary = "系统正在初始化知识状态分析，使用默认设置"
                return
            
            # 获取预生成的知识状态摘要
            self.knowledge_summary = self.knowledge_state_agent.get_knowledge_summary()
            
            if self.knowledge_summary:
                self.logger.log_system("知识状态摘要获取成功")
                print(f"摘要: {self.knowledge_summary}")
                print("-" * 50)
            else:
                self.knowledge_summary = "知识状态分析完成，系统已就绪"
                
        except Exception as e:
            self.logger.log_error("SYSTEM", f"获取知识状态摘要失败: {e}")
            self.knowledge_summary = "知识状态分析初始化失败，使用默认设置"
    
    def start_conversation(self, initial_question: str):
        """开始对话"""
        if self.conversation_id:
            print("系统: 已有对话在进行中，请先结束当前对话。")
            return
        
        # 生成新的对话ID
        self.conversation_id = str(uuid.uuid4())
        self.round_number = 0
        self.conversation_history = []
        
        print(f"\n系统: 新对话已开始 (ID: {self.conversation_id[:8]}...)")
        print("=" * 50)
        
        # 显示知识状态摘要
        if self.knowledge_summary:
            print(f"知识状态摘要: {self.knowledge_summary}")
            print("-" * 50)
        
        # 处理初始问题
        self._process_user_question(initial_question)
    
    def _process_user_question(self, user_question: str):
        """处理用户问题"""
        self.round_number += 1
        
        # 检查是否达到最大轮数
        if self.round_number > self.max_rounds:
            print(f"\n系统: 已达到最大对话轮数({self.max_rounds}轮)，建议开始新对话。")
            print("输入 'new' 开始新对话，或 'quit' 退出系统。")
            return
        
        # 记录用户问题
        self.conversation_history.append({
            "role": "user",
            "content": user_question,
            "round": self.round_number,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"\n用户 (第{self.round_number}轮): {user_question}")
        print("-" * 50)
        
        # 发送消息给教师智能体
        from core import Message
        message = Message(
            id=str(uuid.uuid4()),
            sender="system",
            recipient="teacher",
            type=MessageType.TASK_REQUEST,
            content={
                "conversation_id": self.conversation_id,
                "student_message": user_question,
                "round_number": self.round_number,
                "student_state": {
                    "conversation_history": self.conversation_history,
                    "knowledge_summary": self.knowledge_summary
                }
            },
            timestamp=datetime.now().isoformat()
        )
        self.event_bus.send_message(message)
        
        # 等待教师回复
        print("系统: 正在生成回复，请稍候...")
        
        # 简单等待机制（实际应用中应该使用更优雅的异步处理）
        time.sleep(2)
        
        # 获取教师回复
        teacher_response = self._get_teacher_response()
        
        if teacher_response:
            print(f"\n教师 (第{self.round_number}轮): {teacher_response}")
            print("-" * 50)
            
            # 记录教师回复
            self.conversation_history.append({
                "role": "teacher",
                "content": teacher_response,
                "round": self.round_number,
                "timestamp": datetime.now().isoformat()
            })
        else:
            print("\n系统: 教师回复生成失败，请重试。")
    
    def _get_teacher_response(self) -> str:
        """获取教师回复（通过调用TeacherAgent）"""
        try:
            # 等待教师智能体处理消息并生成回复
            # 不设置硬编码超时，让教师Agent有足够时间完成ICECoT流程
            wait_interval = 0.5  # 检查间隔（秒）
            elapsed_time = 0
            
            print("系统: 正在等待教师智能体生成回复...")
            print("系统: 教师Agent正在进行情绪分析->意图推断->策略选择->响应生成...")
            
            # 持续等待，直到教师Agent生成有效回复
            while True:
                # 检查TeacherAgent是否有回复
                if hasattr(self.teacher_agent, 'conversation_history') and self.teacher_agent.conversation_history:
                    # 获取最新的教师回复
                    for message in reversed(self.teacher_agent.conversation_history):
                        if message.get("role") == "teacher":
                            teacher_response = message.get("content", "")
                            if teacher_response and teacher_response != "正在生成回复...":
                                self.logger.log_system("教师回复获取成功")
                                return teacher_response
                
                # 如果还没有回复，继续等待
                time.sleep(wait_interval)
                elapsed_time += wait_interval
                
                # 显示等待进度，但不设置最大等待时间
                if elapsed_time % 10 == 0:  # 每10秒显示一次进度
                    print(f"系统: 教师Agent仍在处理中... ({elapsed_time}秒)")
                    print("系统: 请耐心等待，ICECoT流程需要时间完成...")
                
                # 可选：如果等待时间过长，给出提示但不强制终止
                if elapsed_time > 120:  # 2分钟后给出提示
                    print("系统: 教师Agent处理时间较长，这可能是正常的，请继续等待...")
                    print("系统: 复杂的教学策略需要更多时间来分析学生状态和生成个性化回复...")
                
        except Exception as e:
            self.logger.log_error("SYSTEM", f"获取教师回复失败: {e}")
            print(f"系统: 获取教师回复时发生错误: {e}")
            return "抱歉，我在生成回复时遇到了一些问题。请重新描述一下你的问题，我会尽力帮助你。"
    
    def _analyze_default_exercise_records(self):
        """调用知识状态追踪智能体分析默认记录"""
        if not self.knowledge_state_agent or not self.exercise_records:
            self.logger.log_system("知识状态智能体或习题记录未准备好，无法分析。")
            return
        
        try:
            # 将习题记录转换为JSON字符串
            exercise_data_json = json.dumps(self.exercise_records)
            
            # 发送消息给知识状态智能体
            from core import Message
            message = Message(
                id=str(uuid.uuid4()),
                sender="system",
                recipient="knowledge_state",
                type=MessageType.DATA_REQUEST,
                content={"exercise_data": exercise_data_json},
                timestamp=datetime.now().isoformat()
            )
            self.event_bus.send_message(message)
            
            self.logger.log_system("已发送习题记录给知识状态智能体进行分析。")
            
        except Exception as e:
            self.logger.log_error("SYSTEM", f"发送习题记录给知识状态智能体失败: {e}")
    
    def continue_conversation(self, user_input: str):
        """继续对话"""
        if not self.conversation_id:
            print("系统: 请先开始一个对话。")
            return
        
        self._process_user_question(user_input)
    
    def end_conversation(self):
        """结束当前对话"""
        if self.conversation_id:
            print(f"\n系统: 对话已结束 (ID: {self.conversation_id[:8]}...)")
            print("=" * 50)
            
            # 发送结束对话消息
            if self.teacher_agent:
                from core import Message
                message = Message(
                    id=str(uuid.uuid4()),
                    sender="system",
                    recipient="teacher",
                    type=MessageType.SYSTEM_CONTROL,
                    content={"command": "end_conversation"},
                    timestamp=datetime.now().isoformat()
                )
                self.event_bus.send_message(message)
            
            self.conversation_id = None
            self.round_number = 0
            self.conversation_history = []
        else:
            print("系统: 当前没有进行中的对话。")
    
    def get_conversation_summary(self):
        """获取对话摘要"""
        if not self.conversation_history:
            print("系统: 当前没有对话历史。")
            return
        
        print(f"\n对话摘要 (共{len(self.conversation_history)}条消息):")
        print("=" * 50)
        
        for i, message in enumerate(self.conversation_history, 1):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            round_num = message.get("round", "N/A")
            timestamp = message.get("timestamp", "N/A")
            
            print(f"{i}. [{role.upper()}] (第{round_num}轮) - {timestamp}")
            print(f"   内容: {content}")
            print("-" * 30)


def main():
    """主函数"""
    print("=" * 60)
    print("智能苏格拉底教学范式问答系统")
    print("基于AutoGen架构的简化版教学问答系统")
    print("=" * 60)
    
    # 创建系统实例
    system = SocraticQASystem()
    
    print("\n系统: 系统初始化完成，可以开始对话。")
    print("输入 'help' 查看帮助，输入 'quit' 退出系统。")
    print("-" * 50)
    
    # 主循环
    while True:
        try:
            user_input = input("\n请输入您的问题 (或命令): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n系统: 感谢使用，再见！")
                break
            elif user_input.lower() == 'help':
                print("\n帮助信息:")
                print("- 直接输入问题开始对话")
                print("- 输入 'new' 开始新对话")
                print("- 输入 'end' 结束当前对话")
                print("- 输入 'summary' 查看对话摘要")
                print("- 输入 'quit' 退出系统")
            elif user_input.lower() == 'new':
                if system.conversation_id:
                    system.end_conversation()
                initial_question = input("请输入初始问题: ").strip()
                if initial_question:
                    system.start_conversation(initial_question)
                else:
                    print("系统: 请输入有效的问题。")
            elif user_input.lower() == 'end':
                system.end_conversation()
            elif user_input.lower() == 'summary':
                system.get_conversation_summary()
            else:
                # 处理用户问题
                if not system.conversation_id:
                    system.start_conversation(user_input)
                else:
                    system.continue_conversation(user_input)
                    
        except KeyboardInterrupt:
            print("\n\n系统: 检测到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n系统: 发生错误: {e}")
            print("请重试或输入 'quit' 退出系统。")
    
    # 清理资源
    if hasattr(system, 'event_bus'):
        system.event_bus.stop()


if __name__ == "__main__":
    main()
