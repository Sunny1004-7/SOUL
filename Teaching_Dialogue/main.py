# -*- coding: utf-8 -*-
"""
基于AutoGen架构的智能教学对话系统主程序
使用Actor模型、异步消息传递和事件驱动架构，实现更好的模块化和可扩展性
"""
import time
import uuid
import json
import os
from datetime import datetime
from typing import Dict, Any

from core import EventBus, ConversationOrchestrator, BaseAgent, MessageType, Message
from student_agent import StudentAgent
from teacher_agent import TeacherAgent
from monitor_agent import MonitorAgent
from reflection_agent import ReflectionAgent
from knowledge_state_agent import KnowledgeStateAgent



class LLMManager:
    """统一的LLM调用管理器"""
    def __init__(self, llm_config, logger=None):
        self.config = llm_config["config_list"][0]
        self.api_key = self.config["api_key"]
        self.base_url = self.config.get("base_url", "https://api.openai.com/v1")
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.logger = logger
    
    def call_llm(self, messages, temperature=0.7, max_tokens=1000):
        """统一的LLM调用接口"""
        try:
            import requests
            import json
            
            if self.logger:
                self.logger.log("LLM_CALL", f"调用LLM - 模型: {self.model}, 温度: {temperature}")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result_data = response.json()
                result = result_data["choices"][0]["message"]["content"].strip()
                
                if self.logger:
                    self.logger.log("LLM_CALL", f"LLM调用成功，响应长度: {len(result)}字符")
                
                return result
            else:
                error_msg = f"LLM API调用失败: {response.status_code} - {response.text}"
                if self.logger:
                    self.logger.log("LLM_ERROR", error_msg)
                return None
                
        except Exception as e:
            error_msg = f"LLM调用失败: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log("LLM_ERROR", error_msg)
            return None


class SystemLogger:
    """系统日志记录器"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = datetime.now()
        self._write_header()
    
    def _write_header(self):
        """写入日志文件头部信息"""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("AutoGen架构智能教学对话系统运行日志\n")
                f.write(f"运行时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"写入日志文件头部失败: {e}")
    
    def log(self, component, message):
        """记录日志信息"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{component}] {message}\n"
        
        # 写入日志文件
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"写入日志文件失败: {e}")
        
        # 同时打印到控制台
        print(f"[{component}] {message}")
    
    def log_conversation(self, role, content, round_num):
        """记录对话内容"""
        self.log("CONVERSATION", f"第{round_num}轮 - {role}: {content}")
    
    def log_agent_work(self, agent_name, action, details):
        """记录Agent工作过程"""
        self.log(f"{agent_name}_AGENT", f"{action}: {details}")
    
    def log_analysis_result(self, agent_name, analysis_type, result):
        """记录分析结果"""
        if isinstance(result, dict):
            import json
            result_str = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            result_str = str(result)
        self.log(f"{agent_name}_ANALYSIS", f"{analysis_type}结果:\n{result_str}")


class Orchestrator(BaseAgent):
    """基于AutoGen架构的团队协调器"""
    
    def __init__(self, name: str, logger=None):
        super().__init__(name, logger)
        self.conversation_orchestrator = None
        self.current_conversation = None
        self.conversation_data = {}
        
        # 注册消息处理器
        self._register_orchestrator_handlers()
        
        if self.logger:
            self.logger.log_agent_work("ORCHESTRATOR", "初始化完成", "团队协调器已就绪")

    def initialize(self):
        """初始化协调器"""
        self.update_state("ready", True)
        self.update_state("conversations_managed", 0)

    def _register_orchestrator_handlers(self):
        """注册协调器消息处理器"""
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control
        self.message_handlers[MessageType.REFLECTION_RESPONSE] = self._handle_reflection_response

    def set_conversation_orchestrator(self, orchestrator: ConversationOrchestrator):
        """设置对话协调器"""
        self.conversation_orchestrator = orchestrator

    def start_conversation(self, problem_content: str, max_rounds: int = 5) -> str:
        """启动对话"""
        conversation_id = str(uuid.uuid4())
        participants = ["student", "teacher", "monitor", "reflection"]
        
        self.current_conversation = conversation_id
        self.conversation_data[conversation_id] = {
            "problem_content": problem_content,
            "max_rounds": max_rounds,
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "rounds_completed": 0
        }
        
        if self.logger:
            self.logger.log_agent_work("ORCHESTRATOR", "启动对话", f"ID: {conversation_id}")
        
        # 使用对话协调器启动对话
        if self.conversation_orchestrator:
            self.conversation_orchestrator.start_conversation(
                conversation_id, participants, "", problem_content
            )
        
        return conversation_id

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        action = content.get("action")
        conversation_id = content.get("conversation_id")
        
        if action == "end_conversation" and conversation_id:
            self._end_conversation(conversation_id, content)
        elif action == "conversation_ended" and conversation_id:
            self._cleanup_conversation(conversation_id)
        elif action == "add_message" and conversation_id:
            # 添加消息到对话历史
            sender = content.get("sender")
            msg_content = content.get("content")
            msg_type = content.get("message_type", "message")
            if sender and msg_content and self.conversation_orchestrator:
                self.conversation_orchestrator.add_message_to_conversation(
                    conversation_id, sender, msg_content, msg_type
                )

    def _handle_reflection_response(self, message: Message):
        """处理反思响应"""
        content = message.content
        conversation_id = content.get("conversation_id")
        reflection_result = content.get("reflection_result", {})
        
        if self.logger and reflection_result.get("summary"):
            self.logger.log("ORCHESTRATOR", f"反思完成: {reflection_result['summary']}")

    def _end_conversation(self, conversation_id: str, content: Dict[str, Any]):
        """结束对话"""
        if conversation_id in self.conversation_data:
            self.conversation_data[conversation_id]["status"] = "ending"
            self.conversation_data[conversation_id]["end_time"] = datetime.now().isoformat()
            self.conversation_data[conversation_id]["end_reason"] = content.get("reason", "unknown")
            
            final_message = content.get("final_message", "")
            if final_message and self.logger:
                self.logger.log_conversation("学生", final_message, 
                    self.conversation_data[conversation_id].get("rounds_completed", 0))
            
            if self.logger:
                rounds = self.conversation_data[conversation_id].get("rounds_completed", 0)
                reason = content.get("reason", "unknown")
                self.logger.log_agent_work("ORCHESTRATOR", "对话结束", f"总轮数: {rounds}, 原因: {reason}")
            
            # 通知所有Agent对话结束
            for agent_name in ["student", "teacher", "monitor", "reflection"]:
                self.send_message(
                    recipient=agent_name,
                    message_type=MessageType.SYSTEM_CONTROL,
                    content={
                        "action": "conversation_ended",
                        "conversation_id": conversation_id
                    }
                )
            
            # 使用对话协调器结束对话
            if self.conversation_orchestrator:
                self.conversation_orchestrator.end_conversation(conversation_id)

    def _cleanup_conversation(self, conversation_id: str):
        """清理对话状态"""
        if conversation_id in self.conversation_data:
            self.conversation_data[conversation_id]["status"] = "completed"
            
        if self.current_conversation == conversation_id:
            self.current_conversation = None
            
        # 更新统计
        managed_count = self.state.get("conversations_managed", 0) + 1
        self.update_state("conversations_managed", managed_count)

    def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话状态"""
        return self.conversation_data.get(conversation_id, {})


def load_problem_content(problem_file="problem.txt"):
    """加载题目内容"""
    try:
        with open(problem_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "未找到题目文件，请提供具体问题。"


def main():
    """主函数"""
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录 - 使用run_output目录避免权限问题
    output_base_dir = "run_output"  # 使用run_output目录
    os.makedirs(output_base_dir, exist_ok=True)
    output_dir = os.path.join(output_base_dir, f"output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志系统 - 使用输出目录
    system_log_file = os.path.join(output_dir, f"system_log_{timestamp}.txt")
    logger = SystemLogger(system_log_file)
    logger.log("SYSTEM", "AutoGen架构智能教学对话系统启动")
    logger.log("SYSTEM", f"输出目录: {output_dir}")
    
    # 初始化LLM配置
    llm_config = {
        "cache_seed": None,
        "config_list": [{
            "model": "gpt-3.5-turbo",
            "base_url": "https://xh.v1api.cc/v1",
            "api_key": "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm",
            "price": [0, 0]
        }]
    }
    
    # 创建LLM管理器
    llm_manager = LLMManager(llm_config, logger)
    logger.log("SYSTEM", "LLM管理器初始化完成")
    
    # 加载题目内容
    problem_content = load_problem_content()
    logger.log("SYSTEM", f"题目加载完成: {problem_content}")
    
    # 经验库文件路径
    experience_bank_file = "experience_bank.json"  # 使用根目录下的全局经验库
    logger.log("SYSTEM", "经验库文件路径设置完成")
    
    # 创建事件总线
    event_bus = EventBus(logger)
    logger.log("SYSTEM", "事件总线创建完成")
    
    # 创建对话协调器
    conversation_orchestrator = ConversationOrchestrator(event_bus, logger)
    logger.log("SYSTEM", "对话协调器创建完成")
    
    # 创建智能体
    logger.log("SYSTEM", "开始创建智能体...")
    
    student_agent = StudentAgent("student", llm_manager, problem_content, logger=logger)
    teacher_agent = TeacherAgent("teacher", llm_manager, logger=logger)
    monitor_agent = MonitorAgent("monitor", llm_manager, logger=logger)
    reflection_agent = ReflectionAgent("reflection", llm_manager, experience_bank_file, logger=logger)
    knowledge_agent = KnowledgeStateAgent("knowledge", llm_manager, logger=logger)
    orchestrator_agent = Orchestrator("orchestrator", logger=logger)
    
    # 设置对话协调器
    orchestrator_agent.set_conversation_orchestrator(conversation_orchestrator)
    
    logger.log("SYSTEM", "智能体创建完成")
    
    # 注册智能体到事件总线
    agents = [student_agent, teacher_agent, monitor_agent, reflection_agent, knowledge_agent, orchestrator_agent]
    for agent in agents:
        event_bus.register_agent(agent)
        agent.initialize()
    
    logger.log("SYSTEM", "智能体注册完成")
    
    # 启动系统
    event_bus.start()
    for agent in agents:
        agent.start()
    
    logger.log("SYSTEM", "系统启动完成")
    
    print("=== AutoGen架构智能教学对话系统启动 ===")
    print(f"题目: {problem_content}")
    print(f"输出目录: {output_dir}")
    print("=" * 50)
    
    try:
        # 第一步：在对话开始前获取知识状态总结
        logger.log("SYSTEM", "正在获取学生知识状态总结...")
        teacher_agent.send_message(
            recipient="knowledge",
            message_type=MessageType.TASK_REQUEST,
            content={
                "conversation_id": "pre_conversation",
                "analysis_type": "overall_summary"
            }
        )
        
        # 等待知识状态总结完成
        time.sleep(3)  # 给知识状态Agent一些时间生成总结
        
        # 第二步：启动对话
        conversation_id = orchestrator_agent.start_conversation(problem_content, max_rounds=5)
        logger.log("SYSTEM", f"对话启动，ID: {conversation_id}")
        
        # 等待对话完成
        max_wait_time = 300  # 最大等待5分钟
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = orchestrator_agent.get_conversation_status(conversation_id)
            if status.get("status") == "completed":
                logger.log("SYSTEM", "对话正常完成")
                break
            time.sleep(2)
        else:
            logger.log("SYSTEM", "对话超时，强制结束")
            orchestrator_agent._end_conversation(conversation_id, {
                "reason": "timeout",
                "final_message": "对话超时结束"
            })
        
        # 等待系统处理完所有消息
        time.sleep(3)
        
        # 第三步：对话结束后进行总结反思
        logger.log("SYSTEM", "开始对话总结反思...")
        conversation_history = conversation_orchestrator.get_conversation_history(conversation_id)
        if conversation_history:
            # 直接调用反思智能体进行总结反思
            reflection_result = reflection_agent.perform_conversation_reflection(
                conversation_id=conversation_id,
                conversation_history=conversation_history,
                student_final_state=student_agent.get_student_state(),
                teacher_final_state=teacher_agent.get_state()
            )
            
            if reflection_result:
                logger.log("SYSTEM", f"反思完成，生成经验: {len(reflection_result)}条")
            else:
                logger.log("SYSTEM", "反思完成，但未生成经验")
            
            # 经验库已由反思智能体自动保存
            logger.log("SYSTEM", "经验库已由反思智能体自动保存")
        else:
            logger.log("SYSTEM", "无对话历史，跳过反思")
        
    except KeyboardInterrupt:
        logger.log("SYSTEM", "用户中断，系统停止")
    except Exception as e:
        logger.log("ERROR", f"系统运行错误: {e}")
        import traceback
        logger.log("ERROR", f"错误详情: {traceback.format_exc()}")
    finally:
        # 停止系统
        logger.log("SYSTEM", "开始停止系统...")
        
        for agent in agents:
            agent.stop()
        event_bus.stop()
        
        logger.log("SYSTEM", "系统停止完成")
        
        # 经验库已由反思智能体自动保存
        logger.log("SYSTEM", "经验库已由反思智能体自动保存")
        
        # 保存完整对话记录到输出目录
        conversation_json_file = os.path.join(output_dir, f"conversation_log_{timestamp}.json")
        conversation_txt_file = os.path.join(output_dir, f"conversation_{timestamp}.txt")
        
        try:
            conversation_history = conversation_orchestrator.get_conversation_history(conversation_id)
            if conversation_history:
                # 保存JSON格式对话记录
                with open(conversation_json_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "conversation_id": conversation_id,
                        "problem_content": problem_content,
                        "start_time": datetime.now().isoformat(),
                        "total_rounds": len([msg for msg in conversation_history if msg.get("sender") == "student"]),
                        "final_understanding_level": student_agent.get_student_state().get("understanding_level", 0),
                        "conversation_history": conversation_history
                    }, f, ensure_ascii=False, indent=2)
                logger.log("SYSTEM", f"对话记录(JSON)已保存到: {conversation_json_file}")
                
                # 生成纯文本格式的对话文件
                with open(conversation_txt_file, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("智能教学对话记录\n")
                    f.write(f"对话时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"对话ID: {conversation_id}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write("题目内容:\n")
                    f.write(problem_content + "\n\n")
                    f.write("=" * 80 + "\n")
                    f.write("对话内容:\n")
                    f.write("=" * 80 + "\n\n")
                    
                    round_num = 0
                    for msg in conversation_history:
                        if msg.get("sender") == "student":
                            round_num += 1
                            f.write(f"第{round_num}轮对话:\n")
                            f.write(f"学生: {msg.get('content', '')}\n\n")
                        elif msg.get("sender") == "teacher":
                            f.write(f"教师: {msg.get('content', '')}\n\n")
                            f.write("-" * 50 + "\n\n")
                    
                    # 添加统计信息
                    f.write("=" * 80 + "\n")
                    f.write("对话统计:\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"总轮数: {round_num}\n")
                    f.write(f"最终理解程度: {student_agent.get_student_state().get('understanding_level', 0)}/10\n")
                    f.write(f"学生最终情绪: {student_agent.get_student_state().get('current_emotion', '未知')}\n")
                
                logger.log("SYSTEM", f"对话记录(文本)已保存到: {conversation_txt_file}")
                print(f"对话记录已保存到: {conversation_txt_file}")
            else:
                logger.log("SYSTEM", "无对话历史可保存")
        except Exception as e:
            logger.log("ERROR", f"保存对话记录失败: {e}")
        
        # 输出统计信息
        print(f"\n=== 系统运行完成 ===")
        print(f"运行时间戳: {timestamp}")
        print(f"输出目录: {output_dir}")
        print(f"系统日志: {system_log_file}")
        
        if os.path.exists(conversation_txt_file):
            print(f"对话记录(文本): {conversation_txt_file}")
        if os.path.exists(conversation_json_file):
            print(f"对话记录(JSON): {conversation_json_file}")
        
        # 输出智能体统计
        try:
            print(f"\n=== 智能体统计 ===")
            print(f"学生智能体状态: {student_agent.get_student_state()}")
            print(f"监控审核次数: {monitor_agent.get_review_statistics()['total_reviews']}")
            print(f"反思生成经验数: {reflection_agent.get_reflection_statistics()['experiences_generated']}")
            print(f"知识状态分析次数: {knowledge_agent.get_knowledge_statistics()['records_analyzed']}")
            print(f"跟踪知识点数: {knowledge_agent.get_knowledge_statistics()['total_knowledge_points']}")
            
            # 详细的经验库统计
            try:
                # 获取反思统计信息
                reflection_stats = reflection_agent.get_reflection_statistics()
                print(f"\n=== 反思统计 ===")
                print(f"生成经验数量: {reflection_stats.get('experiences_generated', 0)}")
                print(f"反思智能体状态: {reflection_stats.get('agent_status', '未知')}")
                
                # 尝试读取经验库文件统计
                experience_file = "experience_bank.json"
                if os.path.exists(experience_file):
                    try:
                        with open(experience_file, "r", encoding="utf-8") as f:
                            experiences = json.load(f)
                        print(f"\n=== 经验库统计 ===")
                        print(f"经验库文件: {experience_file}")
                        print(f"总经验数量: {len(experiences)}")
                        
                        # 统计主题分布
                        topics = {}
                        emotions = {}
                        for exp_key, exp_data in experiences.items():
                            topic = exp_data.get("problem_scenario", "未知")
                            topics[topic] = topics.get(topic, 0) + 1
                            
                            student_emotions = exp_data.get("student_emotions", ["未知"])
                            for emotion in student_emotions:
                                emotions[emotion] = emotions.get(emotion, 0) + 1
                        
                        if topics:
                            print(f"主题分布: {dict(list(topics.items())[:3])}")  # 只显示前3个
                        if emotions:
                            print(f"情绪分布: {dict(list(emotions.items())[:3])}")  # 只显示前3个
                    except Exception as e:
                        print(f"读取经验库文件失败: {e}")
                else:
                    print(f"\n=== 经验库统计 ===")
                    print(f"经验库文件: {experience_file} (不存在)")
                    print("总经验数量: 0")
            except Exception as e:
                print(f"获取统计信息失败: {e}")
        except Exception as e:
            logger.log("ERROR", f"获取统计信息失败: {e}")
        
        logger.log("SYSTEM", "AutoGen架构智能教学对话系统运行完成")


if __name__ == "__main__":
    main() 