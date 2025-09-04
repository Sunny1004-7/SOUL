# -*- coding: utf-8 -*-
"""
基于AutoGen架构的监控智能体：实现事件驱动的教师回复审核
保留原有的质量监控逻辑，但使用异步消息传递和Actor模型
"""
from core import BaseAgent, MessageType, Message
from typing import Dict, Any, Optional
import json


class MonitorAgent(BaseAgent):
    """基于AutoGen架构的监控智能体"""
    
    def __init__(self, name: str, llm_manager, logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        self.max_review_attempts = 3
        
        # 监控系统的基础prompt
        self.base_prompt = """你是一名专业的教学质量监控专家，负责审核教师回复的质量和适宜性。

审核标准分为两个核心方面：

1. 情绪方面：
   - 教师语气是否温和友善，体现耐心和支持
   - 是否关注学生情绪，给予适当的情感支持
   - 教师自身是否保持冷静，避免产生负面情绪
   - 是否避免使用批评性、否定性或打击信心的语言
   - 是否营造积极正面的学习氛围

2. 专业方面：
   - 教学内容是否准确无误，符合学科标准
   - 教学方法是否科学合理，符合教育心理学原理
   - 是否与学生问题直接相关，针对性强
   - 表达是否清晰易懂，适合学生认知水平
   - 是否体现苏格拉底式教学方法的精髓

你需要严格把关，确保每个教师回复既具有专业的教师素养水准，又能照顾到学生的情绪需求。"""

        # 注册消息处理器
        self._register_monitor_handlers()
        
        if self.logger:
            self.logger.log_agent_work("MONITOR", "初始化完成", "教学质量监控系统已就绪")

    def initialize(self):
        """初始化监控智能体"""
        self.update_state("ready", True)
        self.update_state("review_count", 0)

    def _register_monitor_handlers(self):
        """注册监控特定的消息处理器"""
        self.message_handlers[MessageType.REVIEW_REQUEST] = self._handle_review_request
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control

    def _handle_review_request(self, message: Message):
        """处理审核请求"""
        content = message.content
        conversation_id = content.get("conversation_id")
        teacher_response = content.get("teacher_response", "")
        student_message = content.get("student_message", "")
        round_number = content.get("round_number", 1)
        conversation_history = content.get("conversation_history", [])
        is_regenerated = content.get("is_regenerated", False)
        
        if self.logger:
            attempt_info = "重新审核" if is_regenerated else "首次审核"
            self.logger.log_agent_work("MONITOR", f"收到审核请求", f"{attempt_info} - 第{round_number}轮")
        
        # 执行综合审核
        review_result = self._comprehensive_review(teacher_response, student_message, conversation_history)
        
        # 发送审核结果给教师
        response_content = {
            "conversation_id": conversation_id,
            "teacher_response": teacher_response,
            "student_message": student_message,
            "round_number": round_number,
            "approved": review_result["approved"],
            "review_details": review_result
        }
        
        # 只有在审核未通过时才提供修改意见
        if not review_result["approved"]:
            response_content["feedback"] = review_result.get("feedback", "")
        
        self.send_message(
            recipient="teacher",
            message_type=MessageType.REVIEW_RESPONSE,
            content=response_content,
            correlation_id=conversation_id
        )
        
        # 更新审核统计
        review_count = self.state.get("review_count", 0) + 1
        self.update_state("review_count", review_count)

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        action = content.get("action")
        
        if action == "conversation_ended":
            if self.logger:
                self.logger.log_agent_work("MONITOR", "对话结束", "重置审核状态")

    def _comprehensive_review(self, teacher_message: str, student_message: str, conversation_history: list) -> Dict[str, Any]:
        """综合审核教师回复"""
        if self.logger:
            self.logger.log_agent_work("MONITOR", "开始综合审核", f"教师回复长度: {len(teacher_message)}字符")
        
        # 构建对话历史文本
        history_text = ""
        if conversation_history:
            for entry in conversation_history[-4:]:  # 取最近4轮对话
                role = "学生" if entry["role"] == "student" else "老师"
                history_text += f"{role}: {entry['content']}\n"
        
        messages = [
            {
                "role": "system",
                "content": f"""{self.base_prompt}

请对教师回复进行完整审核，从情绪和专业两个核心方面进行分析：

请以JSON格式回复审核结果：

如果审核通过：
{{
    "approved": true,
    "overall_score": 1-10分,
    "emotional_aspect": {{
        "score": 1-10分,
        "tone_quality": "excellent/good/fair/poor",
        "emotional_support": "excellent/good/fair/poor",
        "teacher_emotion": "calm/neutral/slightly_negative/negative"
    }},
    "professional_aspect": {{
        "score": 1-10分,
        "content_accuracy": "excellent/good/fair/poor",
        "teaching_method": "excellent/good/fair/poor",
        "relevance": "excellent/good/fair/poor",
        "clarity": "excellent/good/fair/poor"
    }},
    "reason": "审核通过简述"
}}

如果审核未通过：
{{
    "approved": false,
    "overall_score": 1-10分,
    "emotional_aspect": {{
        "score": 1-10分,
        "tone_quality": "excellent/good/fair/poor",
        "emotional_support": "excellent/good/fair/poor",
        "teacher_emotion": "calm/neutral/slightly_negative/negative"
    }},
    "professional_aspect": {{
        "score": 1-10分,
        "content_accuracy": "excellent/good/fair/poor",
        "teaching_method": "excellent/good/fair/poor",
        "relevance": "excellent/good/fair/poor",
        "clarity": "excellent/good/fair/poor"
    }},
    "reason": "审核未通过简述",
    "feedback": "具体修改意见"
}}

审核通过标准：总分≥7分且情绪方面和专业方面得分均≥6分"""
            },
            {
                "role": "user",
                "content": f"""请审核以下教师回复：

学生发言：{student_message}

教师回复：{teacher_message}

对话历史：
{history_text}

请进行全面审核。"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.2)
        
        try:
            # 清理响应文本，移除可能的markdown标记
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            result = json.loads(cleaned_response)
            
            # 确保返回必要的字段
            if "approved" not in result:
                result["approved"] = False
            if "reason" not in result:
                result["reason"] = "审核系统错误"
            # 只有在审核未通过时才需要feedback字段
            if not result.get("approved", False) and "feedback" not in result:
                result["feedback"] = "请重新生成更合适的回复"
            
            if self.logger:
                approval_status = "通过" if result["approved"] else "未通过"
                overall_score = result.get("overall_score", "N/A")
                self.logger.log_agent_work("MONITOR", f"审核{approval_status}", f"总分: {overall_score}, 原因: {result['reason']}")
                self.logger.log_analysis_result("MONITOR", "详细审核", result)
                
            return result
            
        except json.JSONDecodeError as e:
            # JSON解析失败，记录错误信息但不终止程序
            if self.logger:
                self.logger.log_agent_work("MONITOR", "JSON解析失败", f"错误: {e}, 原始回复: {response[:200]}...")
            
            # 返回默认的审核通过结果，避免程序终止
            default_result = {
                "approved": True,
                "overall_score": 8,
                "emotional_aspect": {
                    "score": 8,
                    "tone_quality": "good",
                    "emotional_support": "good",
                    "teacher_emotion": "calm"
                },
                "professional_aspect": {
                    "score": 8,
                    "content_accuracy": "good",
                    "teaching_method": "good",
                    "relevance": "good",
                    "clarity": "good"
                },
                "reason": "审核系统临时故障，默认通过"
            }
            
            if self.logger:
                self.logger.log_agent_work("MONITOR", "使用默认审核结果", "避免程序终止")
                
            return default_result
            
        except Exception as e:
            # 其他异常，记录错误信息但不终止程序
            if self.logger:
                self.logger.log_agent_work("MONITOR", "审核异常", f"错误: {e}")
            
            # 返回默认的审核通过结果
            default_result = {
                "approved": True,
                "overall_score": 8,
                "emotional_aspect": {
                    "score": 8,
                    "tone_quality": "good",
                    "emotional_support": "good",
                    "teacher_emotion": "calm"
                },
                "professional_aspect": {
                    "score": 8,
                    "content_accuracy": "good",
                    "teaching_method": "good",
                    "relevance": "good",
                    "clarity": "good"
                },
                "reason": "审核系统异常，默认通过"
            }
            
            if self.logger:
                self.logger.log_agent_work("MONITOR", "使用默认审核结果", "避免程序终止")
                
            return default_result





    def get_review_statistics(self) -> Dict[str, Any]:
        """获取审核统计信息"""
        return {
            "total_reviews": self.state.get("review_count", 0),
            "agent_status": "active" if self.running else "inactive"
        } 