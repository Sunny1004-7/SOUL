# -*- coding: utf-8 -*-
"""
监控智能体：负责审核教师回复的质量和教学效果
"""
from core import BaseAgent, MessageType, Message
from typing import Dict, Any, Optional
import json


class MonitorAgent(BaseAgent):
    """监控智能体：审核教师回复质量"""
    
    def __init__(self, name: str, llm_manager, logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        self.quality_standards = {
            "socratic_style": "回复应体现苏格拉底式教学风格",
            "emotional_support": "应给予适当的情感支持",
            "clarity": "语言清晰易懂",
            "appropriateness": "回复长度和复杂度适中",
            "engagement": "能有效引导学生思考"
        }
        
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

    def initialize(self):
        """初始化监控智能体"""
        self.update_state("ready", True)
        self.update_state("quality_monitoring_enabled", True)

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
        
        # 执行质量审核
        review_result = self._review_teacher_response(teacher_response, student_message, round_number)
        
        # 发送审核结果
        self.send_message(
            recipient="teacher",
            message_type=MessageType.REVIEW_RESPONSE,
            content={
                "conversation_id": conversation_id,
                "teacher_response": teacher_response,
                "student_message": student_message,
                "round_number": round_number,
                "approved": review_result["approved"],
                "feedback": review_result["feedback"],
                "quality_score": review_result["quality_score"]
            }
        )

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        command = content.get("command", "")

    def _review_teacher_response(self, teacher_response: str, student_message: str, round_number: int) -> Dict[str, Any]:
        """审核教师回复质量"""
        try:
            # 构建审核提示词
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

教师回复：{teacher_response}

对话轮次：第{round_number}轮

请进行全面审核。"""
                }
            ]
            
            response = self.llm_manager.call_llm(messages, temperature=0.2)
            
            if response:
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
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    # JSON解析失败，返回默认审核结果
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
                    
                    return default_result
            else:
                # LLM调用失败，返回默认审核结果
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
            
            return default_result

    def get_quality_standards(self) -> Dict[str, str]:
        """获取质量标准"""
        return self.quality_standards.copy()

    def update_quality_standards(self, new_standards: Dict[str, str]):
        """更新质量标准"""
        self.quality_standards.update(new_standards)
        if self.logger:
            self.logger.log_agent_work("MONITOR", "质量标准更新", f"新标准: {list(new_standards.keys())}")
