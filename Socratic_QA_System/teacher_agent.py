# -*- coding: utf-8 -*-
"""
基于AutoGen架构的教师智能体：实现事件驱动的教学行为
保留原有的ICECoT思维链逻辑，但使用异步消息传递和Actor模型
每次调用LLM回复时都插入预先生成的知识状态摘要
"""
from core import BaseAgent, MessageType, Message
from typing import Dict, Any, Optional
import json


class TeacherAgent(BaseAgent):
    """基于AutoGen架构的教师智能体"""
    
    def __init__(self, name: str, llm_manager, logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        self.conversation_id = None
        self.conversation_history = []
        self.knowledge_summary = None  # 存储预生成的知识状态摘要
        self.analysis_ready = False
        
        # 教师角色的基础prompt
        self.base_prompt = """你是一名经验丰富、富有耐心的老师。

你的教学理念：
- 采用苏格拉底式教学方法，通过提问引导学生自己发现答案
- 关注学生的情绪状态，给予适当的情感支持
- 根据学生的知识掌握情况和理解程度调整教学策略
- 使用启发式教学，引导学生主动思考
- 语言温和友善，避免让学生感到压力
- 善于将复杂问题分解为易懂的步骤
- 基于学生的知识薄弱点进行针对性教学

苏格拉底式教学的核心原则：
- 不直接给出答案，而是通过精心设计的问题引导学生思考
- 从学生已知的知识出发，逐步引导到未知领域
- 鼓励学生表达自己的想法，即使想法不完整或错误
- 通过反问和追问帮助学生发现逻辑漏洞
- 让学生通过自己的思考得出结论，增强学习成就感
- 培养学生的批判性思维和独立思考能力

你需要在每次回复时遵循增强的ICECoT思维链：
1. 情绪分析：分析学生当前的情绪状态
2. 知识状态感知：结合学生的知识掌握情况
3. 意图推断：推断学生的真实需求和困难点
4. 策略选择：选择最适合的个性化教学策略
5. 响应生成：生成针对性的教学回复"""

        # 注册消息处理器
        self._register_teacher_handlers()

    def initialize(self):
        """初始化教师智能体"""
        self.update_state("ready", True)
        self.update_state("icecot_enabled", True)
        
        # 获取预生成的知识状态摘要
        self._request_knowledge_summary()

    def _request_knowledge_summary(self):
        """请求知识状态摘要"""
        try:
            # 向知识状态智能体请求摘要
            self.send_message(
                recipient="knowledge_state",
                message_type=MessageType.SYSTEM_CONTROL,
                content={
                    "command": "get_knowledge_summary"
                }
            )
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("TEACHER", "请求知识状态摘要失败", f"错误: {str(e)}")

    def _register_teacher_handlers(self):
        """注册教师特定的消息处理器"""
        self.message_handlers[MessageType.TASK_REQUEST] = self._handle_student_message
        self.message_handlers[MessageType.TASK_RESPONSE] = self._handle_knowledge_analysis
        self.message_handlers[MessageType.REVIEW_RESPONSE] = self._handle_monitor_feedback
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control

    def _handle_student_message(self, message: Message):
        """处理学生消息"""
        content = message.content
        self.conversation_id = content.get("conversation_id")
        student_message = content.get("student_message", "")
        round_number = content.get("round_number", 1)
        student_state = content.get("student_state", {})
        
        if self.logger:
            self.logger.log_agent_work("TEACHER", "收到学生消息", f"第{round_number}轮: {student_message[:50]}...")
        
        # 更新对话历史
        self.conversation_history.append({
            "role": "student",
            "content": student_message,
            "round": round_number,
            "state": student_state
        })
        
        # 先添加一个占位回复，表示正在生成
        self.conversation_history.append({
            "role": "teacher",
            "content": "正在生成回复...",
            "round": round_number,
            "timestamp": "now"
        })
        
        # 直接处理学生消息（使用预生成的知识状态摘要）
        try:
            teacher_response = self._execute_icecot_pipeline_with_knowledge(
                student_message, student_state, round_number
            )
            
            if teacher_response:
                # 更新占位回复为真实回复
                self._update_teacher_response(round_number, teacher_response)
                
                if self.logger:
                    self.logger.log_agent_work("TEACHER", "回复已生成并保存", f"第{round_number}轮回复: {teacher_response[:50]}...")
                
                # 可选：发送给监控智能体审核（但不等待结果）
                self._send_for_monitoring(teacher_response, student_message, round_number)
            else:
                # 如果生成失败，更新占位回复为错误信息
                self._update_teacher_response(round_number, "抱歉，回复生成失败，请重试。")
                
                if self.logger:
                    self.logger.log_agent_work("TEACHER", "回复生成失败", f"第{round_number}轮")
                    
        except Exception as e:
            # 异常情况下，更新占位回复为错误信息
            error_msg = f"抱歉，我在处理你的问题时遇到了一些困难：{str(e)}。请重新描述一下你的问题，我会尽力帮助你。"
            self._update_teacher_response(round_number, error_msg)
            
            if self.logger:
                self.logger.log_agent_work("TEACHER", "回复生成异常", f"第{round_number}轮错误: {str(e)}")

    def _update_teacher_response(self, round_number: int, new_content: str):
        """更新教师回复内容"""
        for i, msg in enumerate(self.conversation_history):
            if (msg.get("role") == "teacher" and 
                msg.get("round") == round_number and 
                msg.get("content") == "正在生成回复..."):
                self.conversation_history[i]["content"] = new_content
                self.conversation_history[i]["timestamp"] = "now"
                break

    def _send_for_monitoring(self, teacher_response: str, student_message: str, round_number: int):
        """发送回复给监控智能体审核"""
        try:
            self.send_message(
                recipient="monitor",
                message_type=MessageType.REVIEW_REQUEST,
                content={
                    "conversation_id": self.conversation_id,
                    "teacher_response": teacher_response,
                    "student_message": student_message,
                    "round_number": round_number
                }
            )
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("TEACHER", "发送监控审核失败", f"错误: {str(e)}")

    def _handle_knowledge_analysis(self, message: Message):
        """处理知识状态分析结果"""
        content = message.content
        
        # 检查是否是知识状态摘要响应
        if "knowledge_summary" in content:
            self.knowledge_summary = content.get("knowledge_summary")
            self.analysis_ready = content.get("analysis_ready", False)
            
            if self.logger:
                self.logger.log_agent_work("TEACHER", "收到知识状态摘要", f"摘要长度: {len(self.knowledge_summary) if self.knowledge_summary else 0}字符")
        else:
            # 处理其他类型的知识分析响应
            knowledge_analysis = content.get("knowledge_analysis", {})
            student_message = content.get("student_message", "")
            
            if self.logger:
                self.logger.log_agent_work("TEACHER", "收到知识状态分析", f"分析结果: {len(knowledge_analysis)}项")

    def _handle_monitor_feedback(self, message: Message):
        """处理监控智能体的反馈"""
        content = message.content
        feedback = content.get("feedback", "")
        approved = content.get("approved", False)
        round_number = content.get("round_number", 0)
        
        if approved:
            # 监控通过，发送回复
            self._send_approved_response(content.get("teacher_response", ""), round_number)
        else:
            # 监控不通过，重新生成回复
            self._regenerate_response(content.get("student_message", ""), feedback, round_number)

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        command = content.get("command", "")
        
        if self.logger:
            self.logger.log_agent_work("TEACHER", "收到系统控制", f"命令: {command}")
        
        if command == "end_conversation":
            self._cleanup_conversation()

    def _execute_icecot_pipeline_with_knowledge(self, student_message: str, student_state: Dict[str, Any], round_number: int) -> str:
        """执行增强的ICECoT思维链流程（使用预生成的知识状态摘要）"""
        try:
            # 1. 情绪分析
            emotion_analysis = self._analyze_student_emotion(student_message, student_state)
            
            # 2. 意图推断（结合预生成的知识状态摘要）
            intention_analysis = self._infer_student_intention_with_knowledge(
                student_message, emotion_analysis, student_state
            )
            
            # 3. 策略选择
            strategy_selection = self._select_teaching_strategy_with_knowledge(
                emotion_analysis, intention_analysis
            )
            
            # 4. 响应生成（插入预生成的知识状态摘要）
            teacher_response = self._generate_teaching_response_with_knowledge(
                student_message, emotion_analysis, intention_analysis, 
                strategy_selection, round_number
            )
            
            return teacher_response
            
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("TEACHER", "ICECoT流程执行失败", f"错误: {str(e)}")
            return "抱歉，我在处理你的问题时遇到了一些困难。请重新描述一下你的问题，我会尽力帮助你。"

    def _analyze_student_emotion(self, student_message: str, student_state: Dict[str, Any]) -> Dict[str, Any]:
        """分析学生情绪状态"""
        try:
            prompt = f"""
请分析以下学生的情绪状态：

学生消息：{student_message}
学生状态：{json.dumps(student_state, ensure_ascii=False)}

请从以下维度分析：
1. 主要情绪：焦虑、困惑、自信、兴奋、沮丧等
2. 情绪强度：1-5分（1=很轻微，5=很强烈）
3. 学习动机：高、中、低
4. 压力水平：高、中、低
5. 情绪变化趋势：上升、稳定、下降

请以JSON格式返回分析结果。
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的情绪分析专家，专门分析学生的学习情绪状态。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_manager.call_llm(messages, temperature=0.3, max_tokens=500)
            
            if response:
                try:
                    emotion_analysis = json.loads(response)
                    return emotion_analysis
                except json.JSONDecodeError:
                    # 如果JSON解析失败，返回默认分析
                    return self._get_default_emotion_analysis()
            else:
                return self._get_default_emotion_analysis()
                
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("TEACHER", "情绪分析失败", f"错误: {str(e)}")
            return self._get_default_emotion_analysis()

    def _get_default_emotion_analysis(self) -> Dict[str, Any]:
        """获取默认的情绪分析结果"""
        return {
            "main_emotion": "neutral",
            "emotion_intensity": 3,
            "learning_motivation": "medium",
            "stress_level": "medium",
            "emotion_trend": "stable"
        }

    def _infer_student_intention_with_knowledge(self, student_message: str, emotion_analysis: Dict[str, Any], student_state: Dict[str, Any]) -> Dict[str, Any]:
        """推断学生意图（结合预生成的知识状态摘要）"""
        try:
            # 构建包含知识状态摘要的提示词
            knowledge_context = ""
            if self.knowledge_summary:
                knowledge_context = f"\n\n学生知识状态摘要：{self.knowledge_summary}"
            
            prompt = f"""
请分析以下学生的真实意图和需求：

学生消息：{student_message}
情绪分析：{json.dumps(emotion_analysis, ensure_ascii=False)}
学生状态：{json.dumps(student_state, ensure_ascii=False)}{knowledge_context}

请从以下维度分析：
1. learning_goal：学生的学习目标（如：理解概念、掌握方法、解决问题、验证想法等）
2. difficulty_type：遇到的困难类型（如：概念不清、关系不明、方法不会、逻辑混乱等）
3. need_level：需求层次（如：认知需求、情感需求、技能需求、应用需求等）
4. learning_preference：学习偏好（如：详细讲解、启发引导、实例演示、步骤分解等）
5. analysis：综合分析（对学生当前状态和需求的详细分析说明）

请严格按照以下JSON格式返回分析结果：
{
  "learning_goal": "学习目标描述",
  "difficulty_type": "困难类型描述",
  "need_level": "需求层次描述",
  "learning_preference": "学习偏好描述",
  "analysis": "综合分析说明"
}
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的教学意图分析专家，专门分析学生的学习意图和需求。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_manager.call_llm(messages, temperature=0.3, max_tokens=600)
            
            if response:
                try:
                    intention_analysis = json.loads(response)
                    return intention_analysis
                except json.JSONDecodeError:
                    return self._get_default_intention_analysis()
            else:
                return self._get_default_intention_analysis()
                
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("TEACHER", "意图推断失败", f"错误: {str(e)}")
            return self._get_default_intention_analysis()

    def _get_default_intention_analysis(self) -> Dict[str, Any]:
        """获取默认的意图分析结果"""
        return {
            "learning_goal": "理解概念",
            "difficulty_type": "概念不清",
            "need_level": "认知需求",
            "learning_preference": "启发引导",
            "analysis": "学生需要通过引导来理解相关概念，建议采用启发式教学方法帮助其澄清理解。"
        }

    def _select_teaching_strategy_with_knowledge(self, emotion_analysis: Dict[str, Any], intention_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """选择教学策略（结合预生成的知识状态摘要）"""
        try:
            # 构建包含知识状态摘要的提示词
            knowledge_context = ""
            if self.knowledge_summary:
                knowledge_context = f"\n\n学生知识状态摘要：{self.knowledge_summary}"
            
            prompt = f"""
基于以下分析，请选择最适合的教学策略：

情绪分析：{json.dumps(emotion_analysis, ensure_ascii=False)}
意图分析：{json.dumps(intention_analysis, ensure_ascii=False)}{knowledge_context}

可选择的教学策略包括：
1. 启发式策略：通过苏格拉底式提问引导学生思考
2. 认知支持策略：概念分解、知识连接、类比教学
3. 情感支持策略：鼓励支持、减压引导
4. 技能训练策略：方法指导、步骤演示
5. 反思策略：错误分析、思维训练
6. 应用策略：实践应用、情境教学
...其它你认为科学合理且适用的教学策略

请分析并选择教学策略，包括：
- primary_strategy：主要策略
- secondary_strategy：辅助策略
- approach：具体实施方法
- tone：交流语调风格
- key_points：关键要点列表
- rationale：选择理由说明

请严格按照以下JSON格式返回策略选择：
{
  "primary_strategy": "主要策略名称",
  "secondary_strategy": "辅助策略名称",
  "approach": "具体实施方法描述",
  "tone": "交流语调风格",
  "key_points": [
    "关键要点1",
    "关键要点2"
  ],
  "rationale": "选择理由的详细说明"
}
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的教学策略专家，专门为不同情况选择最适合的教学策略。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_manager.call_llm(messages, temperature=0.4, max_tokens=500)
            
            if response:
                try:
                    strategy_selection = json.loads(response)
                    return strategy_selection
                except json.JSONDecodeError:
                    return self._get_default_strategy_selection()
            else:
                return self._get_default_strategy_selection()
                
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("TEACHER", "策略选择失败", f"错误: {str(e)}")
            return self._get_default_strategy_selection()

    def _get_default_strategy_selection(self) -> Dict[str, Any]:
        """获取默认的策略选择结果"""
        return {
            "primary_strategy": "启发式策略",
            "secondary_strategy": "认知支持策略",
            "approach": "通过苏格拉底式提问引导学生思考，结合概念分解帮助理解",
            "tone": "鼓励和引导",
            "key_points": [
                "引导学生主动思考",
                "分解复杂概念为简单部分"
            ],
            "rationale": "基于学生的理解需求，采用启发式提问引导思考，同时提供认知支持帮助理解"
        }

    def _generate_teaching_response_with_knowledge(self, student_message: str, emotion_analysis: Dict[str, Any], 
                                                 intention_analysis: Dict[str, Any], strategy_selection: Dict[str, Any], 
                                                 round_number: int) -> str:
        """生成教学回复（插入预生成的知识状态摘要）"""
        try:
            # 构建包含知识状态摘要的完整提示词
            knowledge_context = ""
            if self.knowledge_summary:
                knowledge_context = f"\n\n学生知识状态摘要：{self.knowledge_summary}"
            
            prompt = f"""
你是一名经验丰富的老师，现在需要回复学生的问题。

学生消息：{student_message}
对话轮次：第{round_number}轮

分析结果：
- 情绪状态：{json.dumps(emotion_analysis, ensure_ascii=False)}
- 学习意图：{json.dumps(intention_analysis, ensure_ascii=False)}
- 教学策略：{json.dumps(strategy_selection, ensure_ascii=False)}{knowledge_context}

请根据以上分析，生成一个符合苏格拉底教学范式的回复。要求：

1. 遵循选择的教学策略
2. 使用苏格拉底式提问，引导学生思考
3. 关注学生的情绪状态，给予适当的情感支持
4. 语言温和友善，避免让学生感到压力
5. 回复长度适中，不要过于冗长
6. 体现个性化教学，针对学生的具体情况
7. 结合学生的知识状态特点，提供有针对性的指导

请直接返回回复内容，不要包含任何格式标记。
"""
            
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_manager.call_llm(messages, temperature=0.7, max_tokens=800)
            
            if response:
                return response.strip()
            else:
                return "我理解你的问题，让我用另一种方式来帮助你思考。你能告诉我你目前对这个问题的理解吗？"
                
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("TEACHER", "回复生成失败", f"错误: {str(e)}")
            return "抱歉，我在生成回复时遇到了一些问题。请重新描述一下你的问题，我会尽力帮助你。"

    def _send_approved_response(self, teacher_response: str, round_number: int):
        """发送审核通过的回复"""
        if self.logger:
            self.logger.log_agent_work("TEACHER", "发送审核通过回复", f"第{round_number}轮回复已发送")
        
        # 更新对话历史
        self.conversation_history.append({
            "role": "teacher",
            "content": teacher_response,
            "round": round_number,
            "timestamp": "now"
        })
        
        # 发送系统控制消息，通知回复已发送
        self.send_message(
            recipient="system",
            message_type=MessageType.SYSTEM_CONTROL,
            content={
                "command": "response_sent",
                "conversation_id": self.conversation_id,
                "round_number": round_number,
                "teacher_response": teacher_response
            }
        )

    def _regenerate_response(self, student_message: str, feedback: str, round_number: int):
        """根据反馈重新生成回复"""
        if self.logger:
            self.logger.log_agent_work("TEACHER", "重新生成回复", f"根据反馈: {feedback}")
        
        # 根据反馈重新生成回复
        new_response = self._execute_icecot_pipeline_with_knowledge(
            student_message, {}, round_number
        )
        
        if new_response:
            # 再次发送给监控智能体审核
            self.send_message(
                recipient="monitor",
                message_type=MessageType.REVIEW_REQUEST,
                content={
                    "conversation_id": self.conversation_id,
                    "teacher_response": new_response,
                    "student_message": student_message,
                    "round_number": round_number
                }
            )

    def _cleanup_conversation(self):
        """清理对话资源"""
        if self.logger:
            self.logger.log_agent_work("TEACHER", "清理对话", "对话资源已清理")
        
        self.conversation_id = None
        self.conversation_history = []
