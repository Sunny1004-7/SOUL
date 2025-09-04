# -*- coding: utf-8 -*-
"""
基于AutoGen架构的教师智能体：实现事件驱动的教学行为
保留原有的ICECoT思维链逻辑，但使用异步消息传递和Actor模型
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
        self.overall_knowledge_summary = None  # 存储整体知识状态总结
        self.waiting_for_knowledge = False      # 标记是否正在等待知识分析
        self.pending_student_message = None     # 暂存学生消息
        
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
        
        if self.logger:
            self.logger.log_agent_work("TEACHER", "初始化完成", "知识感知型ICECoT教师智能体已就绪")

    def initialize(self):
        """初始化教师智能体"""
        self.update_state("ready", True)
        self.update_state("icecot_enabled", True)

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
            self.logger.log_agent_work("TEACHER", f"收到学生消息", f"第{round_number}轮: {student_message[:50]}...")
        
        # 更新对话历史
        self.conversation_history.append({
            "role": "student",
            "content": student_message,
            "round": round_number,
            "state": student_state
        })
        
        # 直接处理学生消息（不再等待知识状态分析）
        teacher_response = self._execute_icecot_pipeline_with_knowledge(
            student_message, student_state, round_number
        )
        
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

    def _handle_knowledge_analysis(self, message: Message):
        """处理整体知识状态总结（仅在对话开始前调用一次）"""
        content = message.content
        overall_summary = content.get("overall_knowledge_summary", "")
        
        if self.logger:
            self.logger.log_agent_work("TEACHER", "收到整体知识状态总结", f"总结长度: {len(overall_summary)}字符")
        
        # 存储整体知识状态总结
        self.overall_knowledge_summary = overall_summary
        self.waiting_for_knowledge = False
        
        if self.pending_student_message:
            student_data = self.pending_student_message
            self.pending_student_message = None
            
            teacher_response = self._execute_icecot_pipeline_with_knowledge(
                student_data["student_message"], 
                student_data["student_state"], 
                student_data["round_number"]
            )
            
            if teacher_response:
                # 发送给监控智能体审核
                self.send_message(
                    recipient="monitor",
                    message_type=MessageType.REVIEW_REQUEST,
                    content={
                        "conversation_id": self.conversation_id,
                        "teacher_response": teacher_response,
                        "student_message": student_data["student_message"],
                        "round_number": student_data["round_number"],
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
            self.current_knowledge_analysis = None
            self.waiting_for_knowledge = False
            self.pending_student_message = None
            if self.logger:
                self.logger.log_agent_work("TEACHER", "对话结束", "清理状态完成")

    def _execute_icecot_pipeline_with_knowledge(self, student_message: str, student_state: Dict[str, Any], round_number: int) -> str:
        """执行包含知识状态总结的ICECoT思维链流程"""
        if self.logger:
            self.logger.log_agent_work("TEACHER", "开始ICECoT流程（含知识状态）", f"第{round_number}轮分析")
        
        # 第一步：情绪分析
        emotion_analysis = self._analyze_student_emotion(student_message, student_state)
        
        # 第二步：意图推断
        intention_analysis = self._infer_student_intention_with_knowledge(student_message, emotion_analysis, student_state)
        
        # 第三步：策略选择
        strategy_selection = self._select_teaching_strategy_with_knowledge(emotion_analysis, intention_analysis)
        
        # 第四步：回复生成
        teacher_response = self._generate_teaching_response_with_knowledge(
            student_message, emotion_analysis, intention_analysis, strategy_selection, round_number
        )
        
        if self.logger:
            self.logger.log_agent_work("TEACHER", "ICECoT流程完成（含知识状态）", f"生成回复长度: {len(teacher_response)}字符")
        
        return teacher_response

    def _analyze_student_emotion(self, student_message: str, student_state: Dict[str, Any]) -> Dict[str, Any]:
        """分析学生情绪状态"""
        if self.logger:
            self.logger.log_agent_work("TEACHER", "开始情绪分析", f"基于学生发言内容")
        
        messages = [
            {
                "role": "system",
                "content": """你是一名专业的情绪分析专家。请基于学生的发言内容分析其情绪状态。

分析维度：
1. 主要情绪：困惑、焦虑、沮丧、紧张、开心、自信等
2. 情绪强度：1-10分（1=轻微，10=强烈）
3. 学习态度：积极、消极、中性
4. 自信程度：1-10分（1=完全没信心，10=非常自信）

请以JSON格式回复：
{
    "primary_emotion": "主要情绪",
    "emotion_intensity": 强度分数,
    "learning_attitude": "学习态度",
    "confidence_level": 自信程度分数,
    "analysis": "详细分析"
}"""
            },
            {
                "role": "user",
                "content": f"""学生发言：{student_message}

请基于学生的发言内容分析其情绪状态。"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.3, max_tokens=300)
        try:
            result = json.loads(response)
            if self.logger:
                self.logger.log_analysis_result("TEACHER", "情绪分析", result)
            return result
        except:
            default_result = {
                "primary_emotion": "困惑",
                "emotion_intensity": 5,
                "learning_attitude": "中性",
                "confidence_level": 5,
                "analysis": "情绪分析失败，使用默认结果"
            }
            if self.logger:
                self.logger.log_agent_work("TEACHER", "情绪分析失败", "使用默认结果")
            return default_result

    def _infer_student_intention_with_knowledge(self, student_message: str, emotion_analysis: Dict[str, Any], student_state: Dict[str, Any]) -> Dict[str, Any]:
        """推断学生意图（结合预生成的知识状态摘要）"""
        try:
            # 构建包含知识状态摘要的提示词
            knowledge_context = ""
            if self.overall_knowledge_summary:
                knowledge_context = f"\n\n学生知识状态摘要：{self.overall_knowledge_summary}"
            
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
                    if self.logger:
                        self.logger.log_analysis_result("TEACHER", "意图推断（含知识状态）", intention_analysis)
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
            if self.overall_knowledge_summary:
                knowledge_context = f"\n\n学生知识状态摘要：{self.overall_knowledge_summary}"
            
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
                    if self.logger:
                        self.logger.log_analysis_result("TEACHER", "策略选择（含知识状态）", strategy_selection)
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
        """生成教学回复（结合知识状态总结）"""
        if self.logger:
            strategy = strategy_selection.get('strategy', '未知')
            self.logger.log_agent_work("TEACHER", "开始生成回复（含知识状态）", f"策略: {strategy}")
        
        messages = [
            {
                "role": "system",
                "content": f"""{self.base_prompt}

当前分析结果：
情绪状态：{emotion_analysis}
学习意图：{intention_analysis}
教学策略：{strategy_selection}

学生知识状态总结：
{self.overall_knowledge_summary or "暂无知识状态信息"}

请根据以上分析，生成一个采用苏格拉底式教学方法的针对性回复。回复要求：
1. 体现选定的教学策略和语调
2. 针对学生的具体情绪和需求
3. 结合学生的知识掌握情况，重点关注薄弱点
4. 利用学生的知识强项建立学习信心
5. 语言自然流畅，符合的身份
6. 长度适中，避免冗余发言，不要说废话，直接针对学生问题回答
8. 采用苏格拉底式教学方法：
   - 不直接给出答案，而是通过精心设计的问题引导学生思考
   - 从学生已知的知识出发，逐步引导到未知领域
   - 鼓励学生表达自己的想法，即使想法不完整或错误
   - 通过反问和追问帮助学生发现逻辑漏洞
   - 让学生通过自己的思考得出结论，增强学习成就感
"""
            },
            {
                "role": "user",
                "content": f"""学生发言：{student_message}

这是第{round_number}轮对话。

请生成相应的采用苏格拉底式教学方法的针对性教学回复。"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.7, max_tokens=300)
        result = response if response else "我理解你的困惑，让我们一起来解决这个问题。"
        
        if self.logger:
            if response:
                self.logger.log_agent_work("TEACHER", "教学回复生成成功（含知识状态）", f"回复长度: {len(result)}字符")
            else:
                self.logger.log_agent_work("TEACHER", "教学回复生成失败", "使用默认回复")
        
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
