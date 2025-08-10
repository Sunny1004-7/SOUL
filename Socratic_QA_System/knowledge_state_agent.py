# -*- coding: utf-8 -*-
"""
知识状态智能体：负责分析学生的知识掌握情况和学习进度
系统启动时生成一次完整的知识状态摘要，供后续所有教师LLM调用使用
"""
from core import BaseAgent, MessageType, Message, SampleDataManager
from typing import Dict, Any, Optional, List
import json


class KnowledgeStateAgent(BaseAgent):
    """知识状态智能体：分析学生知识掌握情况"""
    
    def __init__(self, name: str, llm_manager, logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        
        # 知识状态摘要缓存（系统启动时生成一次）
        self.knowledge_summary = None
        self.analysis_ready = False
        
        # 注册消息处理器
        self._register_knowledge_handlers()

    def initialize(self):
        """初始化知识状态智能体"""
        self.update_state("ready", True)
        self.update_state("knowledge_analysis_enabled", True)
        
        # 系统启动时立即生成知识状态摘要
        self._generate_initial_knowledge_summary()

    def _generate_initial_knowledge_summary(self):
        """系统启动时生成一次完整的知识状态摘要"""
        try:
            if self.logger:
                self.logger.log_agent_work("KNOWLEDGE_AGENT", "开始生成初始知识状态摘要", "系统启动时执行")
            
            # 使用统一的示例数据管理器获取习题记录
            exercise_records = SampleDataManager.get_simple_exercise_records()
            
            # 构建原始历史记录文本
            records_text = ""
            for i, record in enumerate(exercise_records, 1):
                records_text += f"记录{i}: 题目「{record['question']}」涉及知识点{record['knowledge_points']}，学生{'答对' if record['is_correct'] else '答错'}\n"
            
            # 只调用一次API生成知识状态摘要
            summary_prompt = f"""你是一名专业的学习分析师，需要基于学生的历史习题作答记录，生成一个简洁但全面的知识状态摘要。

请基于以下学生的历史习题作答记录，生成知识状态摘要：

{records_text}

请生成一个300字以内的摘要，包含：
1. 整体知识水平评估和描述
2. 主要知识领域和核心能力
3. 学习特点和思维模式
4. 教学策略建议重点
5. 发展潜力和改进方向

请直接返回摘要内容，不要包含"摘要："等前缀。这个摘要将用于后续所有教学对话中。"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的教学分析专家，专门生成知识状态摘要。请确保摘要全面、准确、实用。"},
                {"role": "user", "content": summary_prompt}
            ]
            
            summary_response = self.llm_manager.call_llm(messages, temperature=0.3, max_tokens=500)
            
            if summary_response:
                self.knowledge_summary = summary_response.strip()
                self.analysis_ready = True
                
                if self.logger:
                    self.logger.log_agent_work("KNOWLEDGE_AGENT", "知识状态摘要生成完成", f"摘要长度: {len(self.knowledge_summary)}字符")
            else:
                # 如果LLM调用失败，使用默认摘要
                self.knowledge_summary = self._get_default_knowledge_summary()
                self.analysis_ready = True
                
                if self.logger:
                    self.logger.log_agent_work("KNOWLEDGE_AGENT", "使用默认知识状态摘要", "LLM调用失败")
                    
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("KNOWLEDGE_AGENT", "初始知识状态摘要生成失败", f"错误: {str(e)}")
            
            # 异常情况下使用默认摘要
            self.knowledge_summary = self._get_default_knowledge_summary()
            self.analysis_ready = True

    def _get_default_knowledge_summary(self) -> str:
        """获取默认的知识状态摘要"""
        return """学生整体知识水平处于进阶阶段，具备扎实的基础概念理解能力。主要知识领域涵盖数学核心概念、逻辑思维和问题解决策略。学习特点表现为系统性思维，倾向于通过结构化方法解决问题。思维模式偏向线性逻辑，善于逐步分析复杂问题。教学策略建议重点关注概念深化、方法优化和思维拓展，通过苏格拉底式提问引导学生发现知识间的联系。学生具备良好的发展潜力，建议在保持现有优势的基础上，加强发散思维训练和实际应用能力培养。"""

    def get_knowledge_summary(self) -> str:
        """获取知识状态摘要（供教师智能体调用）"""
        if not self.analysis_ready:
            if self.logger:
                self.logger.log_agent_work("KNOWLEDGE_AGENT", "知识状态摘要未就绪", "返回默认摘要")
            return self._get_default_knowledge_summary()
        
        return self.knowledge_summary

    def _register_knowledge_handlers(self):
        """注册知识状态特定的消息处理器"""
        self.message_handlers[MessageType.ANALYSIS_REQUEST] = self._handle_analysis_request
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control

    def _handle_analysis_request(self, message: Message):
        """处理知识分析请求"""
        content = message.content
        conversation_id = content.get("conversation_id")
        
        # 直接返回已生成的知识状态摘要
        self.send_message(
            recipient="teacher",
            message_type=MessageType.TASK_RESPONSE,
            content={
                "conversation_id": conversation_id,
                "knowledge_summary": self.get_knowledge_summary(),
                "analysis_ready": self.analysis_ready
            }
        )

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        command = content.get("command", "")
        
        if command == "get_knowledge_summary":
            # 返回知识状态摘要
            self.send_message(
                recipient=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    "knowledge_summary": self.get_knowledge_summary(),
                    "analysis_ready": self.analysis_ready
                }
            )

