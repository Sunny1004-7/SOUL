# -*- coding: utf-8 -*-
"""
知识状态感知智能体：基于AutoGen架构的学生知识状态分析系统
通过分析学生历史习题作答记录，理解学生的知识掌握状况，
为教师Agent提供简洁的知识状态总结，实现个性化教学。
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core import BaseAgent, MessageType, Message
from student_data_loader import StudentDataLoader

@dataclass
class ExerciseRecord:
    """习题作答记录数据结构（三元组）"""
    question_content: str               # 习题内容
    knowledge_points: List[str]         # 习题所含知识点
    is_correct: bool                    # 学生是否答对

class KnowledgeStateAgent(BaseAgent):
    """知识状态感知智能体"""
    
    def __init__(self, name: str, llm_manager, logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        self.exercise_records: List[ExerciseRecord] = []
        
        # 注册消息处理器
        self._register_knowledge_handlers()
        
        # 加载历史数据
        self.load_student_records()
        
        if self.logger:
            self.logger.log_agent_work("KNOWLEDGE", "初始化完成", f"已加载{len(self.exercise_records)}条习题记录")

    def initialize(self):
        """初始化知识状态智能体"""
        self.update_state("ready", True)
        self.update_state("records_analyzed", 0)

    def _register_knowledge_handlers(self):
        """注册知识状态特定的消息处理器"""
        self.message_handlers[MessageType.TASK_REQUEST] = self._handle_analysis_request
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control

    def load_student_records(self):
        """统一从 StudentDataLoader 读取 stuRec_1000.csv 的第一个学生历史记录"""
        try:
            loader = StudentDataLoader()
            user_id = loader.get_first_student_id()
            if user_id is None:
                if self.logger:
                    self.logger.log_agent_work("KNOWLEDGE", "数据加载失败", "找不到学生ID")
                return
            # 获取该学生所有历史作答记录
            history = loader.get_student_history(user_id)
            for record_data in history:
                record = ExerciseRecord(**record_data)
                self.exercise_records.append(record)
            if self.logger:
                self.logger.log_agent_work("KNOWLEDGE", "数据加载成功", f"加载了{len(self.exercise_records)}条记录")
        except Exception as e:
            error_msg = f"加载学生记录失败: {e}"
            if self.logger:
                self.logger.log_agent_work("KNOWLEDGE", "数据加载失败", error_msg)
            print(error_msg)

    def _handle_analysis_request(self, message: Message):
        """处理知识状态分析请求（仅在对话开始前进行一次整体分析）"""
        content = message.content
        conversation_id = content.get("conversation_id")
        analysis_type = content.get("analysis_type", "comprehensive")
        
        if self.logger:
            self.logger.log_agent_work("KNOWLEDGE", "收到整体知识状态分析请求", f"类型: {analysis_type}")
        
        # 执行整体知识状态分析
        overall_analysis = self._generate_overall_knowledge_summary()
        
        # 发送分析结果给教师Agent
        self.send_message(
            recipient="teacher",
            message_type=MessageType.TASK_RESPONSE,
            content={
                "conversation_id": conversation_id,
                "overall_knowledge_summary": overall_analysis,
                "analysis_type": "overall_summary",
                "timestamp": datetime.now().isoformat()
            },
            correlation_id=conversation_id
        )
        
        # 更新统计
        analyzed_count = self.state.get("records_analyzed", 0) + 1
        self.update_state("records_analyzed", analyzed_count)

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        action = content.get("action")
        
        if action == "conversation_ended":
            if self.logger:
                self.logger.log_agent_work("KNOWLEDGE", "对话结束", "重置分析状态")

    def _generate_overall_knowledge_summary(self) -> str:
        """生成整体知识状态总结（直接让LLM基于原始记录分析）"""
        if self.logger:
            self.logger.log_agent_work("KNOWLEDGE", "开始生成整体知识状态总结", "基于原始历史习题记录")
        
        # 构建原始历史记录文本
        records_text = ""
        for i, record in enumerate(self.exercise_records, 1):
            records_text += f"记录{i}: 题目「{record.question_content}」涉及知识点{record.knowledge_points}，学生{'答对' if record.is_correct else '答错'}\n"
        
        messages = [
            {
                "role": "system",
                "content": """你是一名专业的学习分析师，需要基于学生的历史习题作答记录，生成一个结构化的知识状态分析报告。

你的任务是：
1. 仔细分析每条习题记录，理解题目内容、涉及的知识点和学生的作答情况
2. 综合评估学生的整体知识掌握水平
3. 识别学生的强项和薄弱点
4. 用自然语言描述学生的知识状态特点

请严格按照以下JSON格式输出分析结果，不要添加任何其他内容：

{
    "overall_assessment": {
        "total_exercises": 数字,
        "correct_rate": 数字(0-1之间的小数),
        "overall_level": "优秀/良好/中等/需要加强"
    },
    "knowledge_point_analysis": {
        "strong_points": ["知识点1", "知识点2", ...],
        "weak_points": ["知识点1", "知识点2", ...]
    },
    "detailed_analysis": {
        "strength_analysis": "详细分析学生的强项，用自然语言描述...",
        "weakness_analysis": "详细分析学生的薄弱点，用自然语言描述..."
    }
}

注意事项：
- 所有数值必须准确，基于实际记录计算
- 分析要客观、具体
- 确保JSON格式正确，可以被程序解析
- 只输出JSON，不要有任何其他内容"""
            },
            {
                "role": "user",
                "content": f"""请基于以下学生的历史习题作答记录，生成结构化的知识状态分析报告：

{records_text}

请严格按照指定的JSON格式输出分析结果。"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.3)
        
        if response:
            # 尝试解析JSON，确保格式正确
            try:
                json.loads(response)
                if self.logger:
                    self.logger.log_agent_work("KNOWLEDGE", "整体知识状态总结生成完成", f"总结长度: {len(response)}字符")
                    self.logger.log_analysis_result("KNOWLEDGE", "整体知识状态总结", response)
                return response
            except json.JSONDecodeError:
                if self.logger:
                    self.logger.log_agent_work("KNOWLEDGE", "JSON格式错误", "重新生成")
                # 如果JSON格式错误，重新生成
                return self._regenerate_json_summary(records_text)
        else:
            # LLM分析失败，直接终止程序并报错
            error_msg = "知识状态Agent的LLM分析失败，无法生成知识状态总结"
            if self.logger:
                self.logger.log_agent_work("KNOWLEDGE", "严重错误", error_msg)
            raise RuntimeError(error_msg)

    def _regenerate_json_summary(self, records_text: str) -> str:
        """重新生成JSON格式的总结"""
        messages = [
            {
                "role": "system",
                "content": """你之前的回答格式不正确。请严格按照JSON格式输出，不要包含任何其他文字。

输出格式：
{
    "overall_assessment": {
        "total_exercises": 数字,
        "correct_rate": 数字(0-1之间的小数),
        "overall_level": "优秀/良好/中等/需要加强"
    },
    "knowledge_point_analysis": {
        "strong_points": ["知识点1", "知识点2", ...],
        "weak_points": ["知识点1", "知识点2", ...]
    },
    "detailed_analysis": {
        "strength_analysis": "详细分析学生的强项...",
        "weakness_analysis": "详细分析学生的薄弱点..."
    }
}

只输出JSON，不要有任何其他内容。"""
            },
            {
                "role": "user",
                "content": f"基于以下记录重新生成JSON格式的分析：\n\n{records_text}"
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.1)
        if response:
            try:
                json.loads(response)
                return response
            except json.JSONDecodeError:
                pass
        
        # 重试也失败，直接终止程序并报错
        error_msg = "知识状态Agent的LLM分析重试失败，无法生成有效的JSON格式知识状态总结"
        if self.logger:
            self.logger.log_agent_work("KNOWLEDGE", "严重错误", error_msg)
        raise RuntimeError(error_msg)



    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """获取知识状态统计信息"""
        return {
            "total_exercise_records": len(self.exercise_records),
            "records_analyzed": self.state.get("records_analyzed", 0),
            "data_file": "stuRec_1000.csv" # 数据文件名称
        } 