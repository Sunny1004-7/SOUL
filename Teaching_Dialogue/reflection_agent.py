# -*- coding: utf-8 -*-
"""
基于AutoGen架构的反思智能体：实现事件驱动的教学经验提炼
保留原有的反思分析逻辑，但使用异步消息传递和Actor模型
"""
from core import BaseAgent, MessageType, Message
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime


class ReflectionAgent(BaseAgent):
    """基于AutoGen架构的反思智能体"""
    
    def __init__(self, name: str, llm_manager, experience_file="experience_bank.json", logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        self.experience_file = experience_file
        
        # 注册消息处理器
        self._register_reflection_handlers()
        
        # 初始化
        self.initialize()
        
        if self.logger:
            self.logger.log_agent_work("REFLECTION", "初始化完成", "教育反思系统已就绪")

    def initialize(self):
        """初始化反思智能体"""
        self.update_state("ready", True)
        self.update_state("experiences_generated", 0)
        self.experiences_generated = 0  # 确保实例属性存在

    def _register_reflection_handlers(self):
        """注册反思特定的消息处理器"""
        self.message_handlers[MessageType.REFLECTION_REQUEST] = self._handle_reflection_request
        self.message_handlers[MessageType.SYSTEM_CONTROL] = self._handle_system_control

    def _handle_reflection_request(self, message: Message):
        """处理反思请求（仅在对话结束后进行一次总结反思）"""
        content = message.content
        conversation_id = content.get("conversation_id")
        conversation_history = content.get("conversation_history", [])
        is_conversation_end = content.get("is_conversation_end", False)
        
        if self.logger:
            if is_conversation_end:
                self.logger.log_agent_work("REFLECTION", "收到对话结束反思请求", "开始总结反思")
            else:
                self.logger.log_agent_work("REFLECTION", "收到单轮反思请求", "暂不处理，等待对话结束")
                return
        
        # 只在对话结束时执行总结反思
        if is_conversation_end:
            reflection_result = self._perform_conversation_summary_reflection(conversation_history)
        
        # 发送反思结果给协调器（用于日志记录）
        self.send_message(
            recipient="orchestrator",
            message_type=MessageType.REFLECTION_RESPONSE,
            content={
                "conversation_id": conversation_id,
                    "reflection_result": reflection_result,
                    "is_conversation_end": True
            },
            correlation_id=conversation_id
        )
        
        # 更新统计
        exp_count = self.state.get("experiences_generated", 0) + 1
        self.update_state("experiences_generated", exp_count)

    def _handle_system_control(self, message: Message):
        """处理系统控制消息"""
        content = message.content
        action = content.get("action")
        
        if action == "conversation_ended":
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "对话结束", "反思状态重置")

    def _perform_conversation_summary_reflection(self, conversation_history: list) -> Dict[str, Any]:
        """执行对话总结反思（仅在对话结束后调用一次）"""
        if self.logger:
            total_rounds = len([x for x in conversation_history if x.get("role") == "student"])
            self.logger.log_agent_work("REFLECTION", "开始对话总结反思", f"总轮数: {total_rounds}")
        
        try:
            # 构建完整对话文本
            conversation_text = ""
            for msg in conversation_history:
                role = "学生" if msg.get("sender") == "student" else "老师"
                content = msg.get("content", "")
                conversation_text += f"{role}: {content}\n"
            
            # 通过LLM分析对话并生成标准格式的经验
            experience = self._generate_standard_experience_from_conversation(conversation_text, conversation_history)
            
            if not experience:
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "经验生成失败", "程序终止")
                raise Exception("反思智能体经验生成失败：LLM未返回有效经验")
            
            # 生成经验键
            problem_scenario = experience.get("problem_scenario", "未知问题")
            student_emotions = experience.get("student_emotions", ["未知"])
            emotion_str = "_".join(student_emotions[:2]) if student_emotions else "未知"
            experience_key = f"{problem_scenario[:10]}_{emotion_str}_{len(conversation_history)}"
            
            # 直接存储到JSON文件
            success = self._store_experience_to_json(experience_key, experience)
            
            if success:
                self.experiences_generated += 1
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "经验存储成功", f"键: {experience_key}")
                
                result = {
                    "success": True,
                    "summary": f"对话总结反思完成，已存储经验：{experience_key}",
                    "experience_key": experience_key,
                    "experience": experience,
                    "storage_success": True
                }
            else:
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "经验存储失败", f"键: {experience_key}")
                
                result = {
                    "success": False,
                    "summary": f"对话总结反思完成，但经验存储失败：{experience_key}",
                    "experience_key": experience_key,
                    "experience": experience,
                    "storage_success": False
                }
            
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "对话总结反思完成", f"结果: {result['summary']}")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "summary": f"对话总结反思过程出现错误：{str(e)}",
                "experience_key": None,
                "experience": None,
                "storage_success": False
            }
            
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "对话总结反思失败", f"错误: {str(e)}")
            
            return error_result

    def _store_experience_to_json(self, experience_key: str, experience_data: Dict[str, Any]) -> bool:
        """将经验直接存储到JSON文件"""
        try:
            # 读取现有经验数据
            existing_experiences = {}
            if os.path.exists(self.experience_file):
                try:
                    with open(self.experience_file, "r", encoding="utf-8") as f:
                        existing_experiences = json.load(f)
                except (json.JSONDecodeError, Exception):
                    existing_experiences = {}
            
            # 添加新经验
            existing_experiences[experience_key] = experience_data
            
            # 保存到文件
            with open(self.experience_file, "w", encoding="utf-8") as f:
                json.dump(existing_experiences, f, ensure_ascii=False, indent=2)
            
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "经验存储到JSON成功", 
                    f"文件: {self.experience_file}, 键: {experience_key}")
            
            return True
            
        except Exception as e:
            error_msg = f"存储经验到JSON失败: {e}"
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "JSON存储失败", error_msg)
            return False

    def get_reflection_statistics(self) -> Dict[str, Any]:
        """获取反思统计信息"""
        return {
            "experiences_generated": self.state.get("experiences_generated", 0),
            "agent_status": "active" if self.running else "inactive"
        }

    def _generate_standard_experience_from_conversation(self, conversation_text: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """通过LLM分析对话，生成标准格式的经验数据"""
        if self.logger:
            self.logger.log_agent_work("REFLECTION", "开始LLM经验生成", "分析完整对话")
        
        # 提取学生的第一句发言来分析初始人格
        first_student_message = ""
        for msg in conversation_history:
            if msg.get("sender") == "student":
                first_student_message = msg.get("content", "")
                break
        
        messages = [
            {
                "role": "system",
                "content": """你是一名教育研究专家和反思性教学实践者，擅长从完整教学对话中提炼有价值的教育经验。

你的任务是对一份完整的师生教学对话过程进行总结反思，识别成功的教学策略和可改进之处，并总结为可复用的教学经验。

反思维度：
1. 情绪互动分析：师生情绪变化、情感支持效果
2. 教学策略效果：所采用策略的适宜性和有效性
3. 学习进展评估：学生理解程度变化、困惑解决情况
4. 沟通质量分析：表达清晰度、互动流畅性
5. 个性化程度：是否适合学生特点和需求

你需要从教育心理学、认知科学、教学法等角度进行专业分析，提炼出具有普适性的教学经验。

请仔细分析对话内容，重点关注：
1. 教学问题场景：具体是什么教学内容
2. 学生初始人格：基于学生第一句发言分析学生的初始性格特征和学习态度
3. 学生最终理解程度：对话结束时的理解水平（需要科学严谨的评估）
4. 教师核心策略：最有效的教学策略（最多3个）
5. 整体教学效果：基于学生进步情况评估
6. 适用条件：这种教学方法的适用场景

然后生成标准格式的经验数据，必须严格按照以下JSON格式：

{
    "problem_scenario": "问题场景描述（如：二次方程求解教学）",
    "student_emotions": ["学生初始人格特征1", "学生初始人格特征2"],
    "student_understanding_level": 最终理解程度分数（0-10的浮点数）,
    "teacher_strategies": ["核心策略1", "核心策略2", "核心策略3"],
    "effectiveness_score": 整体效果评分（0-10的浮点数）,
    "applicable_conditions": ["适用条件1", "适用条件2"]
}

重要说明：
- student_emotions应该基于学生第一句发言分析学生的初始人格特征，如"困惑"、"积极"、"紧张"等
- 不要包含对话过程中的情绪变化，只关注初始状态
- student_understanding_level需要基于以下科学标准进行严谨评估：
  * 0-2分：完全不懂，仍在表达困惑
  * 3-4分：有初步理解，但仍有明显疑问
  * 5-6分：基本理解概念，但细节掌握不牢
  * 7-8分：较好理解，能独立应用
  * 9-10分：完全理解，能举一反三
  请根据学生最后几轮发言的内容、是否表达感谢、是否总结学到的内容等来科学判断
- effectiveness_score需要基于教学目标的达成度、学生进步幅度、教学策略的适宜性等综合评估
- 所有字段都必须存在
- student_emotions和teacher_strategies必须是字符串数组
- student_understanding_level和effectiveness_score必须是0-10的浮点数
- 不要添加任何其他字段
- 确保JSON格式完全正确"""
            },
            {
                "role": "user",
                "content": f"""请分析以下师生对话，生成一条高度凝练的标准格式教学经验：

学生第一句发言（用于分析初始人格）：
{first_student_message}

完整对话内容：
{conversation_text}

请严格按照指定格式返回JSON，不要包含任何其他内容。"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.3)
        
        try:
            # 尝试解析JSON响应
            experience = json.loads(response)
            
            # 验证必要字段
            required_fields = ["problem_scenario", "student_emotions", "student_understanding_level", 
                             "teacher_strategies", "effectiveness_score", "applicable_conditions"]
            
            for field in required_fields:
                if field not in experience:
                    raise ValueError(f"缺少必要字段: {field}")
            
            # 验证数据类型
            if not isinstance(experience["student_emotions"], list):
                experience["student_emotions"] = [experience["student_emotions"]]
            
            if not isinstance(experience["teacher_strategies"], list):
                experience["teacher_strategies"] = [experience["teacher_strategies"]]
            
            if not isinstance(experience["applicable_conditions"], list):
                experience["applicable_conditions"] = [experience["applicable_conditions"]]
            
            # 确保数值在合理范围内
            experience["student_understanding_level"] = max(0.0, min(10.0, float(experience["student_understanding_level"])))
            experience["effectiveness_score"] = max(0.0, min(10.0, float(experience["effectiveness_score"])))
            
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "标准经验生成成功", 
                    f"场景: {experience['problem_scenario']}, 初始人格: {experience['student_emotions']}, 效果: {experience['effectiveness_score']}")
                self.logger.log_analysis_result("REFLECTION", "标准经验数据", experience)
            
            return experience
            
        except Exception as e:
            # LLM经验生成失败，直接终止程序
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "经验生成失败", "程序终止")
            raise Exception(f"反思智能体经验生成失败：{str(e)}") 