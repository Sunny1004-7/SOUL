# -*- coding: utf-8 -*-
"""
基于AutoGen架构的反思智能体：实现事件驱动的教学经验提炼
保留原有的反思分析逻辑，但使用异步消息传递和Actor模型
"""
from core import BaseAgent, MessageType, Message
import json
import os
import time


class ReflectionAgent(BaseAgent):
    """基于AutoGen架构的反思智能体"""
    
    def __init__(self, name: str, llm_manager, experience_file="experience_bank.json", logger=None):
        super().__init__(name, logger)
        self.llm_manager = llm_manager
        self.experience_file = experience_file
        
        if self.logger:
            self.logger.log_agent_work("REFLECTION", "初始化完成", "教育反思系统已就绪")

    def initialize(self):
        """初始化反思智能体"""
        # 先调用父类的处理器注册
        super()._register_handlers()
        
        # 注册反思特定的消息处理器
        self.message_handlers[MessageType.REFLECTION_REQUEST] = self._handle_reflection_request
        
        self.update_state("ready", True)
        self.update_state("experiences_generated", 0)
        self.experiences_generated = 0  # 确保实例属性存在

    def _store_experience_to_json(self, key: str, experience: dict):
        """将经验数据存储到JSON文件中"""
        try:
            # 读取现有的经验数据
            existing_experiences = {}
            if os.path.exists(self.experience_file):
                try:
                    with open(self.experience_file, 'r', encoding='utf-8') as f:
                        existing_experiences = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    # 如果文件损坏或不存在，从空字典开始
                    existing_experiences = {}
            
            # 添加新的经验数据
            existing_experiences[key] = experience
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.experience_file) if os.path.dirname(self.experience_file) else '.', exist_ok=True)
            
            # 写入文件
            with open(self.experience_file, 'w', encoding='utf-8') as f:
                json.dump(existing_experiences, f, ensure_ascii=False, indent=2)
            
            # 更新状态
            self.experiences_generated += 1
            self.update_state("experiences_generated", self.experiences_generated)
            
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "经验存储成功", f"存储到文件: {self.experience_file}, key: {key}")
                
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "经验存储失败", f"错误: {str(e)}")
            # 不抛出异常，避免影响整个系统
            return False
        
        return True

    def _handle_reflection_request(self, message: Message):
        """处理反思请求（仅在对话结束后进行一次总结反思）"""
        content = message.content
        conversation_history = content.get("conversation_history", [])
        is_conversation_end = content.get("is_conversation_end", False)
        if not is_conversation_end:
            return

        # 检查对话历史是否为空
        if not conversation_history:
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "对话历史为空", "无法进行反思分析")
            return

        # 构造完整对话文本和学生第一句
        conversation_text = ""
        first_student_message = ""
        for msg in conversation_history:
            role = "学生" if msg.get("sender") == "student" else "老师"
            content_ = msg.get("content", "")
            conversation_text += f"{role}: {content_}\n"
            if not first_student_message and msg.get("sender") == "student":
                first_student_message = content_

        # 检查是否有足够的对话内容
        if len(conversation_text.strip()) < 50:  # 至少50个字符的对话内容
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "对话内容过少", f"对话长度: {len(conversation_text)}字符，无法进行有效分析")
            return

        # 只存在一个详细prompt，直接多行字符串
        prompt = """
你是一名教育研究专家和反思性教学实践者，擅长从完整教学对话中提炼有价值的教育经验。

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
- 确保JSON格式完全正确
"""

        if self.logger:
            self.logger.log_agent_work("REFLECTION", "开始LLM调用", f"对话长度: {len(conversation_text)}字符")
        
        try:
            # 使用完整对话内容，增加max_tokens以处理长对话
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"请分析以下师生对话，生成一条高度凝练的标准格式教学经验：\n\n学生第一句发言（用于分析初始人格）：\n{first_student_message}\n\n完整对话内容：\n{conversation_text}\n\n请严格按照指定格式返回JSON，不要包含任何其他内容。"}
            ]
            
            response = self.llm_manager.call_llm(messages, temperature=0.3, max_tokens=4000)
            
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "LLM调用成功", f"响应长度: {len(response) if response else 0}字符")
            
            # 检查LLM响应是否为空
            if not response:
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "LLM响应为空", "无法生成经验数据")
                return
            
            try:
                experience = json.loads(response)
            except json.JSONDecodeError as json_error:
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "JSON解析失败", f"LLM返回的内容不是有效的JSON格式: {str(json_error)}")
                    self.logger.log_agent_work("REFLECTION", "LLM原始响应", f"响应内容: {response[:200]}...")
                return
            
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "JSON解析成功", f"经验数据: {experience}")
            
            # 验证经验数据格式
            required_fields = ["problem_scenario", "student_emotions", "student_understanding_level", 
                             "teacher_strategies", "effectiveness_score", "applicable_conditions"]
            missing_fields = [field for field in required_fields if field not in experience]
            
            if missing_fields:
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "经验数据格式错误", f"缺少必需字段: {missing_fields}")
                return
            
            # 验证数值字段
            try:
                understanding_level = float(experience["student_understanding_level"])
                effectiveness_score = float(experience["effectiveness_score"])
                
                if not (0 <= understanding_level <= 10) or not (0 <= effectiveness_score <= 10):
                    if self.logger:
                        self.logger.log_agent_work("REFLECTION", "经验数据格式错误", f"评分必须在0-10范围内: 理解度={understanding_level}, 效果={effectiveness_score}")
                    return
                    
            except (ValueError, TypeError):
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "经验数据格式错误", "评分字段必须是数字")
                return
            
            key = f"exp_{int(time.time())}"
            if self._store_experience_to_json(key, experience):
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "经验存储成功", f"存储key: {key}")
            else:
                if self.logger:
                    self.logger.log_agent_work("REFLECTION", "经验存储失败", f"存储key: {key}")
                
        except Exception as e:
            if self.logger:
                self.logger.log_agent_work("REFLECTION", "处理失败", f"错误: {str(e)}")
            # 不抛出异常，避免影响整个系统
            return 