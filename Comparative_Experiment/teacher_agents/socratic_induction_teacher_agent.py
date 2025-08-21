# -*- coding: utf-8 -*-
"""
苏格拉底循循善诱教学范式教师智能体
实现传统的苏格拉底式循循善诱教学方法
"""
from typing import Dict, Any, List, Optional
from config import METHOD_CONFIGS, SOCRATIC_INDUCTION_PROMPT


class SocraticInductionTeacherAgent:
    """苏格拉底循循善诱教学范式教师智能体"""
    
    def __init__(self, name: str, llm_manager):
        self.name = name
        self.llm_manager = llm_manager
        
        # 苏格拉底循循善诱特定配置
        self.temperature = METHOD_CONFIGS["Socratic_Induction"]["temperature"]
        self.max_questions = METHOD_CONFIGS["Socratic_Induction"]["max_questions"]
        self.induction_depth = METHOD_CONFIGS["Socratic_Induction"]["induction_depth"]
        
        # 对话历史存储
        self.conversation_history = []
        
        # 教学进度跟踪
        self.current_induction_step = 0
        self.induction_plan = []
        
        print(f"苏格拉底循循善诱教师智能体初始化完成 - 最大问题数: {self.max_questions}, 引导深度: {self.induction_depth}")

    def generate_response(self, student_message: str, student_state: Dict[str, Any], 
                        round_number: int) -> str:
        """使用苏格拉底循循善诱方法生成教师回复"""
        print(f"苏格拉底循循善诱方法开始生成第{round_number}轮回复")
        
        # 构建基础上下文
        context = self._build_context(student_message, student_state, round_number)
        
        # 如果是第一轮，制定教学引导计划
        if round_number == 1:
            self._create_induction_plan(context, student_message)
        
        # 生成循循善诱的回复
        response = self._generate_socratic_response(context, round_number, student_message)
        
        # 记录对话历史
        self._add_to_conversation_history("teacher", response, round_number)
        
        print(f"苏格拉底循循善诱方法生成完成，回复长度: {len(response)}字符")
        return response

    def _build_context(self, student_message: str, student_state: Dict[str, Any], 
                      round_number: int) -> str:
        """构建上下文"""
        context = f"""当前对话轮次：第{round_number}轮
学生消息：{student_message}
学生当前情绪：{student_state.get('current_emotion', '未知')}
学生人格特征：{student_state.get('persona', '未知')}
当前引导步骤：{self.current_induction_step + 1}/{len(self.induction_plan) if self.induction_plan else 1}"""
        
        return context

    def _create_induction_plan(self, context: str, student_message: str):
        """制定苏格拉底循循善诱的教学引导计划"""
        prompt = f"""{SOCRATIC_INDUCTION_PROMPT}

基于学生的初始问题，设计一个苏格拉底循循善诱的教学引导计划。

学生问题：{student_message}
{context}

请设计{self.max_questions}个相互关联的问题，形成完整的教学引导链条。每个问题应该：
1. 建立在前一个问题的答案基础上
2. 逐步引导学生从简单到复杂
3. 帮助学生自己发现答案
4. 体现从具体到抽象、从已知到未知的原则

请按以下格式输出：
步骤1：[具体问题描述]
步骤2：[基于步骤1的问题]
步骤3：[基于步骤2的问题]
...以此类推

每个步骤都应该有明确的教学目标。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=self.temperature)
        
        if response:
            # 解析引导计划
            self.induction_plan = self._parse_induction_plan(response)
            print(f"教学引导计划制定完成，共{len(self.induction_plan)}个步骤")
        else:
            # 使用默认引导计划
            self.induction_plan = self._create_default_plan(student_message)
            print("使用默认教学引导计划")

    def _parse_induction_plan(self, response: str) -> List[str]:
        """解析引导计划"""
        lines = response.strip().split('\n')
        plan = []
        
        for line in lines:
            line = line.strip()
            if line and ('步骤' in line or 'Step' in line):
                # 提取步骤内容
                if '：' in line:
                    content = line.split('：', 1)[-1].strip()
                elif ':' in line:
                    content = line.split(':', 1)[-1].strip()
                else:
                    content = line
                
                if content:
                    plan.append(content)
        
        return plan[:self.max_questions]

    def _create_default_plan(self, student_message: str) -> List[str]:
        """创建默认的教学引导计划"""
        default_plan = [
            "首先，让我们从你已知的基础概念开始。你能告诉我这道题涉及哪些基本知识点吗？",
            "很好！现在让我们看看这些知识点之间有什么联系。你觉得应该从哪里入手？",
            "基于你的想法，我们来分析一下具体的解题步骤。你觉得第一步应该做什么？",
            "现在让我们验证一下你的思路。你能解释一下为什么这样做吗？",
            "最后，让我们总结一下解题的关键点。你学到了什么？"
        ]
        return default_plan

    def _generate_socratic_response(self, context: str, round_number: int, student_message: str) -> str:
        """生成苏格拉底循循善诱的回复"""
        if not self.induction_plan:
            return self._fallback_response(context)
        
        # 根据当前步骤和学生的回复生成引导问题
        current_step = min(self.current_induction_step, len(self.induction_plan) - 1)
        planned_question = self.induction_plan[current_step]
        
        # 分析学生回复，调整引导策略
        student_analysis = self._analyze_student_response(student_message, round_number)
        
        prompt = f"""{SOCRATIC_INDUCTION_PROMPT}

{context}

教学引导计划 - 步骤{current_step + 1}：{planned_question}

学生回复分析：{student_analysis}

请基于苏格拉底循循善诱教学范式，生成一个引导性的回复。要求：
1. 体现渐进式引导原则
2. 设计启发式问题，激发学生思考
3. 基于学生当前的理解水平调整引导深度
4. 给予适当的情感支持和鼓励
5. 确保问题之间有逻辑联系
6. 引导学生自己发现答案，而不是直接给出

请生成一个自然、温和、有引导性的教学回复。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=self.temperature)
        
        if response:
            # 更新引导步骤
            self._update_induction_progress(student_message, response)
            return response.strip()
        else:
            return self._fallback_response(context)

    def _analyze_student_response(self, student_message: str, round_number: int) -> str:
        """分析学生回复，判断理解程度和引导方向"""
        prompt = f"""分析学生的回复，判断其理解程度和下一步引导方向。

学生回复：{student_message}
当前轮次：{round_number}

请分析：
1. 学生的理解程度（完全不懂/部分理解/基本理解/完全理解）
2. 学生的思维状态（困惑/思考中/有想法/自信）
3. 下一步引导的重点（概念澄清/方法指导/错误纠正/知识巩固）
4. 建议的引导策略

请简洁地总结分析结果。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.5)
        
        if response:
            return response.strip()
        else:
            return "学生正在思考中，需要进一步引导"

    def _update_induction_progress(self, student_message: str, teacher_response: str):
        """更新引导进度"""
        # 判断是否应该进入下一步
        if self._should_proceed_to_next_step(student_message, teacher_response):
            self.current_induction_step += 1
            print(f"引导进度更新：进入步骤 {self.current_induction_step + 1}")

    def _should_proceed_to_next_step(self, student_message: str, teacher_response: str) -> bool:
        """判断是否应该进入下一步引导"""
        # 检查学生回复是否表明理解
        understanding_indicators = [
            "我明白了", "我懂了", "我理解了", "我学会了", "我知道了",
            "原来如此", "这样啊", "我懂了", "我明白了", "我理解了"
        ]
        
        for indicator in understanding_indicators:
            if indicator in student_message:
                return True
        
        # 检查是否已经达到最大步骤数
        if self.current_induction_step >= len(self.induction_plan) - 1:
            return False
        
        # 检查教师回复是否包含明确的引导问题
        if "？" in teacher_response or "?" in teacher_response:
            return False  # 如果包含问题，继续当前步骤
        
        return True

    def _fallback_response(self, context: str) -> str:
        """生成备用回复"""
        return "同学，我理解你的困惑。让我们一起来分析这个问题，你觉得应该从哪里开始呢？"

    def _add_to_conversation_history(self, sender: str, content: str, round_number: int):
        """添加消息到对话历史"""
        message = {
            "sender": sender,
            "content": content,
            "type": "message",
            "round": round_number
        }
        self.conversation_history.append(message)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.conversation_history.copy()

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
        self.current_induction_step = 0
        self.induction_plan = []
