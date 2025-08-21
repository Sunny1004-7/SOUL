# -*- coding: utf-8 -*-
"""
ICL + CoT 组合教师智能体
实现上下文学习+思维链组合方法生成教学回复
"""
from typing import Dict, Any, List
from config import METHOD_CONFIGS, BASE_TEACHER_PROMPT


class ICLCoTTeacherAgent:
    """ICL + CoT 组合教师智能体"""
    
    def __init__(self, name: str, llm_manager):
        self.name = name
        self.llm_manager = llm_manager
        
        # ICL+CoT特定配置
        self.num_examples = METHOD_CONFIGS["ICL_CoT"]["num_examples"]
        self.temperature = METHOD_CONFIGS["ICL_CoT"]["temperature"]
        
        print(f"ICL+CoT组合教师智能体初始化完成 - 示例数量: {self.num_examples}")
        
        # 对话历史存储
        self.conversation_history = []
        

    def generate_response(self, student_message: str, student_state: Dict[str, Any], 
                        round_number: int) -> str:
        """使用ICL+CoT组合方法生成教师回复"""
        print(f"ICL+CoT组合方法开始生成第{round_number}轮回复")
        
        # 构建基础上下文
        context = self._build_context(student_message, student_state, round_number)
        
        # 使用ICL+CoT组合方法生成回复
        response = self._generate_icl_cot_response(context, round_number)
        
        print(f"ICL+CoT组合方法生成完成，回复长度: {len(response)}字符")
        return response

    def _build_context(self, student_message: str, student_state: Dict[str, Any], 
                      round_number: int) -> str:
        """构建上下文"""
        context = f"""当前对话轮次：第{round_number}轮
学生消息：{student_message}
学生当前情绪：{student_state.get('current_emotion', '未知')}
学生人格特征：{student_state.get('persona', '未知')}"""
        
        return context

    def _generate_icl_cot_response(self, context: str, round_number: int) -> str:
        """使用ICL+CoT组合方法生成回复"""
        # 构建示例对话
        examples = self._build_examples(round_number)
        
        prompt = f"""{BASE_TEACHER_PROMPT}

以下是{self.num_examples}个教学对话示例，请学习其中的教学方法和风格：

{examples}

现在请基于以上示例，按照以下思维链步骤来生成教学回复：

步骤1：分析学生当前状态
- 学生的情绪状态如何？
- 学生表达了什么困难？
- 学生需要什么样的帮助？

步骤2：参考示例确定教学策略
- 示例中使用了什么样的苏格拉底式教学方法？
- 如何关注学生的情感需求？
- 如何引导学生主动思考？

步骤3：设计具体回复
- 如何开始对话？
- 应该提出什么问题？
- 如何给予情感支持？

步骤4：生成完整回复
基于以上分析和示例学习，生成一个专业、有情感支持的教学回复。

当前情况：
{context}

请按照这个思维链，结合示例学习，逐步分析并生成回复。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=self.temperature)
        
        if not response:
            return self._fallback_response(context)
        
        return response.strip()

    def _build_examples(self, round_number: int) -> str:
        """构建教学对话示例"""
        examples = []
        
        # 示例1：困惑情绪的学生
        examples.append("""示例1：
学生：老师，这道题我完全不知道从哪里开始，感觉很困惑。
老师：我理解你的困惑，这很正常。让我们一起来分析一下，你觉得这道题涉及哪些知识点呢？可以先回忆一下我们之前学过的内容。
学生：好像涉及函数和方程...
老师：很好！你已经找到了关键点。那么你觉得函数和方程之间有什么关系呢？""")
        
        # 示例2：紧张情绪的学生
        examples.append("""示例2：
学生：老师，我害怕答错，不敢说出我的想法。
老师：你的担心我完全理解，但请记住，在学习过程中犯错是很正常的，这恰恰说明你在思考。让我们一起来探索，即使错了也没关系，重要的是你敢于表达。
学生：我觉得答案可能是...
老师：很好！无论对错，你愿意分享想法就值得表扬。让我们一起来验证一下你的思路。""")
        
        # 示例3：理解困难的学生
        examples.append("""示例3：
学生：老师，我还是不太明白，能再解释一遍吗？
老师：当然可以！这说明你是一个认真的学生。让我们换个角度来理解，你能告诉我你具体哪一步不明白吗？
学生：就是那个转换的步骤...
老师：好的，让我们一步一步来。首先，你能回忆一下我们之前学过的类似转换吗？""")
        
        # 根据轮次选择示例
        if round_number <= 2:
            return examples[0]  # 前两轮使用困惑示例
        elif round_number <= 4:
            return examples[1]  # 中间轮次使用紧张示例
        else:
            return examples[2]  # 后续轮次使用理解困难示例

    def _fallback_response(self, context: str) -> str:
        """生成备用回复"""
        return "同学，我理解你的困惑。让我们一起来分析这个问题，你觉得应该从哪里开始呢？"
