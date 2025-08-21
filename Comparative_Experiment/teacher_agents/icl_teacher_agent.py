# -*- coding: utf-8 -*-
"""
In-Context Learning (ICL) 教师智能体
实现上下文学习方法生成教学回复
"""
from typing import Dict, Any, List
from config import METHOD_CONFIGS, BASE_TEACHER_PROMPT


class ICLTeacherAgent:
    """In-Context Learning 教师智能体"""
    
    def __init__(self, name: str, llm_manager):
        self.name = name
        self.llm_manager = llm_manager
        
        # ICL特定配置
        self.num_examples = METHOD_CONFIGS["ICL"]["num_examples"]
        self.temperature = METHOD_CONFIGS["ICL"]["temperature"]
        
        # 对话历史存储
        self.conversation_history = []
        
        print(f"ICL教师智能体初始化完成 - 示例数量: {self.num_examples}")

    def generate_response(self, student_message: str, student_state: Dict[str, Any], 
                        round_number: int) -> str:
        """使用ICL方法生成教师回复"""
        print(f"ICL方法开始生成第{round_number}轮回复")
        
        # 构建基础上下文
        context = self._build_context(student_message, student_state, round_number)
        
        # 使用上下文学习生成回复
        response = self._generate_icl_response(context, round_number)
        
        # 记录对话历史
        self._add_to_conversation_history("teacher", response, round_number)
        
        print(f"ICL方法生成完成，回复长度: {len(response)}字符")
        return response

    def _build_context(self, student_message: str, student_state: Dict[str, Any], 
                      round_number: int) -> str:
        """构建上下文"""
        context = f"""当前对话轮次：第{round_number}轮
学生消息：{student_message}
学生当前情绪：{student_state.get('current_emotion', '未知')}
学生人格特征：{student_state.get('persona', '未知')}"""
        
        return context

    def _generate_icl_response(self, context: str, round_number: int) -> str:
        """使用上下文学习生成回复"""
        # 构建示例对话
        examples = self._build_examples(round_number)
        
        prompt = f"""{BASE_TEACHER_PROMPT}

以下是{self.num_examples}个教学对话示例，请学习其中的教学方法和风格：

{examples}

现在请基于以上示例，为以下情况生成教学回复：

{context}

请按照示例中的教学风格和方法，生成一个专业、有情感支持的教学回复。"""

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
            return examples[0]  # 使用第一个示例
        elif round_number <= 4:
            return examples[1]  # 使用第二个示例
        else:
            return examples[2]  # 使用第三个示例

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
