# -*- coding: utf-8 -*-
"""
Tree of Thoughts (ToT) 教师智能体
实现多分支思维树方法生成教学回复
"""
from typing import Dict, Any, List, Optional
import random
from config import METHOD_CONFIGS, BASE_TEACHER_PROMPT


class ToTTeacherAgent:
    """Tree of Thoughts 教师智能体"""
    
    def __init__(self, name: str, llm_manager):
        self.name = name
        self.llm_manager = llm_manager
        
        # ToT特定配置
        self.max_branches = METHOD_CONFIGS["ToT"]["max_branches"]
        self.max_depth = METHOD_CONFIGS["ToT"]["max_depth"]
        self.temperature = METHOD_CONFIGS["ToT"]["temperature"]
        
        # 对话历史存储
        self.conversation_history = []
        
        print(f"ToT教师智能体初始化完成 - 最大分支数: {self.max_branches}, 最大深度: {self.max_depth}")

    def generate_response(self, student_message: str, student_state: Dict[str, Any], 
                        round_number: int) -> str:
        """使用ToT方法生成教师回复"""
        print(f"ToT方法开始生成第{round_number}轮回复")
        
        # 构建基础上下文
        context = self._build_context(student_message, student_state, round_number)
        
        # 执行ToT思维树搜索
        best_response = self._execute_tot_search(context, round_number)
        
        # 记录对话历史
        self._add_to_conversation_history("teacher", best_response, round_number)
        
        print(f"ToT方法生成完成，回复长度: {len(best_response)}字符")
        return best_response

    def _build_context(self, student_message: str, student_state: Dict[str, Any], 
                      round_number: int) -> str:
        """构建上下文"""
        context = f"""当前对话轮次：第{round_number}轮
学生消息：{student_message}
学生当前情绪：{student_state.get('current_emotion', '未知')}
学生人格特征：{student_state.get('persona', '未知')}"""
        
        return context

    def _execute_tot_search(self, context: str, round_number: int) -> str:
        """执行ToT思维树搜索"""
        # 第一层：生成多个思维分支
        thought_branches = self._generate_thought_branches(context, round_number)
        
        if not thought_branches:
            return self._fallback_response(context)
        
        # 第二层：评估每个分支
        evaluated_branches = []
        for i, branch in enumerate(thought_branches):
            evaluation = self._evaluate_thought_branch(branch, context, round_number)
            evaluated_branches.append({
                "branch": branch,
                "evaluation": evaluation,
                "index": i
            })
        
        # 第三层：选择最佳分支并扩展
        best_branch = self._select_best_branch(evaluated_branches)
        if best_branch:
            final_response = self._expand_best_branch(best_branch, context, round_number)
            return final_response
        
        # 如果选择失败，使用最高评分的分支
        best_evaluated = max(evaluated_branches, key=lambda x: x["evaluation"])
        return best_evaluated["branch"]

    def _generate_thought_branches(self, context: str, round_number: int) -> List[str]:
        """生成多个思维分支"""
        prompt = f"""{BASE_TEACHER_PROMPT}

你是一名经验丰富的数学老师，需要为第{round_number}轮对话生成多个不同的教学思路。

{context}

请生成{self.max_branches}个不同的教学思路，每个思路应该：
1. 体现苏格拉底式教学方法
2. 关注学生的情绪状态
3. 针对学生的具体问题
4. 体现不同的教学策略

请直接输出{self.max_branches}个思路，每个思路用"思路X："开头，不要其他解释。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=self.temperature)
        
        if not response:
            return []
        
        # 解析响应，提取思路
        lines = response.strip().split('\n')
        branches = []
        for line in lines:
            if line.strip() and ('思路' in line or '思路' in line):
                content = line.split('：', 1)[-1].strip()
                if content:
                    branches.append(content)
        
        return branches[:self.max_branches]

    def _evaluate_thought_branch(self, branch: str, context: str, round_number: int) -> float:
        """评估思维分支的质量"""
        prompt = f"""你是一名教育专家，需要评估以下教学思路的质量。

{context}

教学思路：{branch}

请从以下维度评估这个思路（0-10分）：
1. 苏格拉底式教学体现程度
2. 情感支持适当性
3. 教学策略有效性
4. 语言表达清晰度

请只输出一个0-10之间的数字分数，不要其他内容。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.3)
        
        try:
            score = float(response.strip())
            return max(0, min(10, score))
        except:
            return 5.0

    def _select_best_branch(self, evaluated_branches: List[Dict[str, Any]]) -> Optional[str]:
        """选择最佳分支"""
        if not evaluated_branches:
            return None
        
        # 选择评分最高的分支
        best_branch = max(evaluated_branches, key=lambda x: x["evaluation"])
        return best_branch["branch"]

    def _expand_best_branch(self, best_branch: str, context: str, round_number: int) -> str:
        """扩展最佳分支，生成最终回复"""
        prompt = f"""{BASE_TEACHER_PROMPT}

基于以下教学思路，生成一个完整的教学回复：

{context}

教学思路：{best_branch}

请生成一个完整的教学回复，要求：
1. 体现苏格拉底式教学方法
2. 关注学生的情绪状态
3. 语言温和友善
4. 引导学生思考而不是直接给出答案
5. 长度适中（100-200字）"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=self.temperature)
        
        if not response:
            return self._fallback_response(context)
        
        return response.strip()

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
