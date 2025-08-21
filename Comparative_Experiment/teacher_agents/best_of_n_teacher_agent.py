# -*- coding: utf-8 -*-
"""
Best-of-N (BoN) 教师智能体
实现N选一最优方法生成教学回复
"""
from typing import Dict, Any, List, List, Optional
from config import METHOD_CONFIGS, BASE_TEACHER_PROMPT


class BestOfNTeacherAgent:
    """Best-of-N 教师智能体"""
    
    def __init__(self, name: str, llm_manager):
        self.name = name
        self.llm_manager = llm_manager
        
        # Best-of-N特定配置
        self.num_candidates = METHOD_CONFIGS["Best_of_N"]["num_candidates"]
        self.temperature = METHOD_CONFIGS["Best_of_N"]["temperature"]
        
        print(f"Best-of-N教师智能体初始化完成 - 候选数量: {self.num_candidates}")
        
        # 对话历史存储
        self.conversation_history = []
        

    def generate_response(self, student_message: str, student_state: Dict[str, Any], 
                        round_number: int) -> str:
        """使用Best-of-N方法生成教师回复"""
        print(f"Best-of-N方法开始生成第{round_number}轮回复")
        
        # 构建基础上下文
        context = self._build_context(student_message, student_state, round_number)
        
        # 生成多个候选回复
        candidate_responses = self._generate_candidate_responses(context, round_number)
        
        if not candidate_responses:
            return self._fallback_response(context)
        
        # 评估并选择最佳回复
        best_response = self._select_best_response(candidate_responses, context, round_number)
        
        print(f"Best-of-N方法生成完成，候选数量: {len(candidate_responses)}, 最佳回复长度: {len(best_response)}字符")
        return best_response

    def _build_context(self, student_message: str, student_state: Dict[str, Any], 
                      round_number: int) -> str:
        """构建上下文"""
        context = f"""当前对话轮次：第{round_number}轮
学生消息：{student_message}
学生当前情绪：{student_state.get('current_emotion', '未知')}
学生人格特征：{student_state.get('persona', '未知')}"""
        
        return context

    def _generate_candidate_responses(self, context: str, round_number: int) -> List[str]:
        """生成多个候选回复"""
        prompt = f"""{BASE_TEACHER_PROMPT}

{context}

请生成一个专业、有情感支持的教学回复。"""

        # 生成多个候选回复
        candidate_responses = []
        for i in range(self.num_candidates):
            messages = [
                {"role": "system", "content": prompt}
            ]
            
            response = self.llm_manager.call_llm(messages, temperature=self.temperature)
            
            if response and response.strip():
                candidate_responses.append(response.strip())
        
        return candidate_responses

    def _select_best_response(self, candidate_responses: List[str], 
                            context: str, round_number: int) -> str:
        """选择最佳回复"""
        if len(candidate_responses) == 1:
            return candidate_responses[0]
        
        # 评估每个候选回复的质量
        evaluated_responses = []
        for i, response in enumerate(candidate_responses):
            quality_score = self._evaluate_response_quality(response, context, round_number)
            evaluated_responses.append({
                "response": response,
                "score": quality_score,
                "index": i
            })
        
        # 选择质量分数最高的回复
        best_response = max(evaluated_responses, key=lambda x: x["score"])
        
        return best_response["response"]

    def _evaluate_response_quality(self, response: str, context: str, round_number: int) -> float:
        """评估回复质量"""
        prompt = f"""请评估以下教学回复的质量，从0到10打分：

教学回复：{response}

评估标准：
- 专业性（教学内容准确性、方法科学性）：0-4分
- 情感支持（对学生情绪的关怀、语言温和性）：0-3分
- 教学策略（苏格拉底式方法的运用）：0-3分

请只输出一个0-10之间的数字，不要其他内容。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response_result = self.llm_manager.call_llm(messages, temperature=0.1)
        
        try:
            score = float(response_result.strip())
            return max(0, min(10, score))
        except:
            return 5.0

    def _fallback_response(self, context: str) -> str:
        """生成备用回复"""
        prompt = f"""{BASE_TEACHER_PROMPT}

{context}

请生成一个简洁的教学回复。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.7)
        
        if not response:
            return "同学，我理解你的困惑。让我们一起来分析这个问题，你觉得应该从哪里开始呢？"
        
        return response.strip()
