# -*- coding: utf-8 -*-
"""
Self-Consistency (SC) 教师智能体
实现自一致性采样方法生成教学回复
"""
from typing import Dict, Any, List, List, Optional
from config import METHOD_CONFIGS, BASE_TEACHER_PROMPT


class SelfConsistencyTeacherAgent:
    """Self-Consistency 教师智能体"""
    
    def __init__(self, name: str, llm_manager):
        self.name = name
        self.llm_manager = llm_manager
        
        # Self-Consistency特定配置
        self.num_samples = METHOD_CONFIGS["Self_Consistency"]["num_samples"]
        self.temperature = METHOD_CONFIGS["Self_Consistency"]["temperature"]
        
        print(f"Self-Consistency教师智能体初始化完成 - 采样数量: {self.num_samples}")
        
        # 对话历史存储
        self.conversation_history = []
        

    def generate_response(self, student_message: str, student_state: Dict[str, Any], 
                        round_number: int) -> str:
        """使用Self-Consistency方法生成教师回复"""
        print(f"Self-Consistency方法开始生成第{round_number}轮回复")
        
        # 构建基础上下文
        context = self._build_context(student_message, student_state, round_number)
        
        # 生成多个候选回复
        candidate_responses = self._generate_candidate_responses(context, round_number)
        
        if not candidate_responses:
            return self._fallback_response(context)
        
        # 选择最一致的回复
        best_response = self._select_most_consistent_response(candidate_responses, context, round_number)
        
        print(f"Self-Consistency方法生成完成，候选数量: {len(candidate_responses)}, 最佳回复长度: {len(best_response)}字符")
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
        for i in range(self.num_samples):
            messages = [
                {"role": "system", "content": prompt}
            ]
            
            response = self.llm_manager.call_llm(messages, temperature=self.temperature)
            
            if response and response.strip():
                candidate_responses.append(response.strip())
        
        return candidate_responses

    def _select_most_consistent_response(self, candidate_responses: List[str], 
                                       context: str, round_number: int) -> str:
        """选择最一致的回复"""
        if len(candidate_responses) == 1:
            return candidate_responses[0]
        
        # 计算每个回复与其他回复的一致性分数
        consistency_scores = []
        for i, response in enumerate(candidate_responses):
            score = self._calculate_consistency_score(response, candidate_responses, i)
            consistency_scores.append({
                "response": response,
                "score": score,
                "index": i
            })
        
        # 选择一致性分数最高的回复
        best_response = max(consistency_scores, key=lambda x: x["score"])
        
        return best_response["response"]

    def _calculate_consistency_score(self, target_response: str, all_responses: List[str], 
                                   target_index: int) -> float:
        """计算目标回复与其他回复的一致性分数"""
        if len(all_responses) <= 1:
            return 1.0
        
        # 计算语义相似度
        similarity_scores = []
        for i, other_response in enumerate(all_responses):
            if i != target_index:
                similarity = self._calculate_semantic_similarity(target_response, other_response)
                similarity_scores.append(similarity)
        
        # 返回平均相似度作为一致性分数
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

    def _calculate_semantic_similarity(self, response1: str, response2: str) -> float:
        """计算两个回复的语义相似度"""
        # 使用LLM计算语义相似度
        prompt = f"""请评估以下两个教学回复的语义相似度，从0到1打分：

回复1：{response1}

回复2：{response2}

请只输出一个0到1之间的数字，表示相似度。1表示完全相同，0表示完全不同。"""

        messages = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.1)
        
        try:
            similarity = float(response.strip())
            return max(0.0, min(1.0, similarity))
        except:
            # 如果LLM调用失败，使用简单的文本相似度
            return self._simple_text_similarity(response1, response2)

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        # 分词并计算Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

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
