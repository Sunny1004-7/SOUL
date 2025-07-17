# -*- coding: utf-8 -*-
"""
对话分析模块：负责智能判断对话是否应该结束
基于学生表达理解和感谢，最长10轮对话
"""
from typing import Dict, Any, List
import json


class ConversationAnalyzer:
    """对话分析器：智能判断对话是否应该结束"""
    
    def __init__(self, llm_manager, logger=None):
        self.llm_manager = llm_manager
        self.logger = logger
        self.max_rounds = 10  # 最长对话轮次

    def analyze_conversation_end(self, student_message: str, conversation_history: List[Dict], 
                                round_number: int, problem_content: str) -> Dict[str, Any]:
        """分析对话是否应该结束"""
        if self.logger:
            self.logger.log_agent_work("CONVERSATION_ANALYZER", "开始对话结束分析", f"第{round_number}轮")
        
        # 如果超过最大轮次，直接结束
        if round_number >= self.max_rounds:
            if self.logger:
                self.logger.log_agent_work("CONVERSATION_ANALYZER", "对话结束", f"达到最大轮次{self.max_rounds}")
            return {
                "should_end": True,
                "reason": f"达到最大对话轮次{self.max_rounds}",
                "round_number": round_number
            }
        
        # 构建对话历史文本
        history_text = ""
        recent_history = conversation_history[-6:]  
        for entry in recent_history:
            role = "学生" if entry.get("sender") == "student" else "老师"
            content = entry.get("content", "")
            history_text += f"{role}: {content}\n"
        
        messages = [
            {
                "role": "system",
                "content": """你是一名对话分析专家，负责判断师生对话是否应该结束。

判断标准：
1. 学生明确表达了理解和感谢（如"谢谢老师"、"我明白了"、"我懂了"等）
2. 学生能够总结学到的内容
3. 学生没有提出新的问题
4. 学习目标基本达成

请以JSON格式回复：
{
    "should_end": true/false,
    "reason": "结束或继续的理由",
    "student_understanding": "学生对当前问题的理解程度描述"
}

注意：
- 如果学生表达感谢和理解，应该结束对话
- 如果学生还有疑问或困惑，应该继续对话
- 只关注学生是否理解并感谢，不要过度分析"""
            },
            {
                "role": "user",
                "content": f"""请分析以下师生对话是否应该结束：

题目：{problem_content}
当前轮次：第{round_number}轮

学生最新发言：{student_message}

最近对话历史：
{history_text}

请判断学生是否已经理解并表达感谢，决定是否结束对话。"""
            }
        ]
        
        response = self.llm_manager.call_llm(messages, temperature=0.2)
        
        if not response:
            if self.logger:
                self.logger.log_agent_work("CONVERSATION_ANALYZER", "LLM调用失败", "程序终止")
            raise Exception("对话分析器LLM调用失败：未获得有效响应")
        
        try:
            result = json.loads(response)
            
            # 确保返回必要的字段
            if "should_end" not in result:
                raise ValueError("LLM响应缺少should_end字段")
            if "reason" not in result:
                raise ValueError("LLM响应缺少reason字段")
            if "student_understanding" not in result:
                raise ValueError("LLM响应缺少student_understanding字段")
            
            result["round_number"] = round_number
            
            if self.logger:
                end_status = "结束" if result["should_end"] else "继续"
                self.logger.log_agent_work("CONVERSATION_ANALYZER", f"对话{end_status}", 
                                         f"理由: {result['reason']}")
                self.logger.log_analysis_result("CONVERSATION_ANALYZER", "对话结束分析", result)
            
            return result
            
        except Exception as e:
            # LLM解析失败，直接终止程序
            if self.logger:
                self.logger.log_agent_work("CONVERSATION_ANALYZER", "分析失败", f"错误: {e}")
            raise Exception(f"对话分析器LLM解析失败：{str(e)}") 