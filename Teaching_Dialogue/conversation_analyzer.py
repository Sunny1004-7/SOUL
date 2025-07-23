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
        self.max_rounds = 7  # 最长对话轮次

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
                self.logger.log_agent_work("CONVERSATION_ANALYZER", "LLM调用失败", "使用默认判断")
            
            # LLM调用失败时，使用智能默认判断
            should_end = self._smart_default_judgment(student_message, round_number)
            
            default_result = {
                "should_end": should_end,
                "reason": "LLM调用失败，基于学生表达智能判断" + ("，学生表达理解应结束对话" if should_end else "，学生仍有疑问应继续对话"),
                "student_understanding": "LLM调用失败，无法准确评估学生理解程度",
                "round_number": round_number
            }
            
            if self.logger:
                end_status = "结束" if should_end else "继续"
                self.logger.log_agent_work("CONVERSATION_ANALYZER", f"对话{end_status}（默认）", 
                                         f"理由: {default_result['reason']}")
            
            return default_result
        
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
            # LLM解析失败，使用默认值而不是终止程序
            if self.logger:
                self.logger.log_agent_work("CONVERSATION_ANALYZER", "LLM解析失败", f"使用默认值，错误: {e}")
            
            # 基于学生消息内容智能判断是否应该结束
            should_end = self._smart_default_judgment(student_message, round_number)
            
            default_result = {
                "should_end": should_end,
                "reason": "LLM解析失败，基于学生表达智能判断" + ("，学生表达理解应结束对话" if should_end else "，学生仍有疑问应继续对话"),
                "student_understanding": "LLM解析失败，无法准确评估学生理解程度",
                "round_number": round_number
            }
            
            if self.logger:
                end_status = "结束" if should_end else "继续"
                self.logger.log_agent_work("CONVERSATION_ANALYZER", f"对话{end_status}（默认）", 
                                         f"理由: {default_result['reason']}")
            
            return default_result
    
    def _smart_default_judgment(self, student_message: str, round_number: int) -> bool:
        """智能默认判断是否应该结束对话"""
        # 如果超过最大轮次，直接结束
        if round_number >= self.max_rounds:
            return True
        
        # 检查学生是否表达了理解和感谢
        end_indicators = [
            "谢谢", "感谢", "明白了", "我懂了", "理解了", "清楚了", 
            "知道了", "学会了", "掌握了", "好的", "嗯", "是的",
            "没问题", "清楚了", "明白了", "懂了", "理解了"
        ]
        
        # 检查学生是否还有疑问或困惑
        continue_indicators = [
            "但是", "不过", "还是", "仍然", "依然", "还是有点",
            "不太明白", "不太清楚", "不太理解", "有点困惑", "有点疑问",
            "能再", "可以再", "能否", "可以吗", "怎么", "为什么",
            "什么", "哪个", "哪里", "如何", "怎样"
        ]
        
        message_lower = student_message.lower()
        
        # 如果包含结束指示词且不包含继续指示词，则结束对话
        has_end_indicator = any(indicator in message_lower for indicator in end_indicators)
        has_continue_indicator = any(indicator in message_lower for indicator in continue_indicators)
        
        # 如果学生明确表达感谢和理解，且没有新的疑问，则结束
        if has_end_indicator and not has_continue_indicator:
            return True
        
        # 如果学生有新的疑问或困惑，则继续对话
        if has_continue_indicator:
            return False
        
        # 默认情况下，如果轮次较少（前3轮），倾向于继续对话
        if round_number <= 3:
            return False
        
        # 如果轮次较多且学生没有明确表达，倾向于结束对话
        return True 