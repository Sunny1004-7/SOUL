# -*- coding: utf-8 -*-
"""
对话生成器：生成八种不同教学方法的教学对话数据集
"""
import json
import random
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime
import os

from llm_manager import SimpleLLMManager
from student_agent import StudentAgent
from student_data_loader import StudentDataLoader
from config import DIALOGUE_CONFIG, METHOD_CONFIGS

# 导入八种教师智能体
from teacher_agents.tot_teacher_agent import ToTTeacherAgent
from teacher_agents.self_consistency_teacher_agent import SelfConsistencyTeacherAgent
from teacher_agents.best_of_n_teacher_agent import BestOfNTeacherAgent
from teacher_agents.zero_shot_teacher_agent import ZeroShotTeacherAgent
from teacher_agents.icl_teacher_agent import ICLTeacherAgent
from teacher_agents.cot_teacher_agent import CoTTeacherAgent
from teacher_agents.icl_cot_teacher_agent import ICLCoTTeacherAgent
from teacher_agents.socratic_induction_teacher_agent import SocraticInductionTeacherAgent


class DialogueGenerator:
    """对话生成器：生成八种不同教学方法的教学对话数据集"""
    
    def __init__(self):
        self.llm_manager = SimpleLLMManager()
        self.student_loader = StudentDataLoader()
        
        # 初始化八种教师智能体
        self.teacher_agents = {
            "ToT": ToTTeacherAgent("ToT_Teacher", self.llm_manager),
            "Self_Consistency": SelfConsistencyTeacherAgent("SC_Teacher", self.llm_manager),
            "Best_of_N": BestOfNTeacherAgent("BoN_Teacher", self.llm_manager),
            "Zero_shot": ZeroShotTeacherAgent("Zero_Teacher", self.llm_manager),
            "ICL": ICLTeacherAgent("ICL_Teacher", self.llm_manager),
            "CoT": CoTTeacherAgent("CoT_Teacher", self.llm_manager),
            "ICL_CoT": ICLCoTTeacherAgent("ICL_CoT_Teacher", self.llm_manager),
            "Socratic_Induction": SocraticInductionTeacherAgent("Socratic_Teacher", self.llm_manager)
        }
        
        # 创建输出目录
        self.output_dir = "generated_conversations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("对话生成器初始化完成")

    def generate_conversations_for_all_methods(self):
        """为所有八种方法生成对话数据集"""
        print(f"开始为所有方法生成对话数据集...")
        print(f"每种方法将生成 {DIALOGUE_CONFIG['conversations_per_method']} 个对话")
        
        results = {}
        
        for method_name, teacher_agent in self.teacher_agents.items():
            print(f"\n{'='*60}")
            print(f"正在为方法 {method_name} 生成对话...")
            print(f"{'='*60}")
            
            try:
                conversations = self._generate_conversations_for_method(
                    method_name, teacher_agent
                )
                
                results[method_name] = {
                    "status": "成功",
                    "conversations_count": len(conversations),
                    "file_path": f"{self.output_dir}/{method_name}_conversations.json"
                }
                
                # 保存到文件
                self._save_conversations(method_name, conversations)
                
                print(f"✅ {method_name} 方法生成完成，共 {len(conversations)} 个对话")
                
            except Exception as e:
                results[method_name] = {
                    "status": "失败",
                    "error": str(e)
                }
                print(f"❌ {method_name} 方法生成失败: {e}")
        
        # 输出总结
        self._print_summary(results)
        return results

    def _generate_conversations_for_method(self, method_name: str, teacher_agent) -> List[Dict[str, Any]]:
        """为指定方法生成对话数据集"""
        conversations = []
        
        # 获取可用的学生ID
        available_students = self.student_loader.get_all_student_ids()
        if not available_students:
            raise Exception("没有可用的学生数据")
        
        # 为每个对话选择不同的学生和题目
        for i in range(DIALOGUE_CONFIG['conversations_per_method']):
            print(f"  生成第 {i+1}/{DIALOGUE_CONFIG['conversations_per_method']} 个对话...")
            
            # 随机选择学生
            student_id = random.choice(available_students)
            
            # 获取学生的最后一道题目
            problem_content = self.student_loader.get_last_problem_content(student_id)
            if not problem_content:
                print(f"    警告：学生 {student_id} 没有题目数据，跳过")
                continue
            
            # 生成对话
            conversation = self._generate_single_conversation(
                method_name, teacher_agent, student_id, problem_content
            )
            
            if conversation:
                conversations.append(conversation)
                print(f"    对话生成完成，轮数: {len(conversation['conversation_history'])//2}")
            
            # 重置教师智能体的对话历史
            teacher_agent.reset_conversation()
            
            # 避免API调用过于频繁
            time.sleep(1)
        
        return conversations

    def _generate_single_conversation(self, method_name: str, teacher_agent, 
                                   student_id: str, problem_content: str) -> Dict[str, Any]:
        """生成单个教学对话"""
        try:
            # 创建学生智能体
            student_agent = StudentAgent(
                self.llm_manager, 
                problem_content, 
                student_id
            )
            
            # 生成对话ID
            conversation_id = f"{method_name}_{student_id}_{int(time.time())}"
            
            # 学生生成第一轮发言
            student_message = student_agent.generate_first_message()
            
            # 记录对话历史
            conversation_history = []
            conversation_history.append({
                "sender": "student",
                "content": student_message,
                "type": "message",
                "round": 1
            })
            
            # 开始多轮对话
            current_round = 1
            max_rounds = DIALOGUE_CONFIG['max_rounds']
            min_rounds = DIALOGUE_CONFIG['min_rounds']
            
            while current_round <= max_rounds:
                # 教师生成回复
                teacher_response = teacher_agent.generate_response(
                    student_message, 
                    student_agent.get_student_state(), 
                    current_round
                )
                
                # 记录教师回复
                conversation_history.append({
                    "sender": "teacher",
                    "content": teacher_response,
                    "type": "message",
                    "round": current_round
                })
                
                # 学生生成回复
                student_response = student_agent.generate_response(
                    teacher_response, 
                    current_round + 1
                )
                
                # 记录学生回复
                conversation_history.append({
                    "sender": "student",
                    "content": student_response,
                    "type": "message",
                    "round": current_round + 1
                })
                
                # 判断是否应该结束对话
                if self._should_end_conversation(student_response, current_round, min_rounds):
                    break
                
                # 准备下一轮
                student_message = student_response
                current_round += 1
            
            # 构建对话记录
            conversation = {
                "conversation_id": conversation_id,
                "method": method_name,
                "student_id": student_id,
                "problem_content": problem_content,
                "student_persona": student_agent.get_student_state()["persona"],
                "total_rounds": len(conversation_history) // 2,
                "conversation_history": conversation_history,
                "generation_timestamp": datetime.now().isoformat(),
                "method_config": METHOD_CONFIGS.get(method_name, {})
            }
            
            return conversation
            
        except Exception as e:
            print(f"    对话生成失败: {e}")
            return None

    def _should_end_conversation(self, student_response: str, current_round: int, min_rounds: int) -> bool:
        """判断是否应该结束对话"""
        # 至少进行最小轮数
        if current_round < min_rounds:
            return False
        
        # 检查学生回复是否表示理解或满意
        end_indicators = [
            "我明白了", "我懂了", "我理解了", "谢谢老师", "我学会了",
            "我明白了", "我懂了", "我理解了", "谢谢老师", "我学会了",
            "我明白了", "我懂了", "我理解了", "谢谢老师", "我学会了"
        ]
        
        for indicator in end_indicators:
            if indicator in student_response:
                return True
        
        # 如果轮数过多，也结束对话
        if current_round >= DIALOGUE_CONFIG['max_rounds']:
            return True
        
        return False

    def _save_conversations(self, method_name: str, conversations: List[Dict[str, Any]]):
        """保存对话数据到文件"""
        file_path = f"{self.output_dir}/{method_name}_conversations.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        print(f"    对话数据已保存到: {file_path}")

    def _print_summary(self, results: Dict[str, Any]):
        """打印生成结果总结"""
        print(f"\n{'='*60}")
        print("对话生成完成总结")
        print(f"{'='*60}")
        
        success_count = sum(1 for r in results.values() if r["status"] == "成功")
        total_count = len(results)
        
        print(f"总方法数: {total_count}")
        print(f"成功数: {success_count}")
        print(f"失败数: {total_count - success_count}")
        print(f"成功率: {success_count/total_count*100:.1f}%")
        
        print(f"\n详细结果:")
        for method_name, result in results.items():
            status_icon = "✅" if result["status"] == "成功" else "❌"
            print(f"{status_icon} {method_name}: {result['status']}")
            if result["status"] == "成功":
                print(f"   对话数量: {result['conversations_count']}")
                print(f"   文件路径: {result['file_path']}")
            else:
                print(f"   错误: {result['error']}")


def main():
    """主函数"""
    generator = DialogueGenerator()
    generator.generate_conversations_for_all_methods()


if __name__ == "__main__":
    main()
