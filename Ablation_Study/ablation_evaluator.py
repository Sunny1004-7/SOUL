# -*- coding: utf-8 -*-
"""
消融实验教师回复质量评估器
使用LLM评估4种消融方法生成的教师回复质量
读取JSON格式的对话数据，每次评估一个完整的多轮教学对话
"""
import pandas as pd
import json
import argparse
import sys
import os
from typing import Dict, Any, List
from pathlib import Path

# 添加路径以导入必要的模块
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Comparative_Experiment'))
from llm_manager import SimpleLLMManager


def load_conversation_data(file_path: str) -> List[Dict[str, Any]]:
    """加载对话数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载对话数据，共 {len(data)} 个对话")
        return data
    except Exception as e:
        print(f"加载文件失败: {e}")
        return []


def extract_teacher_responses(conversation: Dict[str, Any]) -> List[str]:
    """从单个对话中提取教师回复"""
    teacher_responses = []
    
    if "conversation_history" in conversation:
        for turn in conversation["conversation_history"]:
            if turn.get("sender") == "teacher" and turn.get("type") == "message":
                content = turn.get("content", "").strip()
                if content:
                    teacher_responses.append(content)
    
    return teacher_responses


def evaluate_single_conversation(agent: Dict, conversation: Dict[str, Any], 
                                dimension: str, llm_manager: SimpleLLMManager) -> Dict[str, Any]:
    """评估单个对话在指定维度的教师回复质量"""
    
    # 提取教师回复
    teacher_responses = extract_teacher_responses(conversation)
    
    if not teacher_responses:
        return {
            "conversation_id": conversation.get("conversation_id", "unknown"),
            "dimension": dimension,
            "score": 0.0,
            "teacher_responses_count": 0,
            "error": "没有找到教师回复"
        }
    
    # 构建评估提示词
    dimension_prompt = f"""
    Task Description:
        You are an education expert and a specialist evaluator in teacher response quality. Your task is to rigorously evaluate the teacher responses in a teaching conversation on the following specific dimension: {dimension}.
        
        Dimension Definitions:
        
        **Emotional Support Dimensions:**
        1. **语气友善性**: Evaluate whether the teacher uses warm, encouraging language that makes students feel comfortable and supported.
        2. **情绪响应敏感度**: Evaluate whether the teacher identifies and responds to students' core emotions (such as anxiety, frustration, confusion, etc.).
        3. **共情表达**: Evaluate whether the teacher shows understanding and acceptance, providing affirmation or comfort.
        4. **安全与恰当性**: Evaluate whether the response avoids harmful, misleading, or biased content, ensuring psychological and cultural safety.
        5. **学习氛围营造**: Evaluate whether the teacher creates a positive, inclusive interactive environment that enhances student confidence.
        
        **Professionalism Dimensions:**
        6. **内容准确性**: Evaluate whether the mathematical/subject knowledge is accurate and conforms to textbooks and standards.
        7. **苏格拉底式引导风格**: Evaluate whether the teacher primarily guides student thinking through questioning rather than directly imparting answers.
        8. **方法适切性**: Evaluate whether the teaching strategies (analogies, examples, decomposition, etc.) are appropriate for student cognition and context.
        9. **表达清晰度**: Evaluate whether the language is concise, logically coherent, and easy to understand.
        10. **认知支持深度**: Evaluate whether the response promotes student reflection, transfer, and independent problem-solving.
        
        Evaluation Process:
        1. Review all teacher responses in the conversation based on the dimension {dimension}.
        2. For each teacher response, assign a score from 0 to 10:
           - 10: Excellent - Fully meets evaluation criteria, outstanding performance
           - 9: Excellent - Almost fully meets evaluation criteria, excellent performance
           - 8: Good - Basically meets evaluation criteria, good performance
           - 7: Good - Mostly meets evaluation criteria, better performance
           - 6: Fair - Partially meets evaluation criteria, average performance
           - 5: Fair - Basically meets evaluation criteria, medium performance
           - 4: Poor - Rarely meets evaluation criteria, poor performance
           - 3: Poor - Partially meets evaluation criteria, below average performance
           - 2: Very Poor - Very rarely meets evaluation criteria, very poor performance
           - 1: Very Poor - Almost never meets evaluation criteria, extremely poor performance
           - 0: Invalid - Completely fails to meet evaluation criteria, no effect
        3. Calculate the average score across all teacher responses to obtain the final score for this conversation.
        
        Teacher Responses to Evaluate:
        {chr(10).join([f"Response {i+1}: {response}" for i, response in enumerate(teacher_responses)])}
        
        Please return only a JSON response in the following format:
        {{
            "conversation_id": "{conversation.get('conversation_id', 'unknown')}",
            "dimension": "{dimension}",
            "average_score": <average_score>,
            "individual_scores": [<score1>, <score2>, ...],
            "evaluation_notes": "<brief explanation of the score>"
        }}
        
        Important:
        - Your response must strictly follow the above JSON format.
        - Do not include any additional text, commentary, or explanations beyond the specified JSON structure.
        - Scores should be between 0 and 10, with 10 being the highest quality.
        """

    messages = [
        {"role": "system", "content": dimension_prompt}
    ]
    
    try:
        response = llm_manager.call_llm(messages, temperature=0.1)
        
        if not response:
            raise ValueError("LLM 未返回结果")
        
        # 清理和解析 LLM 返回的内容
        cleaned_response = response.strip().strip("```json").strip("```").strip()
        response_data = json.loads(cleaned_response)
        
        # 验证返回数据的完整性
        required_fields = ["conversation_id", "dimension", "average_score"]
        for field in required_fields:
            if field not in response_data:
                raise ValueError(f"响应中缺少必需字段: {field}")
        
        return {
            "conversation_id": response_data["conversation_id"],
            "dimension": response_data["dimension"],
            "score": float(response_data["average_score"]),
            "teacher_responses_count": len(teacher_responses),
            "individual_scores": response_data.get("individual_scores", []),
            "evaluation_notes": response_data.get("evaluation_notes", ""),
            "error": None
        }
        
    except Exception as e:
        return {
            "conversation_id": conversation.get("conversation_id", "unknown"),
            "dimension": dimension,
            "score": 0.0,
            "teacher_responses_count": len(teacher_responses),
            "individual_scores": [],
            "evaluation_notes": "",
            "error": f"评估失败: {str(e)}"
        }


def evaluate_ablation_method(method_name: str, data_files: List[str], 
                           dimensions: List[str], llm_manager: SimpleLLMManager) -> Dict[str, Any]:
    """评估一个消融方法的所有数据集"""
    print(f"\n{'='*60}")
    print(f"开始评估消融方法: {method_name}")
    print(f"{'='*60}")
    
    method_results = {
        "method_name": method_name,
        "total_conversations": 0,
        "successful_evaluations": 0,
        "failed_evaluations": 0,
        "dimension_scores": {dim: [] for dim in dimensions},
        "conversation_details": []
    }
    
    # 遍历所有数据文件
    for file_path in data_files:
        print(f"\n正在处理文件: {file_path}")
        
        # 加载对话数据
        conversations = load_conversation_data(file_path)
        if not conversations:
            continue
        
        # 评估每个对话
        for i, conversation in enumerate(conversations):
            conversation_id = conversation.get("conversation_id", f"conv_{i}")
            print(f"  评估对话 {i+1}/{len(conversations)}: {conversation_id}")
            
            method_results["total_conversations"] += 1
            
            # 评估每个维度
            conversation_result = {
                "conversation_id": conversation_id,
                "file_path": file_path,
                "dimension_scores": {}
            }
            
            for dimension in dimensions:
                try:
                    eval_result = evaluate_single_conversation(
                        {"name": "evaluator"}, conversation, dimension, llm_manager
                    )
                    
                    if eval_result["error"] is None:
                        method_results["dimension_scores"][dimension].append(eval_result["score"])
                        conversation_result["dimension_scores"][dimension] = eval_result["score"]
                        method_results["successful_evaluations"] += 1
                    else:
                        method_results["failed_evaluations"] += 1
                        conversation_result["dimension_scores"][dimension] = 0.0
                        
                except Exception as e:
                    print(f"    维度 {dimension} 评估失败: {e}")
                    method_results["failed_evaluations"] += 1
                    conversation_result["dimension_scores"][dimension] = 0.0
            
            method_results["conversation_details"].append(conversation_result)
    
    # 计算每个维度的平均分
    method_results["average_scores"] = {}
    for dimension in dimensions:
        scores = method_results["dimension_scores"][dimension]
        if scores:
            method_results["average_scores"][dimension] = sum(scores) / len(scores)
        else:
            method_results["average_scores"][dimension] = 0.0
    
    # 计算情感支持和专业性的综合得分
    emotional_dimensions = ["语气友善性", "情绪响应敏感度", "共情表达", "安全与恰当性", "学习氛围营造"]
    professional_dimensions = ["内容准确性", "苏格拉底式引导风格", "方法适切性", "表达清晰度", "认知支持深度"]
    
    emotional_scores = [method_results["average_scores"].get(dim, 0) for dim in emotional_dimensions]
    professional_scores = [method_results["average_scores"].get(dim, 0) for dim in professional_dimensions]
    
    method_results["emotional_support_score"] = sum(emotional_scores) / len(emotional_scores) if emotional_scores else 0
    method_results["professionalism_score"] = sum(professional_scores) / len(professional_scores) if professional_scores else 0
    method_results["overall_score"] = (method_results["emotional_support_score"] + method_results["professionalism_score"]) / 2
    
    print(f"\n消融方法 {method_name} 评估完成:")
    print(f"  总对话数: {method_results['total_conversations']}")
    print(f"  成功评估: {method_results['successful_evaluations']}")
    print(f"  失败评估: {method_results['failed_evaluations']}")
    print(f"  情感支持得分: {method_results['emotional_support_score']:.2f}")
    print(f"  专业性得分: {method_results['professionalism_score']:.2f}")
    print(f"  总体得分: {method_results['overall_score']:.2f}")
    
    return method_results


def save_evaluation_results(results: List[Dict[str, Any]], output_file: str):
    """保存评估结果到Excel文件"""
    # 创建汇总表格
    summary_data = []
    for result in results:
        summary_data.append({
            "Ablation_Method": result["method_name"],
            "Total_Conversations": result["total_conversations"],
            "Successful_Evaluations": result["successful_evaluations"],
            "Failed_Evaluations": result["failed_evaluations"],
            "语气友善性": result["average_scores"]["语气友善性"],
            "情绪响应敏感度": result["average_scores"]["情绪响应敏感度"],
            "共情表达": result["average_scores"]["共情表达"],
            "安全与恰当性": result["average_scores"]["安全与恰当性"],
            "学习氛围营造": result["average_scores"]["学习氛围营造"],
            "内容准确性": result["average_scores"]["内容准确性"],
            "苏格拉底式引导风格": result["average_scores"]["苏格拉底式引导风格"],
            "方法适切性": result["average_scores"]["方法适切性"],
            "表达清晰度": result["average_scores"]["表达清晰度"],
            "认知支持深度": result["average_scores"]["认知支持深度"],
            "情感支持综合得分": result["emotional_support_score"],
            "专业性综合得分": result["professionalism_score"],
            "总体综合得分": result["overall_score"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存到Excel文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='消融实验评估汇总', index=False)
        
        # 为每个消融方法创建详细结果表
        for result in results:
            if result["conversation_details"]:
                detail_df = pd.DataFrame(result["conversation_details"])
                sheet_name = f"{result['method_name']}_详细结果"[:31]  # Excel工作表名称限制
                detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n消融实验评估结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估4种消融方法生成的教师回复质量")
    parser.add_argument("--data_dir", type=str, default=".", help="包含消融实验数据文件的目录路径")
    parser.add_argument("--output_file", type=str, default="ablation_evaluation_results.xlsx", help="输出评估结果文件路径")
    parser.add_argument("--methods", nargs="+", default=["1", "2", "3", "4"], help="要评估的消融方法编号 (1-4)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_file = args.output_file
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)
    
    # 消融方法映射
    ablation_methods = {
        "1": {"name": "w/o.Tea", "description": "移除ICECoT思维链"},
        "2": {"name": "w/o.Mod", "description": "移除监控机制"},
        "3": {"name": "w/o.Emo", "description": "移除情绪驱动"},
        "4": {"name": "w/o.Cog", "description": "移除认知驱动"}
    }
    
    # 查找消融实验数据文件
    data_files = []
    for method_id in args.methods:
        if method_id not in ablation_methods:
            print(f"警告: 无效的消融方法编号: {method_id}")
            continue
            
        method_info = ablation_methods[method_id]
        
        # 查找对应的数据目录和文件
        data_folder = data_dir / f"ablation_{method_id}_wo_{'tea' if method_id=='1' else 'mod' if method_id=='2' else 'emo' if method_id=='3' else 'cog'}_data"
        conversations_file = data_folder / f"ablation_{method_id}_conversations.json"
        
        if conversations_file.exists():
            data_files.append((f"Ablation_{method_id}_{method_info['name']}", str(conversations_file)))
        else:
            print(f"警告: 找不到消融方法 {method_id} 的数据文件: {conversations_file}")
    
    if not data_files:
        print(f"错误: 在目录 {data_dir} 中没有找到消融实验数据文件")
        print("请确保目录中包含以下格式的文件：")
        print("  - ablation_1_wo_tea_data/ablation_1_conversations.json")
        print("  - ablation_2_wo_mod_data/ablation_2_conversations.json")
        print("  - ablation_3_wo_emo_data/ablation_3_conversations.json")
        print("  - ablation_4_wo_cog_data/ablation_4_conversations.json")
        sys.exit(1)
    
    print(f"找到 {len(data_files)} 个消融实验数据文件:")
    for method, file in data_files:
        print(f"  - {method}: {Path(file).name}")
    
    # 定义评估维度
    dimensions = [
        "语气友善性",
        "情绪响应敏感度", 
        "共情表达",
        "安全与恰当性",
        "学习氛围营造",
        "内容准确性",
        "苏格拉底式引导风格",
        "方法适切性",
        "表达清晰度",
        "认知支持深度"
    ]
    
    # 初始化LLM管理器
    llm_manager = SimpleLLMManager()
    
    # 评估每个消融方法
    all_results = []
    for method_name, file_path in data_files:
        try:
            result = evaluate_ablation_method(method_name, [file_path], dimensions, llm_manager)
            all_results.append(result)
        except Exception as e:
            print(f"消融方法 {method_name} 评估失败: {e}")
            # 创建失败结果
            failed_result = {
                "method_name": method_name,
                "total_conversations": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "dimension_scores": {dim: [] for dim in dimensions},
                "conversation_details": [],
                "average_scores": {dim: 0.0 for dim in dimensions},
                "emotional_support_score": 0.0,
                "professionalism_score": 0.0,
                "overall_score": 0.0
            }
            all_results.append(failed_result)
    
    # 保存结果
    save_evaluation_results(all_results, output_file)
    
    # 输出最终排名
    print("\n" + "="*60)
    print("消融实验最终评估结果排名")
    print("="*60)
    
    sorted_results = sorted(all_results, key=lambda x: x["overall_score"], reverse=True)
    for i, result in enumerate(sorted_results):
        print(f"{i+1}. {result['method_name']}: {result['overall_score']:.2f}")
        print(f"   情感支持: {result['emotional_support_score']:.2f}, 专业性: {result['professionalism_score']:.2f}")
    
    # 输出各维度对比
    print("\n" + "="*60)
    print("各维度详细得分对比")
    print("="*60)
    for dimension in dimensions:
        print(f"\n{dimension}:")
        dim_scores = [(result['method_name'], result['average_scores'][dimension]) for result in sorted_results]
        for name, score in dim_scores:
            print(f"  {name}: {score:.2f}")


if __name__ == "__main__":
    main()
