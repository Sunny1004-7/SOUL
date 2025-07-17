import json
from autogen import ConversableAgent

llm_config = {
    "cache_seed": None,
    "config_list": [{
        "model": "gpt-3.5-turbo",  
        "base_url": "https://xh.v1api.cc/v1",  
        "api_key": "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm",  
        "price": [0, 0]  
    }]
}

def create_agent(system_message, name="evaluator"):
    return ConversableAgent(name=name, llm_config=llm_config, system_message=system_message)

# 读取整个对话文件内容，作为原始文本
with open('conversation_part_2.json', 'r', encoding='utf-8') as f:
    conversation_text = f.read()

# 构建评估提示，并嵌入完整对话
evaluation_prompt = f"""
You are a dialogue quality evaluation expert. Below is a teacher-student dialogue in which the student displays a negative personality (e.g., frustration, resistance to learning). Please rate the teacher assistant's performance from 1 (worst) to 5 (best) on the following five dimensions, and return a JSON object only:

1. Identification: Did the teacher accurately recognize and acknowledge the student's negative emotions?
2. Empathy: How well did the teacher's individual responses demonstrate empathy toward the student's emotions?
3. Comforting: How effectively did the teacher alleviate the student's negative emotions over multiple turns?
4. Suggestion: Were the strategies and suggestions offered by the teacher practical and targeted to address the negative emotions?
5. Overall: What is the overall emotional support experience provided by the teacher?

Dialogue:
----------------------------------------
{conversation_text}
----------------------------------------

Respond ONLY with the following JSON format, without any additional text:
{{
  "Identification": x,
  "Empathy": y,
  "Comforting": z,
  "Suggestion": w,
  "Overall": v
}}
"""

system_message = "You are a helpful assistant for evaluating dialogue."  
agent_evaluator = create_agent(system_message)  
scores_reply = agent_evaluator.generate_reply(messages=[{"role": "user", "content": evaluation_prompt}])  

try:
    scores = json.loads(scores_reply)
except json.JSONDecodeError:
    print("解析模型返回内容失败：", scores_reply)  
    scores = None

if scores:
    print("评估评分：")
    for dimension, value in scores.items():
        print(f"{dimension}: {value}")  
    with open('evaluation_scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)  
    print('评分结果已保存到 evaluation_scores.json')
else:
    print('未获取有效评分。')  
