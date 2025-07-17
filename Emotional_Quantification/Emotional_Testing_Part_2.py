import random
import json
import time
from autogen import ConversableAgent
import persona_loader

llm_config = {
    "cache_seed": None,
    "config_list": [{
        "model": "gpt-3.5-turbo",
        "base_url": "https://xh.v1api.cc/v1",
        "api_key": "sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm",
        "price": [0, 0]
    }]
}

def create_agent(prompt, name):
    return ConversableAgent(name=name, llm_config=llm_config, system_message=prompt)

# 从 persona_loader 中随机挑选学生和教师人格
student_key = random.choice(list(persona_loader.STUDENT_PERSONAS.keys()))
teacher_key = random.choice(list(persona_loader.TEACHER_PERSONAS.keys()))
student_prompt = persona_loader.STUDENT_PERSONAS[student_key]
teacher_prompt = persona_loader.TEACHER_PERSONAS[teacher_key]

agent_teacher = create_agent(teacher_prompt, name=teacher_key)
agent_student = create_agent(student_prompt, name=student_key)

conversation_history = []

# 种子消息，向教师提供初始问题以启动对话
initial_question = (
    "Question: Julie is reading a 120-page book. Yesterday she read 12 pages, "
    "and today she read twice as many pages as yesterday. "
    "If she wants to read half of the remaining pages tomorrow, how many pages should she read?"
)
messages = [{"role": "user", "content": initial_question}]

# 进行 10 轮对话，每轮包含教师发言与学生回应
for round_idx in range(1, 11):
    # 教师发言
    teacher_reply = agent_teacher.generate_reply(messages=messages)
    messages.append({"role": "assistant", "content": teacher_reply})

    # 学生回应
    student_reply = agent_student.generate_reply(messages=messages)
    messages.append({"role": "user", "content": student_reply})

    # 记录本轮对话
    conversation_history.append({
        "round": round_idx,
        "teacher": teacher_reply,
        "student": student_reply
    })

    print(f"Round {round_idx}:")
    print(" Teacher:", teacher_reply)
    print(" Student:", student_reply)
    print("-" * 50)
    time.sleep(1)

output = {
    "teacher_key": teacher_key,
    "teacher_prompt": teacher_prompt,
    "student_key": student_key,
    "student_prompt": student_prompt,
    "conversation": conversation_history
}
with open("conversation_part_2.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Dialogue and personas saved to conversation_part_2.json")
