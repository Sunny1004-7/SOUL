import random
import json
import time
from autogen import ConversableAgent
import persona_loader

"""
Emotional Testing Part 2: 动态性格发展测试
- 学生从初始人格开始，但性格会随着对话自然发展
- 模拟真实学生在学习过程中的性格和情绪变化
- 教师使用标准苏格拉底教学范式
- 生成包含性格发展和学生情感量化的对话数据集
"""

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

# 从 persona_loader 中随机挑选学生人格，教师使用标准苏格拉底教学范式
student_key = random.choice(list(persona_loader.STUDENT_PERSONAS.keys()))
student_persona = persona_loader.STUDENT_PERSONAS[student_key]

# 学生初始发言prompt：体现其人格
student_initial_prompt = f"""You are a high school student with the following personality and characteristics:

{student_persona}

Based on your personality, you should:
1. Respond naturally according to your character traits
2. Show appropriate emotional reactions to the teacher's guidance
3. Express your thoughts, feelings, and attitudes about learning
4. Engage with the mathematical problem in a way that reflects your personality
5. Maintain consistency with your character throughout the conversation

Remember to stay in character and respond as a real student would, based on your personality profile."""

# 学生后续发言prompt：基于初始人格，但允许性格自然发展
student_followup_prompt = f"""You are a high school student who started with the following initial personality and characteristics:

{student_persona}

However, like a real student, your personality and emotional state can naturally evolve and change based on:
1. The teacher's guidance and teaching style
2. Your learning progress and understanding
3. The difficulty level of the problems
4. Your emotional reactions to success or failure
5. The overall learning experience

When responding:
1. Use your initial personality as a foundation, but don't rigidly stick to it
2. Show natural emotional development and personality growth
3. Respond authentically to the teacher's latest guidance
4. Express how your feelings and attitudes might have changed through the learning process
5. Consider the conversation history and respond appropriately to the teacher's latest guidance
6. Be a dynamic, evolving student character, not a static personality template

Remember: You are a living, breathing student whose personality naturally develops through the learning experience. Stay authentic to how a real student would grow and change during a tutoring session."""

# 教师使用标准的苏格拉底教学范式，不设定特定人格
teacher_prompt = "You are a standard Socratic AI teacher agent. Use the Socratic method to guide students through problems step by step. Ask leading questions, encourage critical thinking, and help students discover answers themselves. Do not show emotional reactions or personality traits - focus purely on educational guidance."

agent_teacher = create_agent(teacher_prompt, name="socratic_teacher")
agent_student_initial = create_agent(student_initial_prompt, name="student_initial")
agent_student_followup = create_agent(student_followup_prompt, name="student_followup")

conversation_history = []
messages = []

# 进行10轮对话，学生先发言
for round_idx in range(1, 11):
    if round_idx == 1:
        # 第一轮：学生根据初始人格prompt发言，建立基础性格特征
        student_reply = agent_student_initial.generate_reply(messages=messages)
    else:
        # 后续轮次：学生基于初始人格，但允许性格自然发展，根据对话内容动态调整
        student_reply = agent_student_followup.generate_reply(messages=messages)
    
    messages.append({"role": "user", "content": student_reply})

    # 教师始终使用苏格拉底教学范式回应
    teacher_reply = agent_teacher.generate_reply(messages=messages)
    messages.append({"role": "assistant", "content": teacher_reply})

    # 学生情感状态评估
    student_emotion_eval = {
        "role": "user",
        "content": (
            "Based on the conversation history and your current emotional state, please evaluate your emotional state as a student, including the following three emotions:" +
            "\nFirst: Frustrated – how frustrated you feel about the learning process or the teacher's guidance." +
            "\nSecond: Anxious – how anxious or worried you feel about your performance or understanding." +
            "\nThird: Disengaged – how disconnected or uninterested you feel about the current lesson." +
            "\nFor each emotion, choose a number from 1 to 5 that best represents its intensity, where 1 means 'Not at all' and 5 means 'Extremely'." +
            "\nReply ONLY with exactly three numbers separated by commas and nothing else."
        )
    }

    # 获取学生情感评分
    student_emotion_reply = agent_student_followup.generate_reply(messages + [student_emotion_eval])
    try:
        student_emotion_scores = [int(score.strip()) for score in student_emotion_reply.strip().split(",")]
    except Exception:
        student_emotion_scores = []

    # 记录本轮对话，包含学生情感状态
    conversation_history.append({
        "round": round_idx,
        "student": student_reply,
        "teacher": teacher_reply,
        "student_emotion_scores": student_emotion_scores
    })

    print(f"Round {round_idx}:")
    print(" Student:", student_reply)
    print(" Teacher:", teacher_reply)
    print(f" Student emotions (Frustrated, Anxious, Disengaged): {student_emotion_scores}")
    print("-" * 50)
    time.sleep(1)

# 数据格式：包含学生人格、对话内容和学生情感状态
output = {
    "student_persona": student_persona,
    "conversation": conversation_history
}

with open("conversation_part_2.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Dialogue with student emotion scores has been saved to conversation_part_2.json")
