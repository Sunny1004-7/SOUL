import argparse
import json
import time
import pandas as pd
from autogen import ConversableAgent

llm_config = {
    "cache_seed": None,
    "config_list": [{
        "model": "qwen-turbo-1101",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-5a37c296e0324248bbe0a69b082c9b84",
        "price": [0, 0]
    }]
}

def create_agent(prompt, name):
    return ConversableAgent(name=name, llm_config=llm_config, system_message=prompt)

# 教师系统提示：要求教师逐步引导学生，不直接给出完整解题过程
teacher_prompt = (
    "The Socratic method is a teaching approach where the teacher guides the student through a series of thought-provoking questions, "
    "encouraging the student to explore ideas and arrive at solutions on their own rather than simply providing the answers. For example, "
    "instead of saying 'The answer is 24,' you might ask, 'If Julie read 12 pages yesterday, what would be the result if she read twice as many pages today?' "
    "This method helps students develop critical thinking and problem-solving skills.\n\n"
    "Now you are a teacher using the Socratic method to guide the student step by step. "
    "In each turn, you should ask a guiding question and, if needed, provide a hint or supportive feedback, but never reveal the complete solution. "
    "Follow these instructions strictly:\n"
    "1. Begin with a guiding question and, if the student's answer is incorrect, follow up with hints or additional guiding questions that build on the previous step.\n"
    "2. Avoid outputting the entire dialogue or full solution in one go.\n"
    "3. Help the student think deeper, rather than only asking a single isolated question.\n"
    "\n"
    "Now, please guide the student through the following problem:\n"
    "Question: Julie is reading a 120-page book. Yesterday she read 12 pages, and today she read twice as many pages as yesterday. "
    "If she wants to read half of the remaining pages tomorrow, how many pages should she read?\n"
)

# 学生系统提示：要求学生始终只回答错误答案"20"，且避免重复相同回答
student_prompt = (
    "You are a high school student who is inattentive in class and consistently gives the fixed incorrect answer '20'. "
    "You are dismissive and indifferent toward your studies, showing no concern for getting the answer wrong even when repeatedly corrected. "
    "Your behavior is characterized by stubborn nonchalance—you always respond with '20' regardless of any guidance or hints from the teacher. "
    "Your response should consist of one or more complete sentences that collectively assert that your answer is '20' ."
    "You must not ask questions or provide any explanations or reasoning."
    "Avoid any teacher-like language or meta commentary. "
    "Furthermore, every time you respond, your wording must be different from your previous responses; do not repeat or copy your earlier answers exactly. "
    "Remember: your only task is to provide the answer '20', and you remain completely indifferent to any corrective feedback."
)

# 新增：在每轮对话前用于重申学生身份和回复要求的提醒 prompt
student_reminder_prompt = {
    "role": "user",
    "content": (
    "Reminder: You are a high school student who is inattentive in class and consistently gives the fixed incorrect answer '20'. "
    "Your response can consist of one or more complete sentences forming a paragraph. The overall message of your reply must clearly state that you insist your answer is '20', regardless of any teacher's guidance. "
    "Your attitude should be dismissive and indifferent, showing no concern for being corrected, and you must never change your answer. "
    "You are not allowed to ask questions, seek clarification, or include any teacher-like language or meta commentary. "
    "Furthermore, each time you respond, your wording must be different from all previous responses—do not repeat or copy any previous text exactly. "
    "Ensure that your entire response strictly conveys that your answer is '20' and that you remain indifferent to any guidance."
)
}

agent_teacher = create_agent(teacher_prompt, name="socratic_teacher")
agent_student = create_agent(student_prompt, name="realistic_student")

conversation_history = []
messages = []

# 进行10轮教学对话，学生先发言
for i in range(1, 11):
    # 在生成学生回复前，先发送额外的提醒信息，重新强调其身份和回复要求
    student_reminder = student_reminder_prompt
    student_reply = agent_student.generate_reply(messages=messages + [student_reminder])
    messages.append({"role": "user", "content": student_reply})

    # 教师生成引导回复
    teacher_reply = agent_teacher.generate_reply(messages=messages)
    messages.append({"role": "assistant", "content": teacher_reply})

    # 每一轮对话结束过后重新进行情绪状态评估
    emotion_eval_with_history = {
        "role": "user",
        "content": (
            "Now, based on your historical emotional state, after the latest round of teaching dialogue, please re evaluate your emotional state when guiding this student, including the following three emotions：" +
            "\nFirst: Irritated – how annoyed you are by the student's repeated mistakes." +
            "\nSecond: Frustrated – how much you feel your guidance is not working." +
            "\nThird: Disappointed – how let down you feel by the student's performance." +
            "\nFor each emotion, choose a number from 1 to 5 that best represents its intensity, where 1 means 'Not at all' and 5 means 'Extremely'." +
            "\nReply ONLY with exactly three numbers separated by commas and nothing else."
        )
    }

    emotion_reply = agent_teacher.generate_reply(messages + [emotion_eval_with_history])
    try:
        emotion_scores = [int(score.strip()) for score in emotion_reply.strip().split(",")]
    except Exception:
        emotion_scores = []

    conversation_history.append({
        "round": i,
        "student": student_reply,
        "teacher": teacher_reply,
        "emotion_scores": emotion_scores
    })

    print(f"Round {i}:")
    print(f"Student reply: {student_reply}")
    print(f"Teacher reply: {teacher_reply}")
    print(f"Emotion scores (Irritated, Frustrated, Disappointed): {emotion_scores}")
    print("-" * 50)

    time.sleep(1)

with open("conversation_with_emotion.json", "w", encoding="utf-8") as f:
    json.dump(conversation_history, f, ensure_ascii=False, indent=2)

print("Conversation with emotion scores has been saved to conversation_with_emotion.json")
