import pandas as pd
import random
from persona_loader import STUDENT_PERSONAS

input_path = "subject_data/Student_Record.csv"
df = pd.read_csv(input_path)

personas = list(STUDENT_PERSONAS.keys())

unique_users = df['user_id'].unique()
user_persona = {uid: random.choice(personas) for uid in unique_users}

df['student_persona'] = ""  

first_indices = df.drop_duplicates(subset=['user_id'], keep='first').index

for idx in first_indices:
    uid = df.at[idx, 'user_id']
    df.at[idx, 'student_persona'] = user_persona[uid]

output_path = "subject_data/stuRec_1000_with_persona.csv"
df.to_csv(output_path, index=False)
print(f"已生成带人格标签的文件：{output_path}")
