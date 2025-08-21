# -*- coding: utf-8 -*-
"""
学生历史习题作答记录数据加载器
从Student_Record.csv文件中读取学生的历史做题记录
"""
import pandas as pd
import os
from typing import List, Dict, Any, Optional


class StudentDataLoader:
    """学生历史习题作答记录数据加载器"""
    
    def __init__(self, csv_file_path: str = "../Emotional_Quantification/data/Student_Record.csv"):
        """
        初始化数据加载器
        
        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """加载CSV数据"""
        try:
            if os.path.exists(self.csv_file_path):
                # 读取CSV文件，指定编码为utf-8
                self.data = pd.read_csv(self.csv_file_path, encoding='utf-8')
                print(f"成功加载学生数据，共{len(self.data)}条记录")
            else:
                print(f"警告：文件 {self.csv_file_path} 不存在")
                self.data = pd.DataFrame()
        except Exception as e:
            print(f"加载数据时出错：{e}")
            self.data = pd.DataFrame()
    
    def get_first_student_id(self) -> Optional[str]:
        """
        获取第一个学生的ID
        
        Returns:
            第一个学生的ID，如果没有数据则返回None
        """
        if self.data is None or self.data.empty:
            return None
        
        # 获取第一个唯一的user_id
        first_user_id = self.data['user_id'].iloc[0]
        return first_user_id
    
    def get_student_history_except_last(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取指定学生的历史习题作答记录（排除最后一条）
        
        Args:
            user_id: 学生ID
            
        Returns:
            学生历史记录列表（排除最后一条）
        """
        if self.data is None or self.data.empty:
            return []
        
        # 筛选指定学生的记录
        student_records = self.data[self.data['user_id'] == user_id].copy()
        
        if student_records.empty:
            print(f"未找到学生 {user_id} 的记录")
            return []
        
        # 按时间排序（假设有timestamp字段）
        if 'timestamp' in student_records.columns:
            student_records = student_records.sort_values('timestamp')
        
        # 转换为字典列表，只保留需要的字段
        history = []
        for _, record in student_records.iterrows():
            content = record.get('content', '')
            option = record.get('option', '')
            full_content = f"{content}\n{option}" if option else content
            history.append({
                'user_id': record.get('user_id', ''),
                'content': full_content,
                'option': record.get('option', ''),
                'concept_id': record.get('concept_id', ''),
                'is_correct': record.get('is_correct', 0)
            })
        
        # 排除最后一条记录
        if len(history) > 1:
            return history[:-1]
        else:
            return []
    
    def get_last_problem_content(self, user_id: str) -> Optional[str]:
        """
        获取指定学生的最后一道题目内容（content+option）
        """
        if self.data is None or self.data.empty:
            return None
        student_records = self.data[self.data['user_id'] == user_id].copy()
        if student_records.empty:
            return None
        if 'timestamp' in student_records.columns:
            student_records = student_records.sort_values('timestamp')
        if not student_records.empty:
            last_row = student_records.iloc[-1]
            content = last_row.get('content', '')
            option = last_row.get('option', '')
            return f"{content}\n{option}" if option else content
        return None

    def get_all_student_ids(self) -> List[str]:
        """
        获取所有学生的ID列表
        """
        if self.data is None or self.data.empty:
            return []
        
        return self.data['user_id'].unique().tolist()

    def get_random_student_id(self) -> Optional[str]:
        """
        随机获取一个学生ID
        """
        student_ids = self.get_all_student_ids()
        if student_ids:
            return random.choice(student_ids)
        return None


# 测试函数
if __name__ == "__main__":
    import random
    
    loader = StudentDataLoader()
    
    # 获取第一个学生ID
    first_user_id = loader.get_first_student_id()
    if first_user_id:
        print(f"第一个学生ID: {first_user_id}")
        
        # 获取历史记录（排除最后一条）
        history = loader.get_student_history_except_last(first_user_id)
        print(f"历史记录数量: {len(history)}")
        
        # 获取最后一道题目
        last_problem = loader.get_last_problem_content(first_user_id)
        print(f"最后一道题目: {last_problem[:100]}..." if last_problem else "无题目")
        
        # 获取所有学生ID
        all_students = loader.get_all_student_ids()
        print(f"总学生数: {len(all_students)}")
        
        # 随机选择一个学生
        random_student = loader.get_random_student_id()
        print(f"随机选择的学生ID: {random_student}")
