import pandas as pd
import sqlite3

# 设置 pandas 显示选项，避免长文本被截断太多（设置为显示 150 个字符）
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.max_columns', None)

print("==================== 1. CSV 文件预览 ====================")
try:
    df_csv = pd.read_csv('data/ma_corpus.csv')
    print(f"CSV 数据集总大小: {df_csv.shape[0]} 行, {df_csv.shape[1]} 列\n")
    # 随机抽取 3 行展示，比固定看前 3 行更能反映整体情况
    print(df_csv.sample(3))
except Exception as e:
    print(f"读取 CSV 出错: {e}")

print("\n==================== 2. SQLite DB 文件预览 ====================")
try:
    # 连接数据库
    conn = sqlite3.connect('data/ma_corpus.db')

    # 自动获取数据库中所有的表名
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(tables_query, conn)

    if tables.empty:
        print("数据库中没有找到任何表。")
    else:
        print(f"数据库中包含的表: {tables['name'].tolist()}\n")

        # 默认读取第一个表的内容进行预览
        target_table = tables['name'][0]
        print(f"正在预览表 '{target_table}' 的随机 3 行数据:")

        # 随机读取 3 行
        df_db = pd.read_sql_query(f"SELECT * FROM {target_table} ORDER BY RANDOM() LIMIT 3", conn)
        print(df_db)

    conn.close()
except Exception as e:
    print(f"读取 SQLite 数据库出错: {e}")