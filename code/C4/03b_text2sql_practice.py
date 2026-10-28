# 03b_text2sql_practice.py —— 动手实践：踩作者埋的两个雷（用完可删）
# 雷1 "查询最近7天的订单"：知识库示例是 MySQL 语法，SQLite 不认识 → 看 fix_sql 修复链
# 雷2 "查询每个产品的总销售额"：正确解法要 JOIN order_items，但真实库没这张表 → 看 LLM 怎么摔、怎么爬
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), 'text2sql'))
from text2sql.text2sql_agent import SimpleText2SQLAgent

# ========== 建演示库（与 03_text2sql_demo.py 同款三张表） ==========
db_path = "text2sql_practice.db"
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE users (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE,
    age INTEGER, city TEXT)""")
cursor.execute("""CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT,
    price REAL, stock INTEGER)""")
cursor.execute("""CREATE TABLE orders (
    id INTEGER PRIMARY KEY, user_id INTEGER, product_id INTEGER,
    quantity INTEGER, order_date TEXT, total_price REAL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id))""")

users_data = [
    (1, '张三', 'zhangsan@email.com', 25, '北京'),
    (2, '李四', 'lisi@email.com', 32, '上海'),
    (3, '王五', 'wangwu@email.com', 28, '广州'),
    (4, '赵六', 'zhaoliu@email.com', 35, '深圳'),
    (5, '陈七', 'chenqi@email.com', 29, '杭州'),
]
products_data = [
    (1, 'iPhone 15', '电子产品', 7999.0, 50),
    (2, 'MacBook Pro', '电子产品', 12999.0, 20),
    (3, 'Nike运动鞋', '服装', 599.0, 100),
    (4, '办公椅', '家具', 899.0, 30),
    (5, '台灯', '家具', 199.0, 80),
    (6, 'iPad', '电子产品', 3999.0, 40),
    (7, 'Adidas外套', '服装', 399.0, 60),
]
orders_data = [
    (1, 1, 1, 1, '2024-01-15', 7999.0),
    (2, 2, 3, 2, '2024-01-16', 1198.0),
    (3, 3, 5, 1, '2024-01-17', 199.0),
    (4, 1, 2, 1, '2024-01-18', 12999.0),
    (5, 4, 4, 1, '2024-01-19', 899.0),
    (6, 5, 6, 1, '2024-01-20', 3999.0),
    (7, 2, 7, 1, '2024-01-21', 399.0),
]
cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users_data)
cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products_data)
cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders_data)
conn.commit()
conn.close()
print(f"演示数据库已创建: {db_path}")

# ========== 初始化代理 + 加载知识库 ==========
agent = SimpleText2SQLAgent(api_key=os.getenv("DEEPSEEK_API_KEY"))
agent.connect_database(db_path)
agent.load_knowledge_base()

# ========== 两个刁钻问题 ==========
questions = [
    "查询最近7天的订单",
    "查询每个产品的总销售额",
]

print("\n" + "=" * 60)
for q in questions:
    result = agent.query(q)
    print("-" * 60)
    if result["success"]:
        print(f"最终成功! 重试次数: {result['retry_count']}")
        rows = result["results"]["rows"]
        print(f"返回 {len(rows)} 行:")
        for row in rows[:3]:
            print("  ", row)
    else:
        print(f"最终失败: {result['error']}")
        print(f"最后的SQL: {result['sql']}")
    print("-" * 60)

# ========== 清场 ==========
agent.cleanup()
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"已删除演示数据库: {db_path}")
