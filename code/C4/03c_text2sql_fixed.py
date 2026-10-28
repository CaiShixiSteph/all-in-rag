# 03c_text2sql_fixed.py —— 改良实验：给 agent 打两个补丁，看它能不能爬出坑（用完可删）
# 补丁1（治"地图不准"）：把真实库结构读出来，补进知识库
# 补丁2（治"信号模糊"）：继承 SimpleSQLGenerator，只重写 fix_sql 这一个方法
# 实验设计：generate_sql 保持原样（让第一次尝试还是会摔），只改"爬起来"的能力
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), 'text2sql'))
from text2sql.text2sql_agent import SimpleText2SQLAgent
from text2sql.sql_generator import SimpleSQLGenerator
from langchain.schema import HumanMessage


# ========== 补丁2：改良版修复提示词 ==========
class ImprovedSQLGenerator(SimpleSQLGenerator):
    """只重写 fix_sql 一个方法，其余（generate_sql 等）全部继承原版"""

    def fix_sql(self, original_sql: str, error_message: str, knowledge_results) -> str:
        context = self._build_context(knowledge_results)

        prompt = f"""请修复以下SQL语句的错误。

目标数据库：SQLite（只使用SQLite支持的语法）。

数据库信息：
{context}

原始SQL：
{original_sql}

错误信息：
{error_message}

修复要求：
1. 如果错误提示某个表或函数不存在，说明上下文中的表结构可能已过时，
   请只基于实际存在的表重写SQL
2. 如果是语法错误，检查是否混用了其他数据库（如MySQL）的语法，
   改用SQLite的等价写法
3. 只返回修复后的SQL语句，不要包含任何解释

SQL："""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        # 剥壳（和原版一样）
        fixed_sql = response.content.strip()
        if fixed_sql.startswith("```sql"):
            fixed_sql = fixed_sql[6:]
        if fixed_sql.startswith("```"):
            fixed_sql = fixed_sql[3:]
        if fixed_sql.endswith("```"):
            fixed_sql = fixed_sql[:-3]

        return fixed_sql.strip()


# ========== 建演示库（与 03b 同款三张表） ==========
db_path = "text2sql_fixed.db"
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

# ========== 初始化代理 ==========
agent = SimpleText2SQLAgent(api_key=os.getenv("DEEPSEEK_API_KEY"))
agent.connect_database(db_path)
agent.load_knowledge_base()

# 换掉生成器：用补丁2的改良版（generate 不变，fix 变强）
agent.sql_generator = ImprovedSQLGenerator(api_key=os.getenv("DEEPSEEK_API_KEY"))

# ========== 补丁1：把真实库结构补进知识库 ==========
print("\n补丁1: 读取真实库结构并补进知识库...")
table_info = agent.get_table_info()          # 现成接口：读真实表结构
contents, types = [], []
for t in table_info:
    content = f"真实表名: {t['table_name']}\n真实字段:\n"
    for col in t["columns"]:
        content += f"  - {col['name']} ({col['type']})\n"
    contents.append(content)
    types.append("ddl")                      # 归入"表结构信息"组
agent.knowledge_base._insert_data(contents, types)  # 借用内部插入方法（实践脚本里可以）
print(f"已补入 {len(contents)} 张真实表结构")

# ========== 还是那两个雷 ==========
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
