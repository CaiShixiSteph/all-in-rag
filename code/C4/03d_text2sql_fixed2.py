# 03d_text2sql_fixed2.py —— 改良实验第二弹：补丁3（用完可删）
# 补丁3（治"事实层错误"）：把真实库结构作为"权威事实"直接写进每个提示词，
#   不走语义检索碰运气。真实 Text2SQL 系统把这步叫 schema linking。
# 在 03c 两个补丁的基础上：
#   - generate_sql 的上下文 = 权威真实结构 + 检索知识（冲突时以真实结构为准）
#   - fix_sql 的上下文 = 同样
#   - 每个问题开跑前，先把检索到的 top5 知识打印出来（让我们看见 LLM 拿到了什么）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), 'text2sql'))
from text2sql.text2sql_agent import SimpleText2SQLAgent
from text2sql.sql_generator import SimpleSQLGenerator
from langchain.schema import HumanMessage


# ========== 改良版生成器（在 03c 补丁2的基础上加补丁3） ==========
class SchemaAwareGenerator(SimpleSQLGenerator):
    """继承原版，重写 generate_sql 和 fix_sql：上下文头部永远先放权威真实结构"""

    def __init__(self, api_key: str = None, real_schema: str = ""):
        super().__init__(api_key=api_key)
        self.real_schema = real_schema

    def _authoritative_context(self, knowledge_results) -> str:
        """真实结构放最前面 + 检索知识殿后，冲突时以真实结构为准"""
        context = self._build_context(knowledge_results)
        return ("=== 实际数据库结构（从数据库实时读取；若与下面的检索知识冲突，以此为准） ===\n"
                f"{self.real_schema}\n\n"
                f"{context}")

    @staticmethod
    def _strip_fence(text: str) -> str:
        """剥掉 LLM 爱加的代码围栏（原版同款逻辑，抽出来共用）"""
        text = text.strip()
        if text.startswith("```sql"):
            text = text[6:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def generate_sql(self, user_query: str, knowledge_results) -> str:
        context = self._authoritative_context(knowledge_results)

        prompt = f"""你是一个SQL专家。请根据以下信息将用户问题转换为SQL查询语句。

数据库信息：
{context}

用户问题：{user_query}

要求：
1. 只返回SQL语句，不要包含任何解释
2. 确保SQL语法正确
3. 使用上下文中提供的表名和字段名
4. 如果需要JOIN，请根据表结构进行合理关联

SQL语句："""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return self._strip_fence(response.content)

    def fix_sql(self, original_sql: str, error_message: str, knowledge_results) -> str:
        context = self._authoritative_context(knowledge_results)

        prompt = f"""请修复以下SQL语句的错误。

目标数据库：SQLite（只使用SQLite支持的语法）。

数据库信息：
{context}

原始SQL：
{original_sql}

错误信息：
{error_message}

修复要求：
1. 如果错误提示某个表或函数不存在，说明检索知识可能已过时，
   请只基于"实际数据库结构"里真实存在的表重写SQL
2. 如果是语法错误，检查是否混用了其他数据库（如MySQL）的语法，
   改用SQLite的等价写法
3. 只返回修复后的SQL语句，不要包含任何解释

SQL："""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return self._strip_fence(response.content)


# ========== 建演示库（同 03b/03c 三张表） ==========
db_path = "text2sql_fixed2.db"
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

# ========== 初始化代理 + 换改良生成器 ==========
agent = SimpleText2SQLAgent(api_key=os.getenv("DEEPSEEK_API_KEY"))
agent.connect_database(db_path)
agent.load_knowledge_base()

# 读真实库结构，拼成"权威事实"文本
table_info = agent.get_table_info()
real_schema = ""
for t in table_info:
    cols = ", ".join(f"{c['name']} {c['type']}" for c in t["columns"])
    real_schema += f"表 {t['table_name']}: {cols}\n"
print("权威真实结构:\n" + real_schema)

agent.sql_generator = SchemaAwareGenerator(
    api_key=os.getenv("DEEPSEEK_API_KEY"), real_schema=real_schema)

# ========== 还是那两个雷 ==========
questions = [
    "查询最近7天的订单",
    "查询每个产品的总销售额",
]

print("=" * 60)
for q in questions:
    # 开跑前：手动检索一次，看看 LLM 会拿到什么"检索知识"
    print(f"\n问题: {q}")
    kb_results = agent.knowledge_base.search(q, 5)
    print("检索到的 top5 知识:")
    for r in kb_results:
        one_line = r["content"][:70].replace("\n", " ")
        print(f"  [{r['type']}] {one_line}")

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
