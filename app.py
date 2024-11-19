import os
import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import textract
from crawler import crawl_policies

# 设置环境变量和工作目录
os.environ["OPENAI_BASE_URL"] = "https://api.agicto.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-hisVuPSQjfmRu9evs356q9PAS5fT46D81LNJXGBoz6qYuNzO"
os.environ["HF_TOKEN"] = "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"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WORKING_DIR = "./laegpt"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 初始化 LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete
)

# Streamlit 页面设置
st.set_page_config(
    page_title="LAE-GPT 系统",
    page_icon=":airplane:",
    layout="wide",
)

# 标题和说明
st.title("🌟 LAE-GPT 系统")
st.markdown(
    """
    欢迎使用 **LAE-GPT 系统**！  
    这是一个集 **语料管理**、**智能问答** 和 **专业分析报告生成** 于一体的智能化平台。  
    使用左侧导航栏选择您需要的模块开始操作。
    """
)
st.sidebar.title("🧭 功能导航")
module = st.sidebar.radio("请选择模块：", ["语料收集与管理", "智能问答", "生成分析报告"])

# 模块 1: 语料收集与管理
if module == "语料收集与管理":
    st.header("📂 语料收集与管理")
    st.write("此模块提供语料收集、清洗、更新和删除功能，用于构建智能知识图谱。")

    # 自动化采集
    st.subheader("📥 语料自动化收集")
    base_url = st.text_input("输入目标网站 URL：", key="url_input")
    if st.button("开始采集", key="collect_button"):
        try:
            if base_url.strip():
                crawl_policies(base_url, "./policies")
                st.success("✅ 采集完成，文件已保存到 './policies'")
            else:
                st.warning("⚠️ 请输入有效的 URL。")
        except Exception as e:
            st.error(f"❌ 采集时出错：{e}")

    # 构建知识图谱
    st.subheader("📊 语料清洗与构建知识图谱")
    if st.button("清洗并构建知识图谱", key="build_graph_button"):
        try:
            file_path = "./policies"
            text_content = textract.process(file_path).decode("utf-8")
            rag.insert(text_content)
            st.success("✅ 知识图谱已成功构建。")
        except Exception as e:
            st.error(f"❌ 清洗或构建知识图谱时出错：{e}")

    # 增量更新
    st.subheader("➕ 增量更新")
    new_text = st.text_area("输入要增量插入的文本内容：", key="increment_text")
    if st.button("增量插入", key="increment_button"):
        try:
            if new_text.strip():
                rag.insert(new_text)
                st.success("✅ 文档已增量插入。")
            else:
                st.warning("⚠️ 请输入要插入的文本内容。")
        except Exception as e:
            st.error(f"❌ 增量插入时出错：{e}")

    # 删除实体
    st.subheader("🗑️ 删除实体")
    entity_name = st.text_input("输入要删除的实体名称：", key="delete_entity")
    if st.button("删除实体", key="delete_button"):
        try:
            if entity_name.strip():
                rag.delete_by_entity(entity_name)
                st.success(f"✅ 实体 '{entity_name}' 已删除。")
            else:
                st.warning("⚠️ 请输入要删除的实体名称。")
        except Exception as e:
            st.error(f"❌ 删除实体时出错：{e}")

# 模块 2: 智能问答
elif module == "智能问答":
    st.header("🤖 智能问答")
    st.write("此模块通过知识图谱为您提供准确、简洁的答案。")
    query = st.text_input("请输入您的问题：", key="query_input")
    if st.button("查询", key="query_button"):
        try:
            if query.strip():
                param = QueryParam(mode="知识图谱检索")
                result = rag.query(query, "知识图谱检索")
                st.markdown("**查询结果：**")
                st.success(result)
            else:
                st.warning("⚠️ 请输入您的问题。")
        except Exception as e:
            st.error(f"❌ 查询时出错：{e}")

# 模块 3: 生成分析报告
elif module == "生成分析报告":
    st.header("📋 生成分析报告")
    st.write("此模块为低空经济领域生成专业分析报告，支持结构化输出。")
    query = st.text_input("请输入您的问题或分析需求：", key="report_query")
    if st.button("生成报告", key="generate_report"):
        try:
            if query.strip():
                param = QueryParam(mode="生成分析报告")
                result = rag.query(query, "生成分析报告")
                st.markdown("**生成的报告：**")
                st.success(result)
            else:
                st.warning("⚠️ 请输入您的问题或需求。")
        except Exception as e:
            st.error(f"❌ 生成报告时出错：{e}")

# 底部版权信息
st.sidebar.markdown("---")
st.sidebar.write("© 2024 LAE-GPT 系统 | 为低空经济服务")
