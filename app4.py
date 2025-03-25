import os
import json
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from datetime import datetime

# LangChain ライブラリ
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# 追加で「map_reduce」用
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

SUMMARY_INDEX_NAME = "concur-index2"  # 要約用
SUMMARY_NAMESPACE  = "demo-html"

FULL_INDEX_NAME = "concur-index"      # フル用
FULL_NAMESPACE  = "demo-html"

CUSTOM_PROMPT_TEMPLATE = """あなたはConcurドキュメントの専門家です。
以下のドキュメント情報(検索結果)とユーザーの質問を踏まえて、
ChatGPT-4モデルとして詳しくかつ分かりやすい回答を行ってください。

【要件】
- 回答は十分な説明を含み、原理や理由も分かるように解説してください。
- ユーザーが疑問を解消できるよう、段階的な説明や背景情報も交えてください。
- ただしドキュメントの原文を不要に繰り返すのは避け、ポイントのみを的確に述べてください。
- “Context:” などの文言は出さず、テキストの重複や冗長表現を可能な限り減らしてください。
- 答えが分からない場合は「わかりません」と述べてください。

ドキュメント情報:
{context}

ユーザーの質問: {question}

上記を踏まえ、ChatGPT-4モデルとして、詳しくかつ要点を押さえた回答をお願いします:
"""
custom_prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def main():
    st.title("Concur Helper - map_reduce対応版")

    # セッション
    if "summary_history" not in st.session_state:
        st.session_state["summary_history"] = []
    if "detail_history" not in st.session_state:
        st.session_state["detail_history"] = []
    if "selected_docname" not in st.session_state:
        st.session_state["selected_docname"] = None

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # 要約用
    sum_index = pc.Index(SUMMARY_INDEX_NAME)
    docsearch_summary = PineconeVectorStore(
        embedding=embeddings,
        index=sum_index,
        namespace=SUMMARY_NAMESPACE,
        text_key="chunk_text"
    )

    # フル用
    full_index = pc.Index(FULL_INDEX_NAME)
    docsearch_full = PineconeVectorStore(
        embedding=embeddings,
        index=full_index,
        namespace=FULL_NAMESPACE,
        text_key="chunk_text"
    )

    chat_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
        temperature=0
    )

    # サイドバー
    st.sidebar.header("設定ガイドのリスト")
    st.sidebar.markdown(
        """
        <a href="https://koji276.github.io/concur-docs/index.htm" target="_blank">
            <button style="font-size: 1rem; padding: 0.5em 1em; color: black;">
                標準ガイドリスト
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

    # 履歴アップロードなど (省略可)
    uploaded_file = st.sidebar.file_uploader("保存していた会話ファイル (json)", type="json")
    if uploaded_file is not None:
        data = uploaded_file.read()
        try:
            loaded = json.loads(data)
            st.session_state["summary_history"] = loaded.get("summary_history", [])
            st.session_state["detail_history"] = loaded.get("detail_history", [])
            st.session_state["selected_docname"] = loaded.get("selected_docname", None)
            st.success("会話履歴を復元しました。")
        except Exception as e:
            st.error(f"読み込み失敗: {e}")

    def download_history():
        data_save = {
            "summary_history": st.session_state["summary_history"],
            "detail_history": st.session_state["detail_history"],
            "selected_docname": st.session_state["selected_docname"]
        }
        return json.dumps(data_save, ensure_ascii=False, indent=2)

    if st.sidebar.button("現在の会話を保存"):
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"history_{now_str}.json"
        st.sidebar.download_button(
            "ダウンロード (JSON)",
            download_history(),
            file_name,
            "application/json"
        )

    # 1) 要約のRetriever (k=1)
    def get_summary_retriever():
        return docsearch_summary.as_retriever(search_kwargs={"k": 1})

    # 2) 詳細のRetriever
    def get_detail_retriever():
        if not st.session_state["selected_docname"]:
            # doc指定なし => 全体検索( k=5 等 )だが map_reduce は変わらず。
            return docsearch_full.as_retriever(search_kwargs={"k": 5})
        else:
            filter_conf = {"DocName": {"$eq": st.session_state["selected_docname"]}}
            # 大きめ k=999
            return docsearch_full.as_retriever(search_kwargs={"k": 999, "filter": filter_conf})

    # デフォルトのstuff方式は長文になるとトークン超過しやすいので map_reduce チェーンへ
    def create_conversational_chain(retriever):
        return ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=retriever,
            # stuff → map_reduce に変更
            combine_docs_chain=load_qa_chain(chat_llm, chain_type="map_reduce"),
            # あるいは "refine" でもOK
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )

    # 要約チェーン
    def run_summary_chain(question: str):
        chain = create_conversational_chain(get_summary_retriever())
        result = chain({"question": question, "chat_history": []})
        answer = result["answer"]
        docs = result["source_documents"]
        meta_list = [d.metadata for d in docs]

        # 1件だけ
        if docs:
            doc_name = docs[0].metadata.get("DocName", None)
            st.session_state["selected_docname"] = doc_name
        else:
            st.session_state["selected_docname"] = None

        return answer, meta_list

    # 詳細チェーン
    def run_detail_chain(question: str):
        chain = create_conversational_chain(get_detail_retriever())
        result = chain({"question": question, "chat_history": []})
        answer = result["answer"]
        docs = result["source_documents"]
        meta_list = [d.metadata for d in docs]
        return answer, meta_list

    # --- UI ---
    st.header("Step1: 要約検索 (k=1)")
    st.write("1件だけ取得し、そのDocNameをStep2で検索対象とします。")

    with st.form("summary_form"):
        summary_q = st.text_input("概要質問を入力してください")
        submit_sum = st.form_submit_button("送信 (要約)")
        if submit_sum and summary_q.strip():
            with st.spinner("要約インデックスを検索中..."):
                ans, meta = run_summary_chain(summary_q)
            st.session_state["summary_history"].append((summary_q, ans))

            st.subheader("要約回答")
            st.write(ans)
            st.write("参照メタデータ:")
            for m in meta:
                st.write(f"- DocName: {m.get('DocName')}")
                st.write(f"  GuideNameJp: {m.get('GuideNameJp')}")
                st.write(f"  FullLink: {m.get('FullLink')}")
            if st.session_state["selected_docname"]:
                st.success(f"=> DocName='{st.session_state['selected_docname']}' をStep2で全文検索します。")

    st.header("Step2: 詳細検索 (map_reduce)")
    st.write("Step1 で選ばれた DocName 全文を対象に検索します。")
    st.info("上の回答から、気になる文をコピーして下に貼って詳細質問してください。")

    with st.form("detail_form"):
        detail_q = st.text_area("詳細質問 (コピペ推奨)", height=120)
        submit_det = st.form_submit_button("送信 (詳細)")
        if submit_det and detail_q.strip():
            with st.spinner("フルインデックスを map_reduce で検索中..."):
                ans, meta = run_detail_chain(detail_q)
            st.session_state["detail_history"].append((detail_q, ans))

            st.subheader("詳細回答")
            st.write(ans)
            st.write("参照メタデータ:")
            for m in meta:
                st.write(f"- DocName: {m.get('DocName')}")
                st.write(f"  GuideNameJp: {m.get('GuideNameJp')}")
                st.write(f"  SectionTitle1: {m.get('SectionTitle1')}")
                st.write(f"  SectionTitle2: {m.get('SectionTitle2')}")
                st.write(f"  FullLink: {m.get('FullLink')}")

    # 履歴確認
    st.header("会話履歴")
    if st.checkbox("表示する"):
        st.subheader("=== 要約 ===")
        for i, (q, a) in enumerate(st.session_state["summary_history"], start=1):
            st.markdown(f"**Q{i}**: {q}\n\n**A{i}**: {a}\n---")

        st.subheader("=== 詳細 ===")
        for i, (q, a) in enumerate(st.session_state["detail_history"], start=1):
            st.markdown(f"**Q{i}**: {q}\n\n**A{i}**: {a}\n---")


if __name__ == "__main__":
    main()
