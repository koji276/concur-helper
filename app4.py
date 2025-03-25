import os
import json
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone

# LangChain の各種クラス
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

# -- 要約インデックス (概要用) --
SUMMARY_INDEX_NAME = "concur-index2"
SUMMARY_NAMESPACE  = "demo-html"

# -- フルインデックス (詳細用) --
FULL_INDEX_NAME = "concur-index"
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
    st.title("Concur Helper - 要約・詳細コピペ版")

    # -- セッション管理 --
    if "summary_history" not in st.session_state:
        st.session_state["summary_history"] = []  # 要約用の会話履歴
    if "detail_history" not in st.session_state:
        st.session_state["detail_history"] = []   # 詳細用の会話履歴

    # Pinecone 初期化
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # 1) 要約インデックス (concur-index2)
    sum_index = pc.Index(SUMMARY_INDEX_NAME)
    docsearch_summary = PineconeVectorStore(
        embedding=embeddings,
        index=sum_index,
        namespace=SUMMARY_NAMESPACE,
        text_key="chunk_text"
    )

    # 2) フルインデックス (concur-index)
    full_index = pc.Index(FULL_INDEX_NAME)
    docsearch_full = PineconeVectorStore(
        embedding=embeddings,
        index=full_index,
        namespace=FULL_NAMESPACE,
        text_key="chunk_text"
    )

    # LLM
    chat_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
        temperature=0
    )

    # -- QA Chain 作成関数 --
    def run_summary_chain(query_text: str, conversation_history=None):
        """要約インデックスを使って概要回答を生成"""
        if conversation_history is None:
            conversation_history = []
        retriever = docsearch_summary.as_retriever(search_kwargs={"k": 3})
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        result = chain({"question": query_text, "chat_history": conversation_history})
        answer = result["answer"]
        src_docs = result["source_documents"]
        meta_list = [d.metadata for d in src_docs]
        return answer, meta_list

    def run_detail_chain(query_text: str, conversation_history=None):
        """フルインデックスを使って詳細回答を生成"""
        if conversation_history is None:
            conversation_history = []
        retriever = docsearch_full.as_retriever(search_kwargs={"k": 5})
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        result = chain({"question": query_text, "chat_history": conversation_history})
        answer = result["answer"]
        src_docs = result["source_documents"]
        meta_list = [d.metadata for d in src_docs]
        return answer, meta_list

    # -- UIレイアウト --
    st.markdown("### Step1: まず、知りたい内容を入力してください。(要約インデックスで回答)")

    with st.form(key="summary_form"):
        summary_question = st.text_input("例: 『勘定科目コードとは何ですか？』など")
        submit_summary = st.form_submit_button("送信 (概要)")
        if submit_summary and summary_question.strip():
            with st.spinner("要約インデックスから回答中..."):
                answer, meta = run_summary_chain(summary_question, st.session_state["summary_history"])
            # 履歴に追加
            st.session_state["summary_history"].append((summary_question, answer))
            # 画面表示
            st.markdown("#### 【要約インデックスからの回答】")
            st.write(answer)
            st.write("参照した設定ガイド:")
            for m in meta:
                st.write(f"- DocName: {m.get('DocName')}")
                st.write(f"  GuideNameJp: {m.get('GuideNameJp')}")
                st.write(f"  FullLink: {m.get('FullLink')}")
            st.markdown("---")
            st.info("さらに詳細を知りたい部分があれば、下記の「Step2」にコピペして検索してください。")

    st.markdown("### Step2: 上記回答から興味のある部分をコピーして、下記に貼り付けてください。（フルインデックスで詳細検索）")

    with st.form(key="detail_form"):
        detail_question = st.text_area("コピペしたいパラグラフや単語を入力してください", height=100)
        submit_detail = st.form_submit_button("送信 (詳細)")
        if submit_detail and detail_question.strip():
            with st.spinner("フルインデックスから詳細を検索中..."):
                detail_answer, detail_meta = run_detail_chain(detail_question, st.session_state["detail_history"])
            # 履歴に追加
            st.session_state["detail_history"].append((detail_question, detail_answer))
            # 画面表示
            st.markdown("#### 【フルインデックスからの詳細回答】")
            st.write(detail_answer)
            st.write("参照した設定ガイド:")
            for m in detail_meta:
                st.write(f"- DocName: {m.get('DocName')}")
                st.write(f"  GuideNameJp: {m.get('GuideNameJp')}")
                st.write(f"  SectionTitle1: {m.get('SectionTitle1')}")
                st.write(f"  SectionTitle2: {m.get('SectionTitle2')}")
                st.write(f"  FullLink: {m.get('FullLink')}")
            st.markdown("---")


    # 必要に応じてStep1,Step2 の履歴を表示する場合はこちら
    if st.checkbox("会話履歴を表示"):
        st.subheader("【要約インデックスのやりとり】")
        for q, a in st.session_state["summary_history"]:
            st.markdown(f"**User**: {q}\n\n**AI**: {a}\n---")

        st.subheader("【フルインデックスのやりとり】")
        for q, a in st.session_state["detail_history"]:
            st.markdown(f"**User**: {q}\n\n**AI**: {a}\n---")


if __name__ == "__main__":
    main()

