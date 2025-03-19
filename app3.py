import os
import json
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone

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

INDEX_NAME = "concur-index"
NAMESPACE = "demo-html"

WORKFLOW_GUIDES = [
    "ワークフロー（概要）(2023年10月14日版)",
    "ワークフロー（承認権限者）(2023年8月25日版)",
    "ワークフロー（原価対象の承認者)(2023年8月25日版)",
    "ワークフロー（メール通知）(2020年3月24日版)"
]

WORKFLOW_OVERVIEW_URL = "https://koji276.github.io/concur-docs/Exp_SG_Workflow_General-jp.html#_Toc150956193"

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
    st.title("Concur Helper ‐ 開発者支援ボット")

    # セッション初期化
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "focus_guide" not in st.session_state:
        st.session_state["focus_guide"] = "なし"

    # Pinecone & VectorStore
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    my_index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    docsearch = PineconeVectorStore(
        embedding=embeddings,
        index=my_index,
        namespace=NAMESPACE,
        text_key="chunk_text"
    )

    # -----------------------------
    # サイドバー: 「設定ガイドのリスト」ボタン (HTML+リンク)
    # -----------------------------
    st.sidebar.header("設定ガイドのリスト")
    # Markdown＋HTMLを使って、ボタン風の見た目に
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

    # -----------------------------
    # サイドバー: ガイドのフォーカス
    # -----------------------------
    st.sidebar.header("ガイドのフォーカス")
    focus_guide_selected = st.sidebar.selectbox(
        "特定のガイド名にフォーカスする場合は選択してください:",
        options=["なし"] + WORKFLOW_GUIDES,
        index=0
    )
    st.session_state["focus_guide"] = focus_guide_selected

    # -----------------------------
    # 会話履歴の管理
    # -----------------------------
    st.sidebar.header("会話履歴の管理")
    uploaded_file = st.sidebar.file_uploader("保存していた会話ファイルを選択 (.json)", type="json")

    if uploaded_file is not None:
        uploaded_content = uploaded_file.read()
        try:
            loaded_json = json.loads(uploaded_content)
            st.session_state["chat_messages"] = loaded_json.get("chat_messages", [])
            st.session_state["history"] = loaded_json.get("history", [])

            # タプル化
            new_history = []
            for item in st.session_state["history"]:
                if isinstance(item, list) and len(item) == 2:
                    new_history.append(tuple(item))
                else:
                    new_history.append(item)
            st.session_state["history"] = new_history

            st.success("以前の会話履歴を復元しました！")
        except Exception as e:
            st.error(f"アップロードに失敗しました: {e}")

    def download_chat_history():
        data_to_save = {
            "chat_messages": st.session_state["chat_messages"],
            "history": st.session_state["history"]
        }
        return json.dumps(data_to_save, ensure_ascii=False, indent=2)

    if st.sidebar.button("現在の会話を保存"):
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"chat_history_{now_str}.json"
        json_data = download_chat_history()
        st.sidebar.download_button(
            label="ダウンロード (JSON)",
            data=json_data,
            file_name=file_name,
            mime="application/json"
        )

    # -----------------------------
    # 検索ロジック
    # -----------------------------
    def get_filtered_retriever(query_text: str):
        # ユーザーのフォーカス設定＆質問内容に応じてPineconeのメタデータフィルタを切り替える
        # 1) 特定ガイドにフォーカス
        if st.session_state["focus_guide"] != "なし":
            focus_filter = {
                "GuideNameJp": {"$eq": st.session_state["focus_guide"]}
            }
            return docsearch.as_retriever(search_kwargs={"k": 3, "filter": focus_filter})

        # 2) "ワークフロー" & not "仮払い" → ワークフロー4ガイドに限定
        if ("ワークフロー" in query_text) and ("仮払い" not in query_text):
            wf_filter = {
                "GuideNameJp": {"$in": WORKFLOW_GUIDES}
            }
            return docsearch.as_retriever(search_kwargs={"k": 3, "filter": wf_filter})

        # それ以外は全体検索
        return docsearch.as_retriever(search_kwargs={"k": 3})

    chat_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o",
        temperature=0
    )

    def post_process_answer(user_question: str, raw_answer: str) -> str:
        # ワークフロー関連の質問には必ずワークフロー(概要)のURLを追記
        if "ワークフロー" in user_question:
            if WORKFLOW_OVERVIEW_URL not in raw_answer:
                raw_answer += (
                    f"\n\nなお、ワークフローの全般情報については、以下のガイドもご参照ください:\n"
                    f"{WORKFLOW_OVERVIEW_URL}"
                )
        return raw_answer

    def run_qa_chain(query_text: str, conversation_history):
        retriever = get_filtered_retriever(query_text)
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        result = chain({"question": query_text, "chat_history": conversation_history})
        final_answer = post_process_answer(query_text, result["answer"])
        return {
            "answer": final_answer,
            "source_documents": result.get("source_documents", [])
        }

    # -----------------------------
    # メイン画面: チャット入力UI
    # -----------------------------
    chat_placeholder = st.empty()

    with st.container():
        user_input = st.text_input("新しい質問を入力してください", "")
        if st.button("送信"):
            if user_input.strip():
                with st.spinner("回答を生成中..."):
                    result = run_qa_chain(user_input, st.session_state["history"])

                answer = result["answer"]
                source_info = []
                if "source_documents" in result:
                    for doc in result["source_documents"]:
                        source_info.append(doc.metadata)

                st.session_state["history"].append((user_input, answer))
                st.session_state["chat_messages"].append({
                    "user": user_input,
                    "assistant": answer,
                    "sources": source_info
                })

    # -----------------------------
    # チャット履歴表示
    # -----------------------------
    with chat_placeholder.container():
        st.subheader("=== 会話履歴 ===")
        for chat_item in st.session_state["chat_messages"]:
            user_q = chat_item["user"]
            ai_a   = chat_item["assistant"]
            srcs   = chat_item["sources"]

            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant"):
                st.write(ai_a)
                if srcs:
                    st.write("##### 参照した設定ガイド:")
                    for meta in srcs:
                        doc_name = meta.get("DocName", "")
                        guide_jp = meta.get("GuideNameJp", "")
                        sec1     = meta.get("SectionTitle1", "")
                        sec2     = meta.get("SectionTitle2", "")
                        link     = meta.get("FullLink", "")
                        st.markdown(f"- **DocName**: {doc_name}")
                        st.markdown(f"  **GuideNameJp**: {guide_jp}")
                        st.markdown(f"  **SectionTitle1**: {sec1}")
                        st.markdown(f"  **SectionTitle2**: {sec2}")
                        st.markdown(f"  **FullLink**: {link}")

if __name__ == "__main__":
    main()
