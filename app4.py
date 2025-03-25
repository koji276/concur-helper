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

# 要約が格納されているインデックス
SUMMARY_INDEX_NAME = "concur-index2"  # ① 要約インデックス
SUMMARY_NAMESPACE  = "demo-html"

# フルドキュメントが格納されているインデックス
FULL_INDEX_NAME = "concur-index"      # ② フル版インデックス
FULL_NAMESPACE  = "demo-html"

# 参考: app3.pyにあるワークフロー関連定義 (必要に応じて継承)
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
    st.title("Concur Helper ‐ 要約・詳細二段階ボット")

    # --- セッション初期化 ---
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "focus_guide" not in st.session_state:
        st.session_state["focus_guide"] = "なし"

    # Pinecone 初期化
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    # ① 要約インデックスの VectorStore
    sum_index = pc.Index(SUMMARY_INDEX_NAME)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    docsearch_summary = PineconeVectorStore(
        embedding=embeddings,
        index=sum_index,
        namespace=SUMMARY_NAMESPACE,
        text_key="chunk_text"
    )

    # ② フルインデックスの VectorStore
    full_index = pc.Index(FULL_INDEX_NAME)
    docsearch_full = PineconeVectorStore(
        embedding=embeddings,
        index=full_index,
        namespace=FULL_NAMESPACE,
        text_key="chunk_text"
    )

    # サイドバー
    st.sidebar.header("ガイドのリスト（サンプル）")
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

    # 会話履歴のアップロード機能などは app3.py と同様
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

    # ---- LLM 準備 ----
    chat_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",  # or "gpt-3.5-turbo"等
        temperature=0
    )

    # ---- QA Chain 関数 ----
    def run_summary_chain(query_text: str, conversation_history):
        """
        1) 要約インデックス(concur-index2)に問い合わせて大枠の回答を得る
        2) 関連メタデータを返す (DocNameなど)
        """
        retriever = docsearch_summary.as_retriever(search_kwargs={"k": 3})
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        result = chain({"question": query_text, "chat_history": conversation_history})
        answer = result["answer"]
        src_docs = result.get("source_documents", [])

        # メタデータ収集
        meta_list = []
        for d in src_docs:
            meta_list.append(d.metadata)

        return answer, meta_list

    def run_full_chain(query_text: str, conversation_history, filter_docname: str):
        """
        1) フルインデックス(concur-index)に DocNameフィルタ付きで問い合わせ
        2) 詳細回答を得る
        """
        # 例: メタデータ上の "DocName" が filter_docname に一致するものだけ検索
        filters = {"DocName": {"$eq": filter_docname}}
        retriever = docsearch_full.as_retriever(search_kwargs={"k": 5, "filter": filters})

        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        result = chain({"question": query_text, "chat_history": conversation_history})
        answer = result["answer"]
        src_docs = result.get("source_documents", [])
        meta_list = [d.metadata for d in src_docs]
        return answer, meta_list

    # ---- ユーザーインタフェース ----
    chat_placeholder = st.empty()

    with st.container():
        user_input = st.text_input("新しい質問を入力してください", "")
        if st.button("送信"):
            if user_input.strip():
                with st.spinner("概要を検索しています(要約インデックス)..."):
                    # 1) 要約インデックスで回答
                    summary_answer, summary_meta = run_summary_chain(
                        user_input, st.session_state["history"]
                    )
                # 回答に「詳細を見たいですか？」を付け足して返す
                # ここでは回答の最後に文言を付与して、ユーザーにYes/Noを促す簡易実装
                summary_answer += "\n\nさらに詳しい情報が必要ですか？「Yes」か「No」でお答えください。"

                # 会話履歴を更新
                st.session_state["history"].append((user_input, summary_answer))
                st.session_state["chat_messages"].append({
                    "user": user_input,
                    "assistant": summary_answer,
                    "sources": summary_meta  # 要約インデックスで参照したメタデータ
                })

    # ---- 追加のYes/No回答を処理 ----
    # もしユーザーが次に "Yes" と入力したらフル検索へ進む
    # (もちろん "Yes" と入力するタイミングや処理フローは設計次第です)
    with st.container():
        detail_input = st.text_input("続けて回答する場合、Yes / No / または別の質問を入力:", "")
        if st.button("続きの送信"):
            if detail_input.strip():
                if detail_input.strip().lower() in ["yes", "y"]:
                    # もっと詳しく知りたい → フル版を検索
                    # ただし、今はサンプルとして「最初に見つかったDocName」を例に
                    # 実運用では複数のDocNameがある場合、選択UIを出すなど要改修
                    if not st.session_state["chat_messages"]:
                        st.warning("直前の要約インデックス回答が見つかりません。")
                    else:
                        # 最新アシスタントメッセージのsourceを取得
                        last_msg = st.session_state["chat_messages"][-1]
                        last_sources = last_msg.get("sources", [])
                        if not last_sources:
                            st.warning("要約インデックスの検索結果メタデータがありません。")
                        else:
                            first_docname = last_sources[0].get("DocName", "")
                            if not first_docname:
                                st.warning("DocNameが見つかりません。詳細検索を実行できません。")
                            else:
                                # フル検索で回答
                                with st.spinner(f"詳細検索: DocName={first_docname} ..."):
                                    detail_answer, detail_meta = run_full_chain(
                                        user_input,  # or detail_input, ここは好み
                                        st.session_state["history"],
                                        filter_docname=first_docname
                                    )
                                # 会話履歴に追加
                                st.session_state["history"].append((detail_input, detail_answer))
                                st.session_state["chat_messages"].append({
                                    "user": detail_input,
                                    "assistant": detail_answer,
                                    "sources": detail_meta
                                })
                elif detail_input.strip().lower() in ["no", "n"]:
                    # 詳細不要
                    # 会話を終了 or 別処理
                    st.session_state["history"].append((detail_input, "かしこまりました。詳細は不要ですね。"))
                    st.session_state["chat_messages"].append({
                        "user": detail_input,
                        "assistant": "かしこまりました。詳細は不要ですね。"
                    })
                else:
                    # 新たな質問として扱う (再度 要約インデックス検索など)
                    with st.spinner("概要を検索しています(要約インデックス)..."):
                        summary_answer, summary_meta = run_summary_chain(
                            detail_input, st.session_state["history"]
                        )
                    summary_answer += "\n\nさらに詳しい情報が必要ですか？「Yes」か「No」でお答えください。"
                    # 会話更新
                    st.session_state["history"].append((detail_input, summary_answer))
                    st.session_state["chat_messages"].append({
                        "user": detail_input,
                        "assistant": summary_answer,
                        "sources": summary_meta
                    })

    # ---- チャット履歴の表示 ----
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
