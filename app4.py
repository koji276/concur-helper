import os
import json
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from datetime import datetime

# LangChain ライブラリ (pip install langchain_openai langchain_pinecone 等)
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv()

# --------------------------------------------------
# APIキー・環境変数
# --------------------------------------------------
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

# --------------------------------------------------
# インデックス定義
# --------------------------------------------------
SUMMARY_INDEX_NAME = "concur-index2"  # 要約用
SUMMARY_NAMESPACE  = "demo-html"

FULL_INDEX_NAME = "concur-index"      # フル用
FULL_NAMESPACE  = "demo-html"

# --------------------------------------------------
# ワークフロー系など(既存プログラムと同様)
# --------------------------------------------------
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
    st.title("Concur Helper - 1ドキュメント要約 + 全文詳細")

    # --------------------------------------------------
    # セッション初期化
    # --------------------------------------------------
    if "summary_history" not in st.session_state:
        st.session_state["summary_history"] = []  # 要約の履歴
    if "detail_history" not in st.session_state:
        st.session_state["detail_history"] = []   # 詳細の履歴

    # メモ: Step1 で取得した DocName を保持する
    if "selected_docname" not in st.session_state:
        st.session_state["selected_docname"] = None

    # --------------------------------------------------
    # Pinecone 初期化
    # --------------------------------------------------
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # 1) 要約インデックス
    sum_index = pc.Index(SUMMARY_INDEX_NAME)
    docsearch_summary = PineconeVectorStore(
        embedding=embeddings,
        index=sum_index,
        namespace=SUMMARY_NAMESPACE,
        text_key="chunk_text"
    )

    # 2) フルインデックス
    full_index = pc.Index(FULL_INDEX_NAME)
    docsearch_full = PineconeVectorStore(
        embedding=embeddings,
        index=full_index,
        namespace=FULL_NAMESPACE,
        text_key="chunk_text"
    )

    # --------------------------------------------------
    # Chat LLM
    # --------------------------------------------------
    chat_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
        temperature=0
    )

    # --------------------------------------------------
    # サイドバー
    # --------------------------------------------------
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

    # ガイドのフォーカス (不要なら削除可)
    st.sidebar.header("ガイドのフォーカス")
    focus_guide_selected = st.sidebar.selectbox(
        "特定のガイドにフォーカス",
        options=["なし"] + WORKFLOW_GUIDES,
        index=0
    )

    # 会話履歴の管理
    st.sidebar.header("会話履歴の管理")
    uploaded_file = st.sidebar.file_uploader("保存していた会話ファイルを選択 (.json)", type="json")

    if uploaded_file is not None:
        uploaded_content = uploaded_file.read()
        try:
            loaded_json = json.loads(uploaded_content)
            st.session_state["summary_history"] = loaded_json.get("summary_history", [])
            st.session_state["detail_history"] = loaded_json.get("detail_history", [])
            st.session_state["selected_docname"] = loaded_json.get("selected_docname", None)
            st.success("以前の会話履歴を復元しました！")
        except Exception as e:
            st.error(f"アップロードに失敗しました: {e}")

    def download_chat_history():
        data_to_save = {
            "summary_history": st.session_state["summary_history"],
            "detail_history": st.session_state["detail_history"],
            "selected_docname": st.session_state["selected_docname"]
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

    # --------------------------------------------------
    # メタデータフィルタの作成 (step1とstep2で使う)
    # --------------------------------------------------
    def get_summary_retriever():
        """
        要約インデックス( conur-index2 )用。
        k=1 で単一ドキュメントのみ取得。
        focus_guide_selected が "なし" でなければ、そのガイド名のフィルタ。
        """
        if focus_guide_selected != "なし":
            filter_conf = {"GuideNameJp": {"$eq": focus_guide_selected}}
            return docsearch_summary.as_retriever(search_kwargs={"k": 1, "filter": filter_conf})
        else:
            return docsearch_summary.as_retriever(search_kwargs={"k": 1})

    def get_detail_retriever():
        """
        フルインデックス( concur-index )用。
        - Step2では doc_name でフィルタし、k=999 で全チャンク取得
          (「全文を参照」させるため)
        - ただしフォーカスガイドがある場合、それもAND条件にするかどうかは要件次第。
          ここでは例としてフォーカスガイドの絞り込みはせず "selected_docname" を優先します。
        """
        if not st.session_state["selected_docname"]:
            # doc_nameが無い場合(まだStep1してない)は fallback
            return docsearch_full.as_retriever(search_kwargs={"k": 5})
        else:
            # doc_nameで絞り込む
            filter_conf = {"DocName": {"$eq": st.session_state["selected_docname"]}}
            # 全文を参照したいため k=999 (大きめに設定)
            return docsearch_full.as_retriever(search_kwargs={"k": 999, "filter": filter_conf})

    # --------------------------------------------------
    # 回答後付処理
    # --------------------------------------------------
    def post_process_answer(user_question: str, raw_answer: str) -> str:
        if ("ワークフロー" in user_question) and ("仮払い" not in user_question):
            if WORKFLOW_OVERVIEW_URL not in raw_answer:
                raw_answer += (
                    f"\n\nなお、ワークフローの全般情報については、以下のガイドもご参照ください:\n"
                    f"{WORKFLOW_OVERVIEW_URL}"
                )
        return raw_answer

    # --------------------------------------------------
    # チェーン (summary / detail)
    # --------------------------------------------------
    def run_summary_chain(query_text: str):
        """
        Step1: 要約インデックスで検索 (k=1) → ドキュメントを1つだけ取得
        """
        retriever = get_summary_retriever()
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        result = chain({"question": query_text, "chat_history": []})
        answer = post_process_answer(query_text, result["answer"])
        src_docs = result.get("source_documents", [])

        # 1件だけのはずだが念のため
        if src_docs:
            doc_name = src_docs[0].metadata.get("DocName", None)
            st.session_state["selected_docname"] = doc_name  # Step2で使う
        else:
            st.session_state["selected_docname"] = None

        meta_list = [d.metadata for d in src_docs]
        return answer, meta_list

    def run_detail_chain(query_text: str):
        """
        Step2: フルインデックスから "selected_docname" を対象に k=999で検索し、全文を参照
        """
        retriever = get_detail_retriever()
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        result = chain({"question": query_text, "chat_history": []})
        answer = post_process_answer(query_text, result["answer"])
        src_docs = result.get("source_documents", [])
        meta_list = [d.metadata for d in src_docs]
        return answer, meta_list

    # --------------------------------------------------
    # UI (2ステップ)
    # --------------------------------------------------
    st.header("Step1: 概要検索")
    st.write("質問してください。要約インデックスから1件だけ取得します。")

    with st.form(key="summary_form"):
        summary_question = st.text_input("例: 『勘定科目コードとは何ですか？』")
        do_summary = st.form_submit_button("送信（概要）")
        if do_summary and summary_question.strip():
            with st.spinner("要約インデックスを検索中 (k=1) ..."):
                answer, meta = run_summary_chain(summary_question)

            st.session_state["summary_history"].append((summary_question, answer))

            st.markdown("### 要約回答")
            st.write(answer)
            st.markdown("#### 参照した設定ガイド (1件)")
            for m in meta:
                doc_name   = m.get("DocName", "")
                guide_name = m.get("GuideNameJp", "")
                link       = m.get("FullLink", "")
                st.write(f"- **DocName**: {doc_name}")
                st.write(f"  **GuideNameJp**: {guide_name}")
                st.write(f"  **FullLink**: {link}")

            # 選ばれたDocNameを表示
            if st.session_state["selected_docname"]:
                st.success(f"⇒ '{st.session_state['selected_docname']}' に絞ってStep2で検索します。")
            else:
                st.warning("ドキュメントが見つかりませんでした。Step2での検索は全体になります。")

    st.header("Step2: 詳細検索")
    st.info(
        "上記回答で興味がある部分をコピーし、下欄に貼って送信してください。\n"
        "Step1で特定したDocNameがあれば、そのドキュメント全体(k=999)を対象に検索します。"
    )

    with st.form(key="detail_form"):
        detail_question = st.text_area("例: 上の文章から気になるフレーズなどをコピペ", height=100)
        do_detail = st.form_submit_button("送信（詳細）")
        if do_detail and detail_question.strip():
            with st.spinner("フルインデックスを検索中 (特定DocName + 全文) ..."):
                detail_answer, detail_meta = run_detail_chain(detail_question)

            st.session_state["detail_history"].append((detail_question, detail_answer))

            st.markdown("### 詳細回答")
            st.write(detail_answer)
            st.markdown("#### 参照した設定ガイド")
            for m in detail_meta:
                doc_name   = m.get("DocName", "")
                guide_name = m.get("GuideNameJp", "")
                sec1       = m.get("SectionTitle1", "")
                sec2       = m.get("SectionTitle2", "")
                link       = m.get("FullLink", "")
                st.write(f"- **DocName**: {doc_name}")
                st.write(f"  **GuideNameJp**: {guide_name}")
                st.write(f"  **SectionTitle1**: {sec1}")
                st.write(f"  **SectionTitle2**: {sec2}")
                st.write(f"  **FullLink**: {link}")

    # --------------------------------------------------
    # 会話履歴 (オプション)
    # --------------------------------------------------
    st.header("会話履歴")
    if st.checkbox("表示する"):
        st.subheader("=== 概要検索履歴 ===")
        for i, (q, a) in enumerate(st.session_state["summary_history"], start=1):
            st.markdown(f"**Q{i}**: {q}\n\n**A{i}**: {a}\n---")

        st.subheader("=== 詳細検索履歴 ===")
        for i, (q, a) in enumerate(st.session_state["detail_history"], start=1):
            st.markdown(f"**Q{i}**: {q}\n\n**A{i}**: {a}\n---")


if __name__ == "__main__":
    main()
