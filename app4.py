import os
import json
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from datetime import datetime

# --------------------------------------------------
# ★ Streamlit のページ設定を最初に呼び出す ★
# --------------------------------------------------
st.set_page_config(layout="wide")

# LangChainライブラリ（インポートはこのままでOK）
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv()

# --------------------------------------------------
# APIキー・環境変数を読み込み
# --------------------------------------------------
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

# --------------------------------------------------
# インデックスの定義
# --------------------------------------------------
SUMMARY_INDEX_NAME = "concur-index2"   # 要約インデックス
SUMMARY_NAMESPACE  = "demo-html"

FULL_INDEX_NAME = "concur-index"       # フルインデックス
FULL_NAMESPACE  = "demo-html"

# --------------------------------------------------
# ワークフロー系の例
# --------------------------------------------------
WORKFLOW_GUIDES = [
 "ワークフロー（概要）(2023年10月14日版)",
 "ワークフロー（承認権限者）(2023年8月25日版)",
 "ワークフロー（原価対象の承認者)(2023年8月25日版)",
 "ワークフロー（メール通知）(2020年3月24日版)"
]
WORKFLOW_OVERVIEW_URL = "https://koji276.github.io/concur-docs/Exp_SG_Workflow_General-jp.html#_Toc150956193"

# --------------------------------------------------
# プロンプトテンプレート
# --------------------------------------------------
CUSTOM_PROMPT_TEMPLATE = """あなたはConcurドキュメントの専門家です。
以下のドキュメント情報(検索結果)とユーザーの質問を踏まえて、
ChatGPT-4モデルとして詳しくかつわかりやすい回答を行ってください。

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
    st.title("Concur Helper - 開発者支援ボット")

    # --------------------------------------------------
    # セッション初期化
    # --------------------------------------------------
    if "summary_history" not in st.session_state:
        st.session_state["summary_history"] = []
    if "detail_history" not in st.session_state:
        st.session_state["detail_history"] = []

    # --------------------------------------------------
    # Pinecone 初期化 & VectorStore
    # --------------------------------------------------
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # 要約インデックス
    sum_index = pc.Index(SUMMARY_INDEX_NAME)
    docsearch_summary = PineconeVectorStore(
        embedding=embeddings,
        index=sum_index,
        namespace=SUMMARY_NAMESPACE,
        text_key="chunk_text"
    )

    # フルインデックス
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

    # フォーカスガイドの選択
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
            st.session_state["detail_history"]  = loaded_json.get("detail_history", [])
            st.success("以前の会話履歴を復元しました！")
        except Exception as e:
            st.error(f"アップロードに失敗しました: {e}")

    def download_chat_history():
        data_to_save = {
            "summary_history": st.session_state["summary_history"],
            "detail_history": st.session_state["detail_history"]
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
    # Retriever
    # --------------------------------------------------
    def get_summary_retriever():
        if focus_guide_selected != "なし":
            filter_conf = {"GuideNameJp": {"$eq": focus_guide_selected}}
            return docsearch_summary.as_retriever(search_kwargs={"k": 3, "filter": filter_conf})
        else:
            return docsearch_summary.as_retriever(search_kwargs={"k": 3})

    def get_detail_retriever():
        if focus_guide_selected != "なし":
            filter_conf = {"GuideNameJp": {"$eq": focus_guide_selected}}
            return docsearch_full.as_retriever(search_kwargs={"k": 5, "filter": filter_conf})
        else:
            return docsearch_full.as_retriever(search_kwargs={"k": 5})

    def post_process_answer(user_question: str, raw_answer: str) -> str:
        if ("ワークフロー" in user_question) and ("仮払い" not in user_question):
            if WORKFLOW_OVERVIEW_URL not in raw_answer:
                raw_answer += (
                    f"\n\nなお、ワークフローの全般情報については、以下のガイドもご参照ください:\n"
                    f"{WORKFLOW_OVERVIEW_URL}"
                )
        return raw_answer

    def run_summary_chain(query_text: str):
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
        meta_list = [d.metadata for d in src_docs]
        return answer, meta_list

    def run_detail_chain(query_text: str):
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
    # カラムで左右分割 + カラム内スクロールを実装する
    # --------------------------------------------------
    # ページを左右 1:2 の比率で分割
    col_left, col_right = st.columns([1, 2])

    # カラム内でスクロールさせるための CSS を埋め込み
    st.markdown(
        """
        <style>
        .left-column {
            max-height: 90vh;      /* 左カラムの最大高さを画面高の9割に */
            overflow-y: auto;      /* 縦スクロールバー */
            padding-right: 1rem;   /* 見やすいように右余白 */
        }
        .right-column {
            max-height: 90vh; 
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -------------------------
    # 左カラム: 入力フォームなど
    # -------------------------
    with col_left:
        st.markdown('<div class="left-column">', unsafe_allow_html=True)

        st.subheader("Step1: 概要検索")
        st.write("大まかな質問をどうぞ。回答後、ピンポイントで気になる部分をコピーして「詳細検索」へ。")

        with st.form(key="summary_form"):
            summary_question = st.text_input("例: 『勘定科目コードの概要』『元帳の作業手順』『ワークフローの設定』")
            do_summary = st.form_submit_button("送信 (概要検索)")
            if do_summary and summary_question.strip():
                with st.spinner("回答（概要）を作成中..."):
                    answer, meta = run_summary_chain(summary_question)

                st.session_state["summary_history"].append({
                    "question": summary_question,
                    "answer": answer,
                    "meta": meta
                })

                st.markdown("### 回答（概要）")
                st.write(answer)
                st.write("#### 参照すべき設定ガイド:")
                for m in meta:
                    doc_name   = m.get("DocName", "")
                    guide_name = m.get("GuideNameJp", "")
                    link       = m.get("FullLink", "")
                    st.markdown(f"- **DocName**: {doc_name}")
                    st.markdown(f"  **GuideNameJp**: {guide_name}")
                    st.markdown(f"  **FullLink**: {link}")
                st.write("---")

        st.subheader("Step2: 詳細検索")
        st.info("上の回答からもっと詳しく知りたい部分(パラグラフなど)をコピーして、下に貼り付けます。")

        with st.form(key="detail_form"):
            detail_question = st.text_area("詳しく知りたい内容をコピペして検索", height=120)
            do_detail = st.form_submit_button("送信 (詳細検索)")
            if do_detail and detail_question.strip():
                with st.spinner("回答（詳細）を作成中..."):
                    detail_answer, detail_meta = run_detail_chain(detail_question)

                st.session_state["detail_history"].append({
                    "question": detail_question,
                    "answer": detail_answer,
                    "meta": detail_meta
                })

                st.markdown("### 詳細な回答")
                st.write(detail_answer)
                st.write("#### 参照すべき設定ガイド:")
                for m in detail_meta:
                    doc_name   = m.get("DocName", "")
                    guide_name = m.get("GuideNameJp", "")
                    sec1       = m.get("SectionTitle1", "")
                    sec2       = m.get("SectionTitle2", "")
                    link       = m.get("FullLink", "")
                    st.markdown(f"- **DocName**: {doc_name}")
                    st.markdown(f"  **GuideNameJp**: {guide_name}")
                    st.markdown(f"  **SectionTitle1**: {sec1}")
                    st.markdown(f"  **SectionTitle2**: {sec2}")
                    st.markdown(f"  **FullLink**: {link}")
                st.write("---")

        st.subheader("Step3: 設定ガイド検索")
        st.info("上記のリンク先をクリックすると、関連情報や開発設定画面を参照できます。")

        st.markdown('</div>', unsafe_allow_html=True)  # left-column 終了

    # -------------------------
    # 右カラム: 会話履歴表示
    # -------------------------
    with col_right:
        st.markdown('<div class="right-column">', unsafe_allow_html=True)

        st.subheader("会話履歴（概要・詳細）")
        st.write("これまで送信したQ&Aの履歴です。")

        if st.checkbox("履歴を表示する"):
            st.markdown("### 概要のQ&A")
            for i, item in enumerate(st.session_state["summary_history"], start=1):
                q = item["question"]
                a = item["answer"]
                meta_list = item.get("meta", [])

                st.markdown(f"**Q{i}**: {q}")
                st.markdown(f"**A{i}**: {a}")

                if meta_list:
                    st.write("#### 参照すべき設定ガイド:")
                    for m in meta_list:
                        doc_name   = m.get("DocName", "")
                        guide_name = m.get("GuideNameJp", "")
                        link       = m.get("FullLink", "")
                        st.markdown(f"- **DocName**: {doc_name}")
                        st.markdown(f"  **GuideNameJp**: {guide_name}")
                        st.markdown(f"  **FullLink**: {link}")
                st.write("---")

            st.markdown("### 詳細のQ&A")
            for i, item in enumerate(st.session_state["detail_history"], start=1):
                q = item["question"]
                a = item["answer"]
                meta_list = item.get("meta", [])

                st.markdown(f"**Q{i}**: {q}")
                st.markdown(f"**A{i}**: {a}")

                if meta_list:
                    st.write("#### 参照すべき設定ガイド:")
                    for m in meta_list:
                        doc_name   = m.get("DocName", "")
                        guide_name = m.get("GuideNameJp", "")
                        sec1       = m.get("SectionTitle1", "")
                        sec2       = m.get("SectionTitle2", "")
                        link       = m.get("FullLink", "")
                        st.markdown(f"- **DocName**: {doc_name}")
                        st.markdown(f"  **GuideNameJp**: {guide_name}")
                        st.markdown(f"  **SectionTitle1**: {sec1}")
                        st.markdown(f"  **SectionTitle2**: {sec2}")
                        st.markdown(f"  **FullLink**: {link}")
                st.write("---")

        st.markdown('</div>', unsafe_allow_html=True)  # right-column 終了


# --------------------------------------------------
# メイン
# --------------------------------------------------
if __name__ == "__main__":
    main()
