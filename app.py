import io

import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from umap import UMAP
import plotly.express as px

SEED = 42
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTIMENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
STOPWORDS = [
    "청년", "지역", "광주", "전남", "정보", "경우", "부분", "요즘", "실제로",
    "개인적으로", "생각합니다", "좋겠습니다", "어렵다", "어렵습니다", "필요하다",
    "필요합니다", "있으면",
]

app_state = {}


def read_and_clean_csv(file_path):
    df = None
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("CSV 인코딩을 읽지 못했습니다.")

    if "text" not in df.columns:
        raise ValueError(f"'text' 컬럼이 필요합니다. 현재 컬럼: {list(df.columns)}")

    df = df.copy()
    df["text"] = df["text"].astype("string").str.strip()
    df = df.dropna(subset=["text"])
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df


def _get_cluster_keywords(df, text_col="text", cluster_col="cluster", top_n=6, stopwords=None):
    if stopwords is None:
        stopwords = []

    token_pattern = r"(?u)\b[가-힣]{2,}\b"
    cluster_ids = sorted(df[cluster_col].unique())
    documents = [
        " ".join(df[df[cluster_col] == c][text_col].astype(str))
        for c in cluster_ids
    ]

    vectorizer = TfidfVectorizer(
        token_pattern=token_pattern,
        ngram_range=(1, 2),
        stop_words=stopwords if stopwords else None,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    results = []
    for i, c in enumerate(cluster_ids):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        keywords = [feature_names[idx] for idx in top_indices]
        results.append({"cluster": c, "keywords": keywords})

    return pd.DataFrame(results)


def _get_representative_sentences(df, embeddings, kmeans, text_col="text", cluster_col="cluster", top_n=3):
    centers = kmeans.cluster_centers_
    results = []

    for c in sorted(df[cluster_col].unique()):
        idx = df.index[df[cluster_col] == c].to_numpy()
        cluster_embeddings = embeddings[idx]

        sims = cosine_similarity(cluster_embeddings, [centers[c]]).flatten()
        top_order = sims.argsort()[::-1][:top_n]

        for rank, i in enumerate(top_order, start=1):
            results.append({
                "cluster": c,
                "rank": rank,
                "similarity_to_center": sims[i],
                "text": df.iloc[idx[i]][text_col],
            })

    return pd.DataFrame(results)


def _get_silhouette_scores(embeddings, k_range=range(3, 11)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
        labels = km.fit_predict(embeddings)
        scores[k] = silhouette_score(embeddings, labels)
    recommended_k = max(scores, key=scores.get)

    fig = px.line(
        x=list(scores.keys()), y=list(scores.values()), markers=True,
        labels={"x": "k", "y": "Silhouette Score"}, title="Silhouette Score by k",
    )
    fig.add_vline(x=recommended_k, line_dash="dash", line_color="red")

    return scores, recommended_k, fig


def _label_sentiment(star_label):
    stars = int(star_label[0])
    if stars <= 2:
        return "negative"
    elif stars == 3:
        return "neutral"
    else:
        return "positive"


def _get_sentiment_summary(df, sentiment_pipe, text_col="text", cluster_col="cluster"):
    preds = sentiment_pipe(df[text_col].tolist(), truncation=True, batch_size=32)
    df = df.copy()
    df["sentiment"] = [_label_sentiment(p["label"]) for p in preds]

    sentiment_counts = df.groupby([cluster_col, "sentiment"]).size().unstack(fill_value=0)
    for col in ["negative", "neutral", "positive"]:
        if col not in sentiment_counts.columns:
            sentiment_counts[col] = 0
    sentiment_ratio = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0).reset_index()

    fig = px.bar(
        sentiment_ratio, x=cluster_col, y=["negative", "neutral", "positive"],
        title="Topic별 Sentiment 비율", barmode="stack",
    )

    return df, sentiment_ratio, fig


def build_analysis(file_path, n_clusters=7):
    df = read_and_clean_csv(file_path)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    scores, recommended_k, silhouette_fig = _get_silhouette_scores(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto")
    df["cluster"] = kmeans.fit_predict(embeddings)

    keyword_df = _get_cluster_keywords(df, stopwords=STOPWORDS)
    representative_df = _get_representative_sentences(df, embeddings, kmeans)

    summary_df = df["cluster"].value_counts().sort_index().reset_index()
    summary_df.columns = ["cluster", "count"]
    summary_df = summary_df.merge(keyword_df, on="cluster")
    top1_df = representative_df[representative_df["rank"] == 1][["cluster", "text"]].rename(
        columns={"text": "representative_text"}
    )
    summary_df = summary_df.merge(top1_df, on="cluster")

    pca_coords = PCA(n_components=2).fit_transform(embeddings)
    umap_coords = UMAP(n_components=2, random_state=SEED).fit_transform(embeddings)
    df["x_pca"], df["y_pca"] = pca_coords[:, 0], pca_coords[:, 1]
    df["x_umap"], df["y_umap"] = umap_coords[:, 0], umap_coords[:, 1]

    def _make_map(x_col, y_col, title):
        fig = px.scatter(
            df, x=x_col, y=y_col, color=df["cluster"].astype(str),
            hover_data={"text": True, x_col: False, y_col: False},
            title=title, labels={"color": "cluster"},
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        return fig

    pca_fig = _make_map("x_pca", "y_pca", "Topic Map (PCA)")
    umap_fig = _make_map("x_umap", "y_umap", "Topic Map (UMAP)")

    sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
    df, sentiment_ratio_df, sentiment_fig = _get_sentiment_summary(df, sentiment_pipe)

    state = {
        "df": df,
        "embeddings": embeddings,
        "model": model,
        "kmeans": kmeans,
        "keyword_df": keyword_df,
        "representative_df": representative_df,
        "summary_df": summary_df,
        "topic_map_fig_pca": pca_fig,
        "topic_map_fig_umap": umap_fig,
        "silhouette_scores": scores,
        "recommended_k": recommended_k,
        "silhouette_fig": silhouette_fig,
        "sentiment_ratio_df": sentiment_ratio_df,
        "sentiment_fig": sentiment_fig,
    }

    return state


def semantic_search(query, state, top_k=5, threshold=0.0):
    df = state["df"]
    embeddings = state["embeddings"]
    model = state["model"]

    query_embedding = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        if sims[idx] < threshold:
            continue
        results.append({
            "rank": rank,
            "score": sims[idx],
            "cluster": df.iloc[idx]["cluster"],
            "text": df.iloc[idx]["text"],
        })

    return pd.DataFrame(results)


def run_analysis(file_obj, n_clusters):
    if file_obj is None:
        return "CSV 파일을 업로드하세요.", None, None, None, None, None, gr.update(choices=[]), None

    state = build_analysis(file_obj.name, n_clusters=int(n_clusters))
    app_state["state"] = state

    n_texts = len(state["df"])
    summary_display = state["summary_df"][["cluster", "count", "keywords", "representative_text"]]
    recommended_text = (
        f"추천 k: {state['recommended_k']} "
        f"(silhouette score={state['silhouette_scores'][state['recommended_k']]:.4f})"
    )
    cluster_choices = sorted(state["df"]["cluster"].unique().tolist())

    return (
        f"분석된 의견 수: {n_texts}",
        summary_display,
        state["topic_map_fig_pca"],
        state["topic_map_fig_umap"],
        recommended_text,
        state["silhouette_fig"],
        gr.update(choices=cluster_choices, value=None),
        state["sentiment_fig"],
    )


def filter_by_cluster(cluster_id):
    if "state" not in app_state or cluster_id is None:
        return None
    df = app_state["state"]["df"]
    return df[df["cluster"] == cluster_id][["text"]]


def run_search(query, top_k, threshold):
    if "state" not in app_state:
        return None
    if not query or not query.strip():
        return None
    return semantic_search(query, app_state["state"], top_k=int(top_k), threshold=float(threshold))


def download_results():
    if "state" not in app_state:
        return None
    path = "analysis_result.csv"
    app_state["state"]["df"].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("## Topic Analysis & Semantic Search")

        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="CSV Upload", file_types=[".csv"])
                k_input = gr.Number(label="Number of Topics (k)", value=7, precision=0)
                analyze_btn = gr.Button("Analyze")
                recommended_k_output = gr.Textbox(label="추천 k")
                silhouette_plot = gr.Plot(label="Silhouette Score by k")

            with gr.Column():
                count_output = gr.Textbox(label="분석된 의견 수")
                summary_output = gr.Dataframe(label="Topic Summary", wrap=True)
                download_btn = gr.Button("결과 CSV 다운로드")
                download_file = gr.File(label="다운로드 파일")

        with gr.Row():
            pca_map_output = gr.Plot(label="Topic Map (PCA)")
            umap_map_output = gr.Plot(label="Topic Map (UMAP)")

        sentiment_plot = gr.Plot(label="Topic별 Sentiment 비율")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                cluster_dropdown = gr.Dropdown(label="Cluster 선택", choices=[])
                cluster_filtered_output = gr.Dataframe(label="선택한 Cluster 의견", wrap=True)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Query")
                topk_input = gr.Number(label="Top-K", value=5, precision=0)
                threshold_input = gr.Slider(
                    label="Similarity Threshold", minimum=0.0, maximum=1.0, value=0.0, step=0.01
                )
                search_btn = gr.Button("Semantic Search")

            with gr.Column():
                search_output = gr.Dataframe(label="Semantic Search Result", wrap=True)

        analyze_btn.click(
            fn=run_analysis,
            inputs=[file_input, k_input],
            outputs=[
                count_output, summary_output, pca_map_output, umap_map_output,
                recommended_k_output, silhouette_plot, cluster_dropdown, sentiment_plot,
            ],
        )

        cluster_dropdown.change(
            fn=filter_by_cluster,
            inputs=[cluster_dropdown],
            outputs=[cluster_filtered_output],
        )

        search_btn.click(
            fn=run_search,
            inputs=[query_input, topk_input, threshold_input],
            outputs=[search_output],
        )

        download_btn.click(
            fn=download_results,
            inputs=[],
            outputs=[download_file],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True, debug=True)
