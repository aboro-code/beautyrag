"""Minimal Streamlit frontend for the BeautyRAG API: a chat box for grounded Q&A,
a search box exposing all three retrieval modes, and a similar-products lookup.

Run: streamlit run streamlit_app.py
"""

import requests
import streamlit as st

st.set_page_config(page_title="BeautyRAG", page_icon="✨", layout="wide")

with st.sidebar:
    st.title("BeautyRAG")
    base_url = st.text_input("API base URL", value="http://127.0.0.1:8000").rstrip("/")
    st.caption("Hybrid search + grounded RAG over a Sephora skincare/beauty catalog.")


def product_card(product: dict) -> None:
    price = f"${product['price_usd']:.2f}" if product.get("price_usd") is not None else "—"
    rating = f"{product['rating']:.2f}★ ({product.get('reviews_count') or 0})" if product.get("rating") is not None else "No rating"
    st.markdown(
        f"**{product['product_name']}**  \n"
        f"{product.get('brand_name') or 'Unknown brand'} · {product.get('category') or 'Uncategorized'}  \n"
        f"{price} · {rating} · score={product['score']:.3f}  \n"
        f"`{product['product_id']}`"
    )
    st.divider()


ask_tab, search_tab, recommend_tab = st.tabs(["Ask", "Search", "Recommend"])

with ask_tab:
    st.subheader("Ask a grounded question")
    question = st.text_input("Question", placeholder="What is a good vitamin C serum for brightening skin?")
    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving + generating..."):
            try:
                resp = requests.post(f"{base_url}/ask", json={"question": question}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                if data["cited_product_ids"]:
                    st.caption("Cited: " + ", ".join(data["cited_product_ids"]))
                st.caption(f"{data['latency_ms']:.0f}ms" + (" (cached)" if data.get("cached") else ""))
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

with search_tab:
    st.subheader("Search the catalog")
    col1, col2, col3 = st.columns([3, 1, 1])
    query = col1.text_input("Query", placeholder="lightweight moisturizer for oily skin")
    mode = col2.selectbox("Mode", ["hybrid", "vector", "keyword"])
    limit = col3.number_input("Limit", min_value=1, max_value=50, value=10)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                resp = requests.get(
                    f"{base_url}/search", params={"q": query, "mode": mode, "limit": limit}, timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
                st.caption(f"{len(data['results'])} results in {data['latency_ms']:.0f}ms")
                for product in data["results"]:
                    product_card(product)
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

with recommend_tab:
    st.subheader("Find similar products")
    product_id = st.text_input("Product ID", placeholder="P455368")
    if st.button("Recommend", type="primary") and product_id:
        with st.spinner("Looking up similar products..."):
            try:
                resp = requests.get(f"{base_url}/recommend/{product_id}", timeout=15)
                if resp.status_code == 404:
                    st.warning(f"Product '{product_id}' not found in the catalog.")
                else:
                    resp.raise_for_status()
                    data = resp.json()
                    for product in data["results"]:
                        product_card(product)
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
