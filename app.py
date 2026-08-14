import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import tempfile
import os
import io

st.set_page_config(
    page_title="RNAseq Explorer",
    page_icon="🧬",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
.stApp { background-color: #f0ece3; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.05em; }
.stButton > button {
    background-color: #2a52cc; color: white; border: none;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.1em;
    text-transform: uppercase; font-weight: 700; padding: 0.6em 2em;
}
.stButton > button:hover { background-color: #1e3fa0; color: white; }
div[data-testid="stSidebar"] { background-color: #e8e4db; border-right: 1.5px dashed #c0bbb0; }
.step-header {
    font-size: 0.7em; letter-spacing: 0.25em; text-transform: uppercase;
    color: #2a52cc; margin-bottom: 0.5rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(show_spinner="Fetching GEO dataset...")
def load_geo_data(accession):
    """Download and parse a GEO dataset.

    Tries two strategies:
    1. Standard series matrix (microarray / older RNA-seq with embedded values)
    2. Supplementary count matrix files (.csv.gz, .tsv.gz, .txt.gz) common in
       modern RNA-seq datasets
    """
    import GEOparse
    import requests
    import gzip

    gse = GEOparse.get_GEO(geo=accession, destdir=tempfile.gettempdir(), silent=True)

    # ── Collect sample metadata (always available) ────────────
    sample_metadata = {}
    gsm_title_map = {}  # GSM ID -> sample title (for matching suppl columns)
    for gsm_name, gsm in gse.gsms.items():
        chars = {}
        sample_title = gsm.metadata.get("title", [""])[0]
        chars["title"] = sample_title
        gsm_title_map[gsm_name] = sample_title
        for ch in gsm.metadata.get("characteristics_ch1", []):
            if ":" in ch:
                key, val = ch.split(":", 1)
                chars[key.strip()] = val.strip()
        sample_metadata[gsm_name] = chars

    meta_df = pd.DataFrame(sample_metadata).T
    meta_df.index.name = "sample"

    title = gse.metadata.get("title", [""])[0]
    summary = gse.metadata.get("summary", [""])[0]

    # ── Strategy 1: series matrix (ID_REF / VALUE per sample) ─
    expr_frames = []
    for gsm_name, gsm in gse.gsms.items():
        table = gsm.table
        if table.empty:
            continue
        if "ID_REF" in table.columns and "VALUE" in table.columns:
            series = table.set_index("ID_REF")["VALUE"]
            series.name = gsm_name
            expr_frames.append(series)

    if expr_frames:
        expr_df = pd.concat(expr_frames, axis=1)
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
        expr_df = expr_df.dropna(how="all")

        # Try to map probe IDs to gene symbols using GPL annotation
        for gpl_name, gpl in gse.gpls.items():
            gpl_table = gpl.table
            if gpl_table.empty:
                continue
            symbol_col = None
            for col in gpl_table.columns:
                if col.upper() in ["GENE_SYMBOL", "GENE SYMBOL", "SYMBOL", "GENE_NAME",
                                    "GENE", "ILMN_GENE", "GENE_ASSIGNMENT"]:
                    symbol_col = col
                    break
            if symbol_col and "ID" in gpl_table.columns:
                probe_to_gene = gpl_table.set_index("ID")[symbol_col].dropna()
                probe_to_gene = probe_to_gene[probe_to_gene.astype(str).str.strip() != ""]
                probe_to_gene = probe_to_gene[probe_to_gene.astype(str) != "---"]
                if len(probe_to_gene) > 0:
                    expr_df.index = expr_df.index.map(
                        lambda x: probe_to_gene.get(x, x)
                    )
                    expr_df = expr_df.groupby(expr_df.index).mean()
            break

        return expr_df, meta_df, title, summary

    # ── Strategy 2: supplementary count matrix files ──────────
    suppl_files = gse.metadata.get("supplementary_file", [])

    count_keywords = ["count", "readcount", "read_count", "expression", "matrix",
                      "fpkm", "tpm", "rpkm", "raw"]
    matrix_extensions = (".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".tsv", ".txt")

    candidate_urls = []
    for url in suppl_files:
        url_lower = url.lower()
        if not any(url_lower.endswith(ext) for ext in matrix_extensions):
            continue
        has_keyword = any(kw in url_lower for kw in count_keywords)
        candidate_urls.append((has_keyword, url))

    candidate_urls.sort(key=lambda x: x[0], reverse=True)

    for _, url in candidate_urls:
        try:
            dl_url = url
            if dl_url.startswith("ftp://"):
                dl_url = dl_url.replace(
                    "ftp://ftp.ncbi.nlm.nih.gov/geo/",
                    "https://ftp.ncbi.nlm.nih.gov/geo/",
                )

            resp = requests.get(dl_url, timeout=120)
            resp.raise_for_status()

            is_gz = url.lower().endswith(".gz")
            base = url.lower().removesuffix(".gz")
            sep = "\t" if base.endswith((".tsv", ".txt")) else ","

            if is_gz:
                content = gzip.decompress(resp.content).decode("utf-8")
            else:
                content = resp.text

            expr_df = pd.read_csv(io.StringIO(content), sep=sep, index_col=0)
            expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
            expr_df = expr_df.dropna(how="all")

            if expr_df.shape[0] >= 100 and expr_df.shape[1] >= 2:
                # ── Reconcile column names with metadata ──────
                # Expression columns might be sample names, not GSM IDs.
                # Try to match them to GSM IDs via sample titles.
                expr_cols = set(expr_df.columns)
                gsm_ids = set(meta_df.index)

                if not expr_cols & gsm_ids:
                    # No overlap — try matching by title
                    title_to_gsm = {v: k for k, v in gsm_title_map.items()}
                    # Also try normalised versions (strip, lower)
                    title_to_gsm_lower = {v.strip().lower(): k for k, v in gsm_title_map.items()}

                    col_to_gsm = {}
                    for col in expr_df.columns:
                        if col in title_to_gsm:
                            col_to_gsm[col] = title_to_gsm[col]
                        elif col.strip().lower() in title_to_gsm_lower:
                            col_to_gsm[col] = title_to_gsm_lower[col.strip().lower()]

                    if col_to_gsm:
                        # Rename expression columns to GSM IDs
                        expr_df = expr_df.rename(columns=col_to_gsm)
                    else:
                        # Can't map — rebuild metadata indexed by expression columns
                        # Use column names as sample IDs
                        meta_df = pd.DataFrame({"title": expr_df.columns}, index=expr_df.columns)
                        meta_df.index.name = "sample"

                return expr_df, meta_df, title, summary

        except Exception:
            continue

    raise ValueError(
        "Could not extract expression data from this GEO accession. "
        "The dataset may not contain a standard expression matrix. "
        "Try downloading the supplementary files manually and uploading them."
    )


def _detect_sep_and_read(file_obj):
    """Detect separator and compression, then read as DataFrame."""
    name = file_obj.name.lower()
    compression = "gzip" if name.endswith(".gz") else None
    base = name.removesuffix(".gz")
    sep = "\t" if base.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(file_obj, sep=sep, index_col=0, compression=compression)


def parse_uploaded_counts(counts_file):
    """Parse uploaded count/expression matrix."""
    df = _detect_sep_and_read(counts_file)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


def parse_uploaded_metadata(meta_file):
    """Parse uploaded sample metadata."""
    return _detect_sep_and_read(meta_file)


def run_pca(expr_df, n_components=2):
    """Run PCA on expression matrix (genes x samples)."""
    # Transpose so samples are rows, drop genes with any NaN
    data = expr_df.T.dropna(axis=1)
    # Remove zero-variance genes
    gene_var = data.var()
    data = data.loc[:, gene_var > 0]
    # Take top variable genes
    gene_var = data.var()
    top_genes = gene_var.nlargest(min(2000, len(gene_var))).index
    data_filtered = data[top_genes]

    # Replace any remaining inf/nan
    data_filtered = data_filtered.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_filtered)
    # Guard against NaN from scaling
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(
        components,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=expr_df.columns,
    )
    variance = pca.explained_variance_ratio_

    return pca_df, variance


def run_deg_analysis(expr_df, group1_samples, group2_samples, group1_name, group2_name,
                     use_deseq2=False):
    """Run differential expression analysis."""
    if use_deseq2:
        return _run_pydeseq2(expr_df, group1_samples, group2_samples,
                             group1_name, group2_name)
    else:
        return _run_basic_deg(expr_df, group1_samples, group2_samples,
                              group1_name, group2_name)


def _run_pydeseq2(expr_df, group1_samples, group2_samples, group1_name, group2_name):
    """Run DESeq2-style analysis using pydeseq2."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    all_samples = group1_samples + group2_samples
    counts = expr_df[all_samples].T

    counts = counts.round().astype(int)
    counts = counts.loc[:, (counts > 0).any(axis=0)]

    conditions = ([group1_name] * len(group1_samples) +
                  [group2_name] * len(group2_samples))
    metadata = pd.DataFrame({"condition": conditions}, index=all_samples)

    dds = DeseqDataSet(counts=counts, metadata=metadata, design="~condition")
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["condition", group2_name, group1_name])
    stat_res.summary()

    results = stat_res.results_df.copy()
    results = results.rename(columns={
        "log2FoldChange": "log2FC",
        "pvalue": "pvalue",
        "padj": "padj",
    })
    results = results[["log2FC", "pvalue", "padj"]].dropna()
    results["neg_log10_padj"] = -np.log10(results["padj"].clip(lower=1e-300))
    results = results.sort_values("pvalue")

    return results


def _run_basic_deg(expr_df, group1_samples, group2_samples, group1_name, group2_name):
    """Run basic DEG analysis using t-test for normalized data."""
    g1 = expr_df[group1_samples]
    g2 = expr_df[group2_samples]

    mean1 = g1.mean(axis=1)
    mean2 = g2.mean(axis=1)

    data_range = expr_df.max().max() - expr_df.min().min()
    is_log = data_range < 30

    if is_log:
        log2fc = mean2 - mean1
    else:
        log2fc = np.log2((mean2 + 1) / (mean1 + 1))

    pvalues = []
    for gene in expr_df.index:
        vals1 = g1.loc[gene].values.astype(float)
        vals2 = g2.loc[gene].values.astype(float)
        if np.std(vals1) == 0 and np.std(vals2) == 0:
            pvalues.append(1.0)
        else:
            _, p = stats.ttest_ind(vals1, vals2, equal_var=False, nan_policy="omit")
            pvalues.append(p if not np.isnan(p) else 1.0)

    pvalues = np.array(pvalues)

    # BH correction
    try:
        from scipy.stats import false_discovery_control
        padj = false_discovery_control(pvalues, method="bh")
    except Exception:
        n = len(pvalues)
        sorted_idx = np.argsort(pvalues)
        padj = np.ones(n)
        for rank, idx in enumerate(sorted_idx, 1):
            padj[idx] = pvalues[idx] * n / rank
        padj = np.minimum.accumulate(padj[np.argsort(sorted_idx)][::-1])[::-1]
        padj = np.clip(padj, 0, 1)

    results = pd.DataFrame({
        "log2FC": log2fc,
        "pvalue": pvalues,
        "padj": padj,
    }, index=expr_df.index)

    results["neg_log10_padj"] = -np.log10(results["padj"].clip(lower=1e-300))
    results = results.dropna()
    results = results.sort_values("pvalue")

    return results


def make_volcano_plot(deg_results, fc_thresh=1.0, pval_thresh=0.05, top_n_labels=10):
    """Create an interactive volcano plot."""
    df = deg_results.copy()
    df["significant"] = "Not significant"
    df.loc[
        (df["padj"] < pval_thresh) & (df["log2FC"] > fc_thresh), "significant"
    ] = "Up"
    df.loc[
        (df["padj"] < pval_thresh) & (df["log2FC"] < -fc_thresh), "significant"
    ] = "Down"

    color_map = {"Not significant": "#c0bbb0", "Up": "#cc2a2a", "Down": "#2a52cc"}

    plot_df = df.reset_index()
    # Normalise the gene name column to "gene" regardless of original index name
    gene_col = plot_df.columns[0]
    plot_df = plot_df.rename(columns={gene_col: "gene"})

    fig = px.scatter(
        plot_df,
        x="log2FC",
        y="neg_log10_padj",
        color="significant",
        color_discrete_map=color_map,
        hover_name="gene",
        hover_data={"log2FC": ":.2f", "padj": ":.2e", "significant": False, "neg_log10_padj": False},
        labels={"log2FC": "log₂ Fold Change", "neg_log10_padj": "-log₁₀ adjusted p-value"},
    )

    fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash", line_color="#888", line_width=1)
    fig.add_vline(x=fc_thresh, line_dash="dash", line_color="#888", line_width=1)
    fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="#888", line_width=1)

    sig_genes = df[df["significant"] != "Not significant"].nlargest(top_n_labels, "neg_log10_padj")
    for gene_name, row in sig_genes.iterrows():
        fig.add_annotation(
            x=row["log2FC"], y=row["neg_log10_padj"],
            text=str(gene_name), showarrow=True, arrowhead=0,
            ax=20, ay=-20, font=dict(size=10, family="IBM Plex Mono"),
        )

    fig.update_layout(
        template="plotly_white",
        font_family="IBM Plex Mono",
        plot_bgcolor="#faf8f4",
        paper_bgcolor="#f0ece3",
        legend_title_text="",
        width=800, height=600,
    )

    return fig


def make_pca_plot(pca_df, variance, color_col=None, meta_df=None):
    """Create an interactive PCA plot."""
    plot_df = pca_df.copy()
    plot_df.index.name = "sample"
    plot_df = plot_df.reset_index()

    if color_col and meta_df is not None and color_col in meta_df.columns:
        merge_meta = meta_df[[color_col]].copy()
        merge_meta.index.name = "sample"
        merge_meta = merge_meta.reset_index()
        plot_df = plot_df.merge(merge_meta, on="sample", how="left")
        color = color_col
    else:
        color = None

    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color=color,
        hover_name="sample",
        labels={
            "PC1": f"PC1 ({variance[0]*100:.1f}%)",
            "PC2": f"PC2 ({variance[1]*100:.1f}%)",
        },
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="#1a1a1a")))
    fig.update_layout(
        template="plotly_white",
        font_family="IBM Plex Mono",
        plot_bgcolor="#faf8f4",
        paper_bgcolor="#f0ece3",
        width=800, height=600,
    )

    return fig


def make_gene_plot(expr_df, gene, meta_df=None, group_col=None):
    """Create a box/strip plot for a single gene across samples."""
    if gene not in expr_df.index:
        return None

    values = expr_df.loc[gene]
    plot_df = pd.DataFrame({"sample": values.index, "expression": values.values})

    if group_col and meta_df is not None and group_col in meta_df.columns:
        merge_meta = meta_df[[group_col]].copy()
        merge_meta.index.name = "sample"
        merge_meta = merge_meta.reset_index()
        plot_df = plot_df.merge(merge_meta, on="sample", how="left")
        fig = px.box(
            plot_df, x=group_col, y="expression",
            points="all", hover_name="sample",
            color=group_col,
            labels={"expression": "Expression"},
        )
    else:
        fig = px.strip(
            plot_df, x="sample", y="expression",
            hover_name="sample",
            labels={"expression": "Expression"},
        )
        fig.update_xaxes(tickangle=45)

    fig.update_layout(
        title=f"{gene}",
        template="plotly_white",
        font_family="IBM Plex Mono",
        plot_bgcolor="#faf8f4",
        paper_bgcolor="#f0ece3",
        showlegend=False,
        width=800, height=500,
    )

    return fig


# ============================================================
# APP LAYOUT
# ============================================================

st.markdown("# RNAseq Explorer")
st.markdown("**Visualize transcript profiles and differential expression from public or uploaded RNA-seq data.**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### Getting Started")
    st.markdown("""
1. **Load data** from GEO or upload your own
2. **Explore** with PCA and gene search
3. **Run DEG analysis** and view volcano plots
    """)
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.7em; letter-spacing:0.15em; text-transform:uppercase;">'
        'Built by <a href="https://github.com/mirnakg" style="color:#2a52cc;">Mirna Kheir Gouda</a>'
        '</p>',
        unsafe_allow_html=True,
    )

# ── DATA LOADING ──────────────────────────────────────────────
st.markdown('<div class="step-header">Step 1 — Load Data</div>', unsafe_allow_html=True)

data_source = st.radio(
    "How would you like to provide your data?",
    ["GEO Accession", "Upload Files"],
    horizontal=True,
)

expr_df = None
meta_df = None
dataset_title = ""

if data_source == "GEO Accession":
    geo_col1, geo_col2 = st.columns([3, 1])
    with geo_col1:
        accession = st.text_input("Enter GEO Series accession", placeholder="e.g. GSE53757")
    with geo_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch = st.button("Fetch", type="primary")

    if accession and fetch:
        try:
            expr_df, meta_df, dataset_title, summary = load_geo_data(accession.strip())
            st.session_state["expr_df"] = expr_df
            st.session_state["meta_df"] = meta_df
            st.session_state["dataset_title"] = dataset_title
            st.session_state["dataset_summary"] = summary
        except Exception as e:
            st.error(f"Error loading {accession}: {str(e)}")

    if "expr_df" in st.session_state and data_source == "GEO Accession":
        expr_df = st.session_state["expr_df"]
        meta_df = st.session_state["meta_df"]
        dataset_title = st.session_state.get("dataset_title", "")

else:
    st.markdown("Upload a **count/expression matrix** (genes as rows, samples as columns) "
                "and optionally a **sample metadata** file.")
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        counts_file = st.file_uploader("Expression matrix (.csv, .tsv, .txt, .gz)",
                                       type=["csv", "tsv", "txt", "gz"])
    with up_col2:
        meta_file = st.file_uploader("Sample metadata (.csv, .tsv, .txt, .gz) — optional",
                                     type=["csv", "tsv", "txt", "gz"])

    if counts_file:
        try:
            expr_df = parse_uploaded_counts(counts_file)
            st.session_state["expr_df"] = expr_df
            if meta_file:
                meta_df = parse_uploaded_metadata(meta_file)
                st.session_state["meta_df"] = meta_df
            st.session_state["dataset_title"] = counts_file.name
        except Exception as e:
            st.error(f"Error parsing file: {str(e)}")

    if "expr_df" in st.session_state and data_source == "Upload Files":
        expr_df = st.session_state["expr_df"]
        meta_df = st.session_state.get("meta_df")
        dataset_title = st.session_state.get("dataset_title", "")


# ── ONCE DATA IS LOADED ──────────────────────────────────────
if expr_df is not None:
    st.markdown("---")
    if dataset_title:
        st.markdown(f"**{dataset_title}**")
    st.caption(f"{expr_df.shape[0]:,} genes × {expr_df.shape[1]:,} samples")

    all_samples = list(expr_df.columns)

    # Determine available grouping columns from metadata
    group_options = []
    if meta_df is not None and not meta_df.empty:
        # Only consider metadata columns whose index overlaps with expression columns
        common = set(meta_df.index) & set(all_samples)
        if common:
            for col in meta_df.columns:
                vals = meta_df.loc[meta_df.index.isin(all_samples), col].dropna()
                nunique = vals.nunique()
                if 2 <= nunique <= 50:
                    group_options.append(col)

    # ── TABS ──
    tab_preview, tab_pca, tab_deg, tab_gene = st.tabs(
        ["Data Preview", "PCA", "DEG Analysis", "Gene Search"]
    )

    # ── DATA PREVIEW ──────────────────────────────────────────
    with tab_preview:
        st.markdown('<div class="step-header">Expression Matrix</div>', unsafe_allow_html=True)
        st.dataframe(expr_df.head(50), use_container_width=True, height=400)

        if meta_df is not None and not meta_df.empty:
            st.markdown('<div class="step-header">Sample Metadata</div>', unsafe_allow_html=True)
            st.dataframe(meta_df, use_container_width=True, height=300)

        csv_buf = expr_df.to_csv()
        st.download_button("Download expression matrix (.csv)", csv_buf,
                           file_name="expression_matrix.csv", mime="text/csv")

    # ── PCA ───────────────────────────────────────────────────
    with tab_pca:
        st.markdown('<div class="step-header">Principal Component Analysis</div>',
                    unsafe_allow_html=True)

        pca_color = None
        if group_options:
            pca_color = st.selectbox("Color samples by", ["None"] + group_options,
                                     key="pca_color")
            if pca_color == "None":
                pca_color = None

        try:
            pca_df, variance = run_pca(expr_df)
            fig = make_pca_plot(pca_df, variance, color_col=pca_color, meta_df=meta_df)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("PCA data"):
                st.dataframe(pca_df, use_container_width=True)
        except Exception as e:
            st.error(f"PCA failed: {str(e)}")

    # ── DEG ANALYSIS ──────────────────────────────────────────
    with tab_deg:
        st.markdown('<div class="step-header">Differential Expression Analysis</div>',
                    unsafe_allow_html=True)

        deg_mode = st.radio(
            "How would you like to define groups?",
            ["From metadata column", "Manual sample selection"],
            horizontal=True, key="deg_mode",
        )

        group1_samples = []
        group2_samples = []
        group1_name = "Control"
        group2_name = "Treatment"

        if deg_mode == "From metadata column":
            if not group_options:
                st.warning("No usable grouping columns found in metadata. "
                           "Use **Manual sample selection** instead, or provide "
                           "metadata with a condition column.")
            else:
                deg_group_col = st.selectbox("Group samples by", group_options, key="deg_group")
                groups = meta_df.loc[
                    meta_df.index.isin(all_samples), deg_group_col
                ].dropna().unique().tolist()

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    group1_name = st.selectbox("Control / Reference group", groups, key="g1")
                with dcol2:
                    remaining = [g for g in groups if g != group1_name]
                    group2_name = st.selectbox("Treatment / Comparison group",
                                               remaining if remaining else groups, key="g2")

                group1_samples = meta_df[
                    (meta_df[deg_group_col] == group1_name) &
                    (meta_df.index.isin(all_samples))
                ].index.tolist()
                group2_samples = meta_df[
                    (meta_df[deg_group_col] == group2_name) &
                    (meta_df.index.isin(all_samples))
                ].index.tolist()

                st.caption(f"Group 1: {len(group1_samples)} samples — "
                           f"Group 2: {len(group2_samples)} samples")

        else:  # Manual sample selection
            st.markdown("Select samples for each group from the expression matrix columns.")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                group1_name = st.text_input("Group 1 name", value="Control", key="manual_g1_name")
                group1_samples = st.multiselect(
                    f"Samples in **{group1_name}**", all_samples, key="manual_g1",
                )
            with mcol2:
                available_for_g2 = [s for s in all_samples if s not in group1_samples]
                group2_name = st.text_input("Group 2 name", value="Treatment", key="manual_g2_name")
                group2_samples = st.multiselect(
                    f"Samples in **{group2_name}**", available_for_g2, key="manual_g2",
                )

        # Check if data looks like raw counts
        sample_vals = expr_df.values
        is_counts = (sample_vals % 1 == 0).all() and (sample_vals >= 0).all()

        if is_counts:
            method = st.radio(
                "Analysis method",
                ["DESeq2 (recommended for count data)", "Basic t-test"],
                horizontal=True, key="deg_method",
            )
            use_deseq2 = "DESeq2" in method
        else:
            st.caption("Data appears to be normalized/log-transformed — using t-test with BH correction.")
            use_deseq2 = False

        # Threshold settings
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            fc_thresh = st.number_input("log₂FC threshold", value=1.0, min_value=0.0,
                                        step=0.25, key="fc_thresh")
        with tcol2:
            pval_thresh = st.number_input("Adj. p-value threshold", value=0.05,
                                          min_value=0.001, max_value=1.0,
                                          step=0.01, format="%.3f", key="pval_thresh")
        with tcol3:
            top_labels = st.number_input("Top genes to label", value=10, min_value=0,
                                         max_value=50, key="top_labels")

        can_run_deg = len(group1_samples) >= 2 and len(group2_samples) >= 2

        if st.button("Run DEG Analysis", type="primary", key="run_deg", disabled=not can_run_deg):
            with st.spinner("Running analysis..."):
                try:
                    deg_results = run_deg_analysis(
                        expr_df, group1_samples, group2_samples,
                        group1_name, group2_name, use_deseq2=use_deseq2,
                    )
                    st.session_state["deg_results"] = deg_results
                except Exception as e:
                    st.error(f"DEG analysis failed: {str(e)}")

        if not can_run_deg:
            st.caption("Select at least 2 samples per group to run analysis.")

        # Show results if available
        if "deg_results" in st.session_state:
            deg_results = st.session_state["deg_results"]

            n_up = ((deg_results["padj"] < pval_thresh) & (deg_results["log2FC"] > fc_thresh)).sum()
            n_down = ((deg_results["padj"] < pval_thresh) & (deg_results["log2FC"] < -fc_thresh)).sum()

            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.metric("Total genes tested", f"{len(deg_results):,}")
            rcol2.metric("Upregulated", f"{n_up:,}")
            rcol3.metric("Downregulated", f"{n_down:,}")

            st.markdown('<div class="step-header">Volcano Plot</div>', unsafe_allow_html=True)
            fig = make_volcano_plot(deg_results, fc_thresh=fc_thresh,
                                    pval_thresh=pval_thresh, top_n_labels=top_labels)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("DEG results table"):
                st.dataframe(
                    deg_results[["log2FC", "pvalue", "padj"]].head(200),
                    use_container_width=True,
                )

            csv_buf = deg_results.to_csv()
            st.download_button("Download full DEG results (.csv)", csv_buf,
                               file_name="deg_results.csv", mime="text/csv")

    # ── GENE SEARCH ───────────────────────────────────────────
    with tab_gene:
        st.markdown('<div class="step-header">Gene Search</div>', unsafe_allow_html=True)

        gene_query = st.text_input("Search for a gene", placeholder="e.g. TP53, BRCA1, GAPDH")

        gene_group_col = None
        if group_options:
            gene_group_col = st.selectbox("Group samples by", ["None"] + group_options,
                                          key="gene_group")
            if gene_group_col == "None":
                gene_group_col = None

        if gene_query:
            query = gene_query.strip().upper()
            exact = [g for g in expr_df.index if str(g).upper() == query]
            partial = [g for g in expr_df.index if query in str(g).upper() and str(g).upper() != query]
            matches = exact + partial[:20]

            if not matches:
                st.warning(f"No genes matching '{gene_query}' found.")
            else:
                if len(matches) > 1:
                    selected_gene = st.selectbox("Select gene", matches, key="gene_select")
                else:
                    selected_gene = matches[0]

                fig = make_gene_plot(expr_df, selected_gene, meta_df=meta_df,
                                     group_col=gene_group_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Expression values"):
                        gene_vals = expr_df.loc[selected_gene].to_frame("expression")
                        if meta_df is not None and gene_group_col:
                            gene_vals = gene_vals.merge(
                                meta_df[[gene_group_col]], left_index=True, right_index=True, how="left"
                            )
                        st.dataframe(gene_vals, use_container_width=True)

                    if "deg_results" in st.session_state and selected_gene in st.session_state["deg_results"].index:
                        gene_deg = st.session_state["deg_results"].loc[selected_gene]
                        st.markdown(f"**DEG stats:** log₂FC = {gene_deg['log2FC']:.3f}, "
                                    f"p-adj = {gene_deg['padj']:.2e}")
