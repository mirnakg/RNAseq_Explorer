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

def _adjust_pvalues(pvalues, method="Benjamini-Hochberg"):
    """Adjust p-values for multiple testing.

    Methods:
      - Benjamini-Hochberg (BH): controls FDR
      - Benjamini-Yekutieli (BY): controls FDR under dependence
      - Bonferroni: controls FWER (most conservative)
      - Holm: step-down Bonferroni (controls FWER, less conservative)
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return pvalues

    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    adjusted = np.empty(n)

    if method == "Bonferroni":
        result = np.clip(pvalues * n, 0, 1)
        return result

    elif method == "Holm":
        for i in range(n):
            adjusted[i] = sorted_pvals[i] * (n - i)
        # Enforce monotonicity (step-up)
        for i in range(1, n):
            adjusted[i] = max(adjusted[i], adjusted[i - 1])
        adjusted = np.clip(adjusted, 0, 1)
        result = np.empty(n)
        result[sorted_idx] = adjusted
        return result

    elif method == "Benjamini-Yekutieli":
        c_n = sum(1.0 / k for k in range(1, n + 1))
        adjusted[-1] = min(1.0, sorted_pvals[-1] * c_n)
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i + 1], sorted_pvals[i] * n * c_n / (i + 1))
        adjusted = np.clip(adjusted, 0, 1)
        result = np.empty(n)
        result[sorted_idx] = adjusted
        return result

    else:  # Benjamini-Hochberg (default)
        adjusted[-1] = sorted_pvals[-1]
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i + 1], sorted_pvals[i] * n / (i + 1))
        adjusted = np.clip(adjusted, 0, 1)
        result = np.empty(n)
        result[sorted_idx] = adjusted
        return result


def _looks_like_counts(expr_df):
    """Heuristic check whether expression data is raw counts (integers >= 0).
    Only samples a subset for speed on large matrices."""
    sample = expr_df.iloc[:min(500, expr_df.shape[0]), :min(20, expr_df.shape[1])]
    vals = sample.values
    # Must be numeric, non-negative, and integer-valued
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return False
    return bool((finite >= 0).all() and (finite % 1 == 0).all())


@st.cache_data(show_spinner="Fetching GEO dataset...")
def load_geo_data(accession):
    """Download and parse a GEO dataset."""
    import GEOparse
    import requests
    import gzip

    gse = GEOparse.get_GEO(geo=accession, destdir=tempfile.gettempdir(), silent=True)

    # ── Collect sample metadata ───────────────────────────────
    sample_metadata = {}
    gsm_title_map = {}
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

    # ── Strategy 1: series matrix ─────────────────────────────
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

        # Map probes to gene symbols
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
                    expr_df.index = expr_df.index.map(lambda x: probe_to_gene.get(x, x))
                    expr_df = expr_df.groupby(expr_df.index).mean()
            break

        return expr_df, meta_df, title, summary

    # ── Strategy 2: supplementary files ───────────────────────
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
                # Reconcile columns with metadata
                expr_cols = set(expr_df.columns)
                gsm_ids = set(meta_df.index)

                if not expr_cols & gsm_ids:
                    title_to_gsm = {v: k for k, v in gsm_title_map.items()}
                    title_to_gsm_lower = {v.strip().lower(): k for k, v in gsm_title_map.items()}

                    col_to_gsm = {}
                    for col in expr_df.columns:
                        if col in title_to_gsm:
                            col_to_gsm[col] = title_to_gsm[col]
                        elif str(col).strip().lower() in title_to_gsm_lower:
                            col_to_gsm[col] = title_to_gsm_lower[str(col).strip().lower()]

                    if col_to_gsm:
                        expr_df = expr_df.rename(columns=col_to_gsm)
                    else:
                        # Can't map — rebuild metadata from column names
                        meta_df = pd.DataFrame(
                            {"title": [str(c) for c in expr_df.columns]},
                            index=expr_df.columns,
                        )
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
    data = expr_df.T.copy()

    # Drop columns (genes) that are all NaN or constant
    data = data.dropna(axis=1, how="all")
    data = data.fillna(0)
    data = data.loc[:, data.nunique() > 1]

    if data.shape[1] < 2:
        raise ValueError("Not enough variable genes to run PCA.")

    n_components = min(n_components, data.shape[0], data.shape[1])

    # Top variable genes
    gene_var = data.var()
    top_genes = gene_var.nlargest(min(2000, len(gene_var))).index
    data_filtered = data[top_genes]

    data_filtered = data_filtered.replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_filtered)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(
        components,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=data.index,  # use the transposed index (sample names)
    )
    variance = pca.explained_variance_ratio_

    return pca_df, variance


def _run_pydeseq2(expr_df, group1_samples, group2_samples, group1_name, group2_name):
    """Run DESeq2-style analysis using pydeseq2."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    all_samples = group1_samples + group2_samples
    counts = expr_df[all_samples].T.copy()

    # Ensure non-negative integers
    counts = counts.fillna(0)
    counts = counts.clip(lower=0)
    counts = counts.round().astype(int)

    # Pre-filter genes to reduce memory: keep genes with at least 10 total
    # counts across all samples AND detected in at least 2 samples
    gene_totals = counts.sum(axis=0)
    gene_detected = (counts > 0).sum(axis=0)
    keep = (gene_totals >= 10) & (gene_detected >= 2)
    counts = counts.loc[:, keep]

    if counts.shape[1] < 2:
        raise ValueError("Not enough expressed genes for DESeq2 analysis after filtering.")

    # Sanitise group names for the design formula (no spaces/special chars)
    g1_safe = "group1"
    g2_safe = "group2"
    conditions = ([g1_safe] * len(group1_samples) +
                  [g2_safe] * len(group2_samples))
    metadata = pd.DataFrame({"condition": conditions}, index=all_samples)

    dds = DeseqDataSet(
        counts=counts, metadata=metadata, design="~condition",
        n_cpus=1,  # limit CPU/memory usage
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["condition", g2_safe, g1_safe])
    stat_res.summary()

    results = stat_res.results_df.copy()
    results = results.rename(columns={
        "log2FoldChange": "log2FC",
        "pvalue": "pvalue",
        "padj": "padj",
    })
    cols_present = [c for c in ["log2FC", "pvalue", "padj"] if c in results.columns]
    results = results[cols_present].dropna()
    results["neg_log10_padj"] = -np.log10(results["padj"].clip(lower=1e-300))
    results = results.sort_values("pvalue")

    return results


def _run_basic_deg(expr_df, group1_samples, group2_samples, group1_name, group2_name,
                   padj_method="Benjamini-Hochberg"):
    """Run basic DEG analysis using Welch's t-test."""
    g1 = expr_df[group1_samples].copy()
    g2 = expr_df[group2_samples].copy()

    mean1 = g1.mean(axis=1)
    mean2 = g2.mean(axis=1)

    # Heuristic: if the max value is small, data is likely already log-transformed
    max_val = expr_df.max().max()
    is_log = (max_val < 30) if np.isfinite(max_val) else False

    if is_log:
        log2fc = mean2 - mean1
    else:
        log2fc = np.log2((mean2 + 1) / (mean1 + 1))

    # Vectorised t-test
    n1, n2 = g1.shape[1], g2.shape[1]
    var1 = g1.var(axis=1, ddof=1).fillna(0)
    var2 = g2.var(axis=1, ddof=1).fillna(0)

    se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean2 - mean1) / se.replace(0, np.nan)

    # Welch-Satterthwaite degrees of freedom
    num = (var1 / n1 + var2 / n2) ** 2
    denom = ((var1 / n1) ** 2 / max(n1 - 1, 1) + (var2 / n2) ** 2 / max(n2 - 1, 1))
    df_welch = num / denom.replace(0, np.nan)

    pvalues = pd.Series(1.0, index=expr_df.index)
    valid = t_stat.notna() & df_welch.notna() & (df_welch > 0)
    pvalues[valid] = 2 * stats.t.sf(np.abs(t_stat[valid]), df_welch[valid])
    pvalues = pvalues.fillna(1.0).values

    padj = _adjust_pvalues(pvalues, method=padj_method)

    results = pd.DataFrame({
        "log2FC": log2fc,
        "pvalue": pvalues,
        "padj": padj,
    }, index=expr_df.index)

    results["neg_log10_padj"] = -np.log10(results["padj"].clip(lower=1e-300))
    results = results.replace([np.inf, -np.inf], np.nan).dropna()
    results = results.sort_values("pvalue")

    return results


def run_deg_analysis(expr_df, group1_samples, group2_samples, group1_name, group2_name,
                     use_deseq2=False, padj_method="Benjamini-Hochberg"):
    """Run differential expression analysis."""
    if use_deseq2:
        return _run_pydeseq2(expr_df, group1_samples, group2_samples,
                             group1_name, group2_name)
    else:
        return _run_basic_deg(expr_df, group1_samples, group2_samples,
                              group1_name, group2_name, padj_method=padj_method)


def make_volcano_plot(deg_results, fc_thresh=1.0, pval_thresh=0.05, top_n_labels=10):
    """Create an interactive volcano plot."""
    df = deg_results.copy()

    # Cap -log10(padj) to avoid infinite values breaking the plot
    max_neg_log = df["neg_log10_padj"].replace([np.inf, -np.inf], np.nan).max()
    cap = max_neg_log * 1.05 if pd.notna(max_neg_log) and max_neg_log > 0 else 50
    df["neg_log10_padj"] = df["neg_log10_padj"].clip(upper=cap).fillna(0)

    df["significant"] = "Not significant"
    df.loc[
        (df["padj"] < pval_thresh) & (df["log2FC"] > fc_thresh), "significant"
    ] = "Up"
    df.loc[
        (df["padj"] < pval_thresh) & (df["log2FC"] < -fc_thresh), "significant"
    ] = "Down"

    color_map = {"Not significant": "#c0bbb0", "Up": "#cc2a2a", "Down": "#2a52cc"}

    plot_df = df.reset_index()
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

    # ── STEP 2: DEFINE GROUPS ─────────────────────────────────
    st.markdown('<div class="step-header">Step 2 — Define Groups</div>', unsafe_allow_html=True)
    st.markdown("Assign samples to named groups. These groups are used across all analyses (PCA, DEG, gene search).")

    group_col_name = st.text_input("Group column name", value="condition", key="group_col_name")

    # Dynamic group builder
    if "n_groups" not in st.session_state:
        st.session_state["n_groups"] = 2

    gcol_add, gcol_remove = st.columns([1, 1])
    with gcol_add:
        if st.button("+ Add group", key="add_group"):
            st.session_state["n_groups"] += 1
    with gcol_remove:
        if st.button("- Remove last group", key="remove_group") and st.session_state["n_groups"] > 2:
            st.session_state["n_groups"] -= 1

    n_groups = st.session_state["n_groups"]
    group_cols = st.columns(min(n_groups, 4))

    assigned_samples = []
    group_assignments = {}  # sample -> group name
    group_names = []
    group_sample_lists = {}  # group name -> [samples]

    for i in range(n_groups):
        col = group_cols[i % len(group_cols)]
        with col:
            gname = st.text_input(f"Group {i+1} name", value=f"Group_{i+1}", key=f"gname_{i}")
            group_names.append(gname)
            available = [s for s in all_samples if s not in assigned_samples]
            selected = st.multiselect(f"Samples", available, key=f"gsamples_{i}")
            assigned_samples.extend(selected)
            group_sample_lists[gname] = selected
            for s in selected:
                group_assignments[s] = gname

    # Build a group metadata Series for use everywhere
    group_series = pd.Series(group_assignments, name=group_col_name)
    has_groups = len(group_assignments) >= 2

    if has_groups:
        n_assigned = len(group_assignments)
        n_unassigned = len(all_samples) - n_assigned
        msg = f"{n_assigned} samples assigned"
        if n_unassigned > 0:
            msg += f", {n_unassigned} unassigned"
        st.caption(msg)

    st.markdown("---")

    # Filter expression data to assigned samples only for downstream analysis
    if has_groups:
        assigned_sample_list = list(group_assignments.keys())
        expr_assigned = expr_df[assigned_sample_list]
    else:
        expr_assigned = expr_df

    # ── DATA PREVIEW (collapsible) ────────────────────────────
    with st.expander("Data Preview"):
        st.markdown('<div class="step-header">Expression Matrix</div>', unsafe_allow_html=True)
        st.dataframe(expr_df.head(50), use_container_width=True, height=400)

        if has_groups:
            st.markdown('<div class="step-header">Group Assignments</div>', unsafe_allow_html=True)
            group_df = pd.DataFrame({
                "sample": assigned_sample_list,
                group_col_name: [group_assignments[s] for s in assigned_sample_list],
            }).set_index("sample")
            st.dataframe(group_df, use_container_width=True, height=300)

        if meta_df is not None and not meta_df.empty:
            st.markdown('<div class="step-header">Sample Metadata (from source)</div>',
                        unsafe_allow_html=True)
            st.dataframe(meta_df, use_container_width=True, height=300)

        csv_buf = expr_df.to_csv()
        st.download_button("Download expression matrix (.csv)", csv_buf,
                           file_name="expression_matrix.csv", mime="text/csv")

    # ── TABS ──
    tab_pca, tab_deg, tab_gene = st.tabs(
        ["PCA", "DEG Analysis", "Gene Search"]
    )

    # ── PCA ───────────────────────────────────────────────────
    with tab_pca:
        st.markdown('<div class="step-header">Principal Component Analysis</div>',
                    unsafe_allow_html=True)

        try:
            pca_df, variance = run_pca(expr_assigned)

            # Build plot dataframe with group info
            plot_df = pca_df.copy()
            plot_df.index.name = "sample"
            plot_df = plot_df.reset_index()

            if has_groups:
                plot_df[group_col_name] = plot_df["sample"].map(group_assignments)
                color = group_col_name
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
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("PCA data"):
                st.dataframe(pca_df, use_container_width=True)
        except Exception as e:
            st.error(f"PCA failed: {str(e)}")

    # ── DEG ANALYSIS ──────────────────────────────────────────
    with tab_deg:
        st.markdown('<div class="step-header">Differential Expression Analysis</div>',
                    unsafe_allow_html=True)

        # Only groups with >= 2 samples can be used for DEG
        valid_groups = [g for g, samples in group_sample_lists.items() if len(samples) >= 2]

        if len(valid_groups) < 2:
            st.warning("Define at least 2 groups with 2+ samples each in Step 2 to run DEG analysis.")
        else:
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                group1_name = st.selectbox("Control / Reference group", valid_groups, key="deg_g1")
            with dcol2:
                remaining = [g for g in valid_groups if g != group1_name]
                group2_name = st.selectbox("Treatment / Comparison group", remaining, key="deg_g2")

            group1_samples = group_sample_lists[group1_name]
            group2_samples = group_sample_lists[group2_name]

            st.caption(f"{group1_name}: {len(group1_samples)} samples — "
                       f"{group2_name}: {len(group2_samples)} samples")

            is_counts = _looks_like_counts(expr_df)
            use_deseq2 = is_counts

            if is_counts:
                st.caption("Data detected as raw counts — using DESeq2.")
            else:
                st.caption("Data appears normalized/log-transformed — using Welch's t-test.")

            padj_methods = ["Benjamini-Hochberg", "Benjamini-Yekutieli", "Bonferroni", "Holm"]

            tcol1, tcol2, tcol3, tcol4 = st.columns(4)
            with tcol1:
                fc_thresh = st.number_input("log₂FC threshold", value=1.0, min_value=0.0,
                                            step=0.25, key="fc_thresh")
            with tcol2:
                pval_thresh = st.number_input("Adj. p-value threshold", value=0.05,
                                              min_value=0.001, max_value=1.0,
                                              step=0.01, format="%.3f", key="pval_thresh")
            with tcol3:
                padj_method = st.selectbox("P-value adjustment", padj_methods, key="padj_method")
            with tcol4:
                top_labels = st.number_input("Top genes to label", value=10, min_value=0,
                                             max_value=50, key="top_labels")

            if st.button("Run DEG Analysis", type="primary", key="run_deg"):
                with st.spinner("Running analysis..."):
                    try:
                        deg_results = run_deg_analysis(
                            expr_df, group1_samples, group2_samples,
                            group1_name, group2_name, use_deseq2=use_deseq2,
                            padj_method=padj_method,
                        )
                        st.session_state["deg_results"] = deg_results
                    except Exception as e:
                        st.error(f"DEG analysis failed: {str(e)}")

        if "deg_results" in st.session_state:
            deg_results = st.session_state["deg_results"]

            fc_t = fc_thresh if "fc_thresh" in dir() else 1.0
            pv_t = pval_thresh if "pval_thresh" in dir() else 0.05
            tl = top_labels if "top_labels" in dir() else 10

            n_up = ((deg_results["padj"] < pv_t) & (deg_results["log2FC"] > fc_t)).sum()
            n_down = ((deg_results["padj"] < pv_t) & (deg_results["log2FC"] < -fc_t)).sum()

            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.metric("Total genes tested", f"{len(deg_results):,}")
            rcol2.metric("Upregulated", f"{n_up:,}")
            rcol3.metric("Downregulated", f"{n_down:,}")

            st.markdown('<div class="step-header">Volcano Plot</div>', unsafe_allow_html=True)
            fig = make_volcano_plot(deg_results, fc_thresh=fc_t,
                                    pval_thresh=pv_t, top_n_labels=tl)
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

                # Build plot with group coloring (assigned samples only)
                values = expr_assigned.loc[selected_gene] if has_groups else expr_df.loc[selected_gene]
                plot_df = pd.DataFrame({"sample": values.index, "expression": values.values})

                if has_groups:
                    plot_df[group_col_name] = plot_df["sample"].map(group_assignments)
                    fig = px.box(
                        plot_df, x=group_col_name, y="expression",
                        points="all", hover_name="sample",
                        color=group_col_name,
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
                    title=f"{selected_gene}",
                    template="plotly_white",
                    font_family="IBM Plex Mono",
                    plot_bgcolor="#faf8f4",
                    paper_bgcolor="#f0ece3",
                    showlegend=False,
                    width=800, height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Expression values"):
                    gene_vals = expr_assigned.loc[selected_gene].to_frame("expression") if has_groups else expr_df.loc[selected_gene].to_frame("expression")
                    if has_groups:
                        gene_vals[group_col_name] = gene_vals.index.map(group_assignments)
                    st.dataframe(gene_vals, use_container_width=True)

                if "deg_results" in st.session_state and selected_gene in st.session_state["deg_results"].index:
                    gene_deg = st.session_state["deg_results"].loc[selected_gene]
                    st.markdown(f"**DEG stats:** log₂FC = {gene_deg['log2FC']:.3f}, "
                                f"p-adj = {gene_deg['padj']:.2e}")
