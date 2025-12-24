import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide", page_title="RFQ Optimal Margin Dashboard")

# =========================
# Helpers
# =========================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def require_cols(df: pd.DataFrame, needed: list, name: str):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"[{name}] 시트에서 필요한 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")
        st.stop()

def get_assumption(ass: pd.DataFrame, item: str, default=None):
    row = ass.loc[ass["item"] == item, "value"]
    if len(row) == 0:
        if default is None:
            st.error(f"Assumption 시트에서 '{item}' 값을 찾을 수 없습니다.")
            st.stop()
        return float(default)
    return float(row.values[0])

def fmt0(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):,.0f}"
    except:
        return str(x)

def fmt2(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):,.2f}"
    except:
        return str(x)

def fmt_int(x):
    try:
        if pd.isna(x):
            return ""
        return f"{int(round(float(x))):,}"
    except:
        return str(x)

def y_comma(ax):
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{int(v):,}"))

def make_download_excel(project_name: str,
                        proj: pd.DataFrame,
                        df_lines: pd.DataFrame,
                        proj_res: pd.DataFrame,
                        res: pd.DataFrame,
                        sim: pd.DataFrame) -> bytes:
    """Export tables to Excel with comma-formatted strings (회계 스타일)."""
    output = BytesIO()

    # Display versions with commas
    lines_disp = df_lines.copy()
    money_cols = [
        "material_cost_per_unit","sub_parts_cost_per_unit","processing_cost","sga_cost_per_unit",
        "tool_amort_per_unit","dev_amort_per_unit","unit_cost",
        "line_total_cost","material_total","processing_total","amort_total"
    ]
    for c in money_cols:
        if c in lines_disp.columns:
            lines_disp[c] = lines_disp[c].map(fmt0)
    if "lifetime_qty" in lines_disp.columns:
        lines_disp["lifetime_qty"] = lines_disp["lifetime_qty"].map(fmt_int)

    proj_res_disp = proj_res.copy()
    proj_res_disp["expected_profit"] = proj_res_disp["expected_profit"].map(fmt0)
    proj_res_disp["margin"] = proj_res_disp["margin"].map(lambda x: f"{float(x)*100:.2f}%")

    # Line sweep export (keep a compact table)
    res_comp = res.groupby(["line_id","margin"], as_index=False)["expected_profit"].sum()
    res_comp["expected_profit"] = res_comp["expected_profit"].map(fmt0)
    res_comp["margin"] = res_comp["margin"].map(lambda x: f"{float(x)*100:.2f}%")

    sim_disp = sim.copy()
    if len(sim_disp) > 0:
        # limit size
        sim_disp = sim_disp.head(500)
        for c in ["lifetime_qty"]:
            if c in sim_disp.columns:
                sim_disp[c] = sim_disp[c].map(fmt_int)
        for c in ["margin_rate"]:
            if c in sim_disp.columns:
                sim_disp[c] = sim_disp[c].map(lambda x: f"{float(x)*100:.2f}%")

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        proj.to_excel(writer, index=False, sheet_name="Project_Fact")
        lines_disp.to_excel(writer, index=False, sheet_name="Lines_Calc")
        proj_res_disp.to_excel(writer, index=False, sheet_name="Margin_Sweep_Project")
        res_comp.to_excel(writer, index=False, sheet_name="Margin_Sweep_Lines")
        if len(sim_disp) > 0:
            sim_disp.to_excel(writer, index=False, sheet_name="Similar_History_sample")

    return output.getvalue()

# =========================
# Sidebar: cache control
# =========================
st.sidebar.header("데이터 로드")
if st.sidebar.button("🔄 엑셀 반영(캐시 리셋)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

debug = st.sidebar.checkbox("🧪 DEBUG 표시", value=False)

# =========================
# Load Data
# =========================
@st.cache_data
def load_history():
    df = pd.read_excel("data/RFQ_history.xlsx")
    df = clean_cols(df)

    require_cols(df, ["project","site","product_type","spec","size","lifetime_qty","margin_rate","win_lose"], "RFQ_history")

    df["win_flag"] = (df["win_lose"].astype(str).str.upper() == "Y").astype(int)
    df["lifetime_qty"] = pd.to_numeric(df["lifetime_qty"], errors="coerce").fillna(0)
    df["margin_rate"] = pd.to_numeric(df["margin_rate"], errors="coerce").fillna(0)
    df["spec"] = pd.to_numeric(df["spec"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")

    df["log_qty"] = np.log1p(df["lifetime_qty"])
    return df

@st.cache_data
def load_new_input():
    # 템플릿 구조(설명/공백/헤더) 고려: header=2
    proj = pd.read_excel("data/RFQ_new_input.xlsx", sheet_name="Project_Fact", header=2)
    lines = pd.read_excel("data/RFQ_new_input.xlsx", sheet_name="Lines", header=2)
    cost = pd.read_excel("data/RFQ_new_input.xlsx", sheet_name="Cost_Input", header=2)
    mat = pd.read_excel("data/RFQ_new_input.xlsx", sheet_name="Material_Input", header=2)
    ass = pd.read_excel("data/RFQ_new_input.xlsx", sheet_name="Assumption", header=2)

    proj = clean_cols(proj)
    lines = clean_cols(lines)
    cost = clean_cols(cost)
    mat = clean_cols(mat)
    ass = clean_cols(ass)

    # line_id merge 안정화(공백/형식)
    if "line_id" in lines.columns:
        lines["line_id"] = lines["line_id"].astype(str).str.strip()
    if "line_id" in cost.columns:
        cost["line_id"] = cost["line_id"].astype(str).str.strip()
    if "line_id" in mat.columns:
        mat["line_id"] = mat["line_id"].astype(str).str.strip()

    require_cols(proj, ["project", "sop"], "Project_Fact")
    require_cols(lines, ["line_id","site","project","product_type","spec","size","lifetime_qty"], "Lines")
    require_cols(cost, ["line_id","plate_tool_cost","cutting_tool_cost","cover_tool_cost","dev_total_cost"], "Cost_Input")
    require_cols(mat, ["line_id","main_usage_qty","main_unit_price","sub_usage_qty","sub_unit_price","processing_cost"], "Material_Input")
    require_cols(ass, ["item","value"], "Assumption")

    # numeric safety
    for c in ["spec","size","lifetime_qty"]:
        lines[c] = pd.to_numeric(lines[c], errors="coerce")

    for c in ["plate_tool_cost","cutting_tool_cost","cover_tool_cost","dev_total_cost"]:
        cost[c] = pd.to_numeric(cost[c], errors="coerce").fillna(0)

    for c in ["main_usage_qty","main_unit_price","sub_usage_qty","sub_unit_price","processing_cost"]:
        mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0)

    ass["item"] = ass["item"].astype(str).str.strip()
    ass["value"] = pd.to_numeric(ass["value"], errors="coerce")

    return proj, lines, cost, mat, ass

hist = load_history()
proj, lines, cost, mat, ass = load_new_input()

if debug:
    st.write("DEBUG cost.sum()", cost[["plate_tool_cost","cutting_tool_cost","cover_tool_cost","dev_total_cost"]].sum())
    st.write("DEBUG lines.head()", lines.head())
    st.write("DEBUG cost.head()", cost.head())

# =========================
# Train model from history
# =========================
@st.cache_resource
def train_model(df):
    X = df[["site","product_type","spec","size","log_qty","margin_rate"]].copy()
    y = df["win_flag"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    cat = ["site","product_type","spec","size"]
    num = ["log_qty","margin_rate"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num)
    ])

    pipe = Pipeline([
        ("prep", pre),
        ("model", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)
    return pipe

model = train_model(hist)

# =========================
# Build current project DF
# =========================
df = lines.merge(cost, on="line_id", how="left").merge(mat, on="line_id", how="left")

# fill missing
for c in ["plate_tool_cost","cutting_tool_cost","cover_tool_cost","dev_total_cost"]:
    df[c] = df[c].fillna(0)
df["lifetime_qty"] = df["lifetime_qty"].fillna(0)

# =========================
# Shared amortization rules
# tooling shared by (project, size)
# dev shared by (project, product_type)
# =========================
df["tool_key"] = df["project"].astype(str) + "_" + df["size"].fillna(0).astype(int).astype(str)
df["dev_key"]  = df["project"].astype(str) + "_" + df["product_type"].astype(str)

tool_group_cost = (
    df.groupby("tool_key")[["plate_tool_cost","cutting_tool_cost","cover_tool_cost"]]
      .max()
      .sum(axis=1)
)
tool_group_qty = df.groupby("tool_key")["lifetime_qty"].sum().replace(0, np.nan)
tool_amort_per_unit = (tool_group_cost / tool_group_qty).fillna(0)

dev_group_cost = df.groupby("dev_key")["dev_total_cost"].max()
dev_group_qty  = df.groupby("dev_key")["lifetime_qty"].sum().replace(0, np.nan)
dev_amort_per_unit = (dev_group_cost / dev_group_qty).fillna(0)

df["tool_amort_per_unit"] = df["tool_key"].map(tool_amort_per_unit).fillna(0)
df["dev_amort_per_unit"]  = df["dev_key"].map(dev_amort_per_unit).fillna(0)

# =========================
# Cost engine
# =========================
plate_factor = get_assumption(ass, "plate_factor", 13.3)
sensor_cost  = get_assumption(ass, "sensor_cost", 30)
piece_factor = get_assumption(ass, "piece_factor", 2.5)
inner_cost   = get_assumption(ass, "inner_cost", 90)
sga_ratio    = get_assumption(ass, "sga_ratio", 0.5)

df["material_cost_per_unit"] = (
    df["main_usage_qty"] * df["main_unit_price"] +
    df["sub_usage_qty"]  * df["sub_unit_price"]
)

df["plate_cost"]  = df["size"].astype(float) * plate_factor
df["cover_cost"]  = df["plate_cost"] / 2.0
df["sensor_cost"] = sensor_cost
df["piece_cost"]  = sensor_cost * piece_factor
df["inner_cost"]  = inner_cost

df["sub_parts_cost_per_unit"] = (
    df["plate_cost"] + df["cover_cost"] + df["sensor_cost"] + df["piece_cost"] + df["inner_cost"]
)

df["sga_cost_per_unit"] = df["processing_cost"] * sga_ratio

df["unit_cost"] = (
    df["material_cost_per_unit"] +
    df["sub_parts_cost_per_unit"] +
    df["processing_cost"] +
    df["sga_cost_per_unit"] +
    df["tool_amort_per_unit"] +
    df["dev_amort_per_unit"]
)

# For project summary tables
df["line_total_cost"] = df["unit_cost"] * df["lifetime_qty"]
df["material_total"] = df["material_cost_per_unit"] * df["lifetime_qty"]
df["processing_total"] = df["processing_cost"] * df["lifetime_qty"]
df["amort_total"] = (df["tool_amort_per_unit"] + df["dev_amort_per_unit"]) * df["lifetime_qty"]

# =========================
# Similar history filter: spec+size+product_type (BEST)
# =========================
current_keys = df[["spec","size","product_type"]].dropna().drop_duplicates()
sim = hist.merge(current_keys, on=["spec","size","product_type"], how="inner")

# =========================
# Margin sweep settings
# =========================
st.sidebar.header("Margin Sweep 설정")
m_min = st.sidebar.slider("최소 마진", -0.05, 0.30, 0.04, 0.005)   # allow negative to show loss lines
m_max = st.sidebar.slider("최대 마진", -0.05, 0.30, 0.15, 0.005)
step  = st.sidebar.selectbox("Step", [0.0025, 0.005, 0.01], index=1)
band_pct = st.sidebar.slider("🟢 최적 구간(최대 기대이익 대비 %)", 0.80, 0.99, 0.95, 0.01)

if m_max <= m_min:
    st.sidebar.error("최대 마진은 최소 마진보다 커야 해.")
    st.stop()

margins = np.arange(m_min, m_max + 1e-12, step)

# =========================
# Sweep compute
# =========================
rows = []
for m in margins:
    for _, r in df.iterrows():
        X = pd.DataFrame([{
            "site": r["site"],
            "product_type": r["product_type"],
            "spec": r["spec"],
            "size": r["size"],
            "log_qty": np.log1p(float(r["lifetime_qty"])),
            "margin_rate": m
        }])
        win_p = float(model.predict_proba(X)[0, 1])

        raw_profit = float(r["unit_cost"]) * m * float(r["lifetime_qty"])
        exp_profit = raw_profit * win_p

        rows.append({
            "margin": float(m),
            "line_id": r["line_id"],
            "win_prob": win_p,
            "raw_profit": raw_profit,
            "expected_profit": exp_profit
        })

res = pd.DataFrame(rows)
proj_res = res.groupby("margin", as_index=False)["expected_profit"].sum()
best_margin = float(proj_res.loc[proj_res["expected_profit"].idxmax(), "margin"])
best_val = float(proj_res["expected_profit"].max())

# Optimal band margins
band_threshold = band_pct * best_val
band = proj_res[proj_res["expected_profit"] >= band_threshold].copy()
band_min = float(band["margin"].min()) if len(band) else best_margin
band_max = float(band["margin"].max()) if len(band) else best_margin

# =========================
# UI
# =========================
st.title("RFQ Optimal Margin Decision Dashboard")

# -------- 1) Project Summary --------
st.header("1. 프로젝트 요약")
project_name = str(proj.loc[0, "project"])
sop = proj.loc[0, "sop"]

colA, colB, colC, colD = st.columns(4)
colA.metric("Project", project_name)
colB.metric("SOP", int(sop) if not pd.isna(sop) else "-")
colC.metric("라인 수", int(len(df)))
colD.metric("총 물량", f"{int(df['lifetime_qty'].sum()):,}")

st.subheader("요약(발표용)")
st.markdown(
f"""
- **문제 정의**: 라인(3~5개) 단위 RFQ에서 금형/개발비 상각과 재료비 변동 때문에, “얼마에 제출해야 수주와 이익을 동시에 잡는지”가 매번 감으로 결정됨  
- **해결 방향**: 과거 RFQ 데이터의 *마진율-수주여부* 패턴을 학습하고, 신규 RFQ는 원가 엔진으로 단위원가를 계산한 뒤 마진을 스윕하여 의사결정  
- **해결 방안**: 마진율을 최소~최대 범위로 변화시키며 **수주확률(모델) × 이익(원가 기반)**의 기대이익을 계산 → **기대이익 최대 마진**을 최적해로 선택  
- **결론**: 이번 프로젝트의 최적 마진은 **{best_margin*100:.1f}%** (🟢 최적 구간 {band_min*100:.1f}% ~ {band_max*100:.1f}%)
"""
)

st.divider()

# -------- 2) Cost status --------
st.header("2. 현 프로젝트의 원가 현황")

total_qty = df["lifetime_qty"].sum()
if total_qty == 0:
    st.error("Lines 시트의 lifetime_qty 합계가 0이에요. 수량을 입력해줘야 계산이 됩니다.")
    st.stop()

weighted_unit_cost = (df["unit_cost"] * df["lifetime_qty"]).sum() / total_qty
project_total_cost = float((df["unit_cost"] * df["lifetime_qty"]).sum())

c1, c2, c3 = st.columns(3)
c1.metric("총 물량", f"{int(total_qty):,}")
c2.metric("프로젝트 가중평균 단위원가", f"{weighted_unit_cost:,.2f} KRW/EA")
c3.metric("프로젝트 총원가(추정)", f"{project_total_cost:,.0f} KRW")

st.subheader("라인별 원가 요약(한 줄씩, 회계 스타일)")
show_cols = [
    "line_id","site","product_type","spec","size","lifetime_qty",
    "unit_cost","line_total_cost",
    "material_total","processing_total","amort_total"
]

display_df = df[show_cols].copy()
display_df["lifetime_qty"] = display_df["lifetime_qty"].map(fmt_int)
for c in ["unit_cost","line_total_cost","material_total","processing_total","amort_total"]:
    display_df[c] = display_df[c].map(fmt0)

st.dataframe(
    display_df.sort_values("line_total_cost", ascending=False),
    use_container_width=True
)

with st.expander("라인별 원가 상세(단가 구성요소, 회계 스타일)"):
    detail_cols = [
        "line_id","site","product_type","spec","size","lifetime_qty",
        "material_cost_per_unit","sub_parts_cost_per_unit","processing_cost","sga_cost_per_unit",
        "tool_amort_per_unit","dev_amort_per_unit","unit_cost"
    ]
    ddf = df[detail_cols].copy()
    ddf["lifetime_qty"] = ddf["lifetime_qty"].map(fmt_int)
    for c in ["material_cost_per_unit","sub_parts_cost_per_unit","processing_cost","sga_cost_per_unit","tool_amort_per_unit","dev_amort_per_unit","unit_cost"]:
        ddf[c] = ddf[c].map(fmt0)
    st.dataframe(ddf, use_container_width=True)

# Sanity check table
st.subheader("Sanity Check: 단위원가 구성 Top5 (어디가 폭발했는지 바로 확인)")
san = df.copy()
san["qty"] = san["lifetime_qty"].fillna(0)
san["material"] = san["material_cost_per_unit"].fillna(0)
san["subparts"] = san["sub_parts_cost_per_unit"].fillna(0)
san["proc"] = san["processing_cost"].fillna(0)
san["sga"] = san["sga_cost_per_unit"].fillna(0)
san["amort"] = (san["tool_amort_per_unit"].fillna(0) + san["dev_amort_per_unit"].fillna(0))
san["unit_cost_check"] = san["material"] + san["subparts"] + san["proc"] + san["sga"] + san["amort"]
san["line_total_check"] = san["unit_cost_check"] * san["qty"]

san_disp = san[["line_id","qty","material","subparts","proc","sga","amort","unit_cost_check","line_total_check"]].copy()
san_disp["qty"] = san_disp["qty"].map(fmt_int)
for c in ["material","subparts","proc","sga","amort","unit_cost_check","line_total_check"]:
    san_disp[c] = san_disp[c].map(fmt0)

st.dataframe(
    san_disp.sort_values("line_total_check", ascending=False).head(5),
    use_container_width=True
)
st.caption("Tip: 총원가가 비정상적으로 크면 material(투입량×단가), proc(가공비), subparts(plate_factor 등) 중 하나가 '단위'가 다른 경우가 많아요.")

st.divider()

# -------- 3) Similar history --------
st.header("3. 과거 유사 프로젝트의 마진율 이력과 수주 여부")
st.caption("유사 기준: spec + size + product_type가 일치하는 과거 RFQ")

if len(sim) == 0:
    st.warning("유사 조건(spec+size+type)이 일치하는 과거 RFQ가 없습니다.")
else:
    y = sim["win_flag"] + (np.random.rand(len(sim)) - 0.5) * 0.12
    fig2, ax2 = plt.subplots()
    ax2.scatter(sim["margin_rate"], y, alpha=0.6)
    ax2.set_yticks([0,1])
    ax2.set_yticklabels(["Lose","Win"])
    ax2.set_xlabel("Margin Rate")
    ax2.set_ylabel("Win/Lose (jittered)")
    ax2.set_title(f"Similar RFQs ({len(sim)} rows): Margin vs Win/Lose")
    st.pyplot(fig2)

    with st.expander("유사 RFQ 테이블(상위 50개)"):
        sim_show = sim[["project","site","product_type","spec","size","lifetime_qty","margin_rate","win_lose"]].head(50).copy()
        sim_show["lifetime_qty"] = sim_show["lifetime_qty"].map(fmt_int)
        sim_show["margin_rate"] = sim_show["margin_rate"].map(lambda x: f"{float(x)*100:.2f}%")
        st.dataframe(sim_show, use_container_width=True)

st.divider()

# -------- 4) Optimal margin --------
st.header("4. 학습 결과 기반 현 프로젝트의 최적 마진율")
st.metric("최적 마진율(기대이익 최대)", f"{best_margin*100:.1f}%")
st.caption(f"🟢 최적 마진 구간(최대 기대이익의 {band_pct*100:.0f}% 이상): {band_min*100:.1f}% ~ {band_max*100:.1f}%")

fig3, ax3 = plt.subplots()
ax3.plot(proj_res["margin"], proj_res["expected_profit"], marker="o")

# Shade optimal band
ax3.axvspan(band_min, band_max, alpha=0.15)

# Best margin line
ax3.axvline(best_margin, linestyle="--")
ax3.set_xlabel("Margin Rate")
ax3.set_ylabel("Project Expected Profit (sum of lines)")
ax3.set_title("PROJECT: Margin vs Expected Profit")
ax3.grid(True)
y_comma(ax3)
st.pyplot(fig3)

st.divider()

# -------- 5) Sweep dashboard (line selection + view modes) --------
st.header("5. 최소~최대 마진 범위에서 결과 확인(스윕 대시보드)")

line_ids = sorted(df["line_id"].unique().tolist())
selected_lines = st.multiselect("보고 싶은 라인 선택", line_ids, default=line_ids)

df_sel = df[df["line_id"].isin(selected_lines)].copy()
res_sel = res[res["line_id"].isin(selected_lines)].copy()

if len(df_sel) == 0:
    st.warning("선택된 라인이 없습니다.")
    st.stop()

view_mode = st.radio(
    "라인 그래프 보기 방식",
    ["절대값(원래 값)", "정규화(라인별 0~1)", "로그스케일(절대값)"],
    horizontal=True
)

picked = st.slider("확인할 마진 선택", float(m_min), float(m_max), float(best_margin), float(step))

picked_rows = res_sel[np.isclose(res_sel["margin"], picked)]
project_expected_profit = float(picked_rows["expected_profit"].sum())

# project win prob weighted avg by qty
wprob = (
    picked_rows.merge(df_sel[["line_id","lifetime_qty"]], on="line_id", how="left")
    .eval("win_prob*lifetime_qty").sum()
    / df_sel["lifetime_qty"].sum()
)

col1, col2, col3 = st.columns(3)
col1.metric("선택 마진", f"{picked*100:.1f}%")
col2.metric("예상 수주확률(가중평균)", f"{wprob*100:.1f}%")
col3.metric("선택 라인 기대이익 합", f"{project_expected_profit:,.0f} KRW")

# Line-level snapshot at picked margin
snap = picked_rows.groupby("line_id", as_index=False).agg(
    win_prob=("win_prob","mean"),
    expected_profit=("expected_profit","sum"),
    raw_profit=("raw_profit","sum")
)
snap = snap.merge(df_sel[["line_id","site","product_type","spec","size","lifetime_qty","unit_cost"]], on="line_id", how="left")

# Define loss lines: expected_profit < 0 (possible when margin is negative)
snap["is_loss"] = snap["expected_profit"] < 0

def style_loss(row):
    if row.get("is_loss", False):
        return ["color: #b00020; font-weight: 700"] * len(row)
    return [""] * len(row)

st.subheader("🔴 라인별 손익 스냅샷(선택 마진 기준)")
snap_disp = snap.copy()
snap_disp["lifetime_qty"] = snap_disp["lifetime_qty"].map(fmt_int)
snap_disp["unit_cost"] = snap_disp["unit_cost"].map(fmt0)
snap_disp["expected_profit"] = snap_disp["expected_profit"].map(fmt0)
snap_disp["raw_profit"] = snap_disp["raw_profit"].map(fmt0)
snap_disp["win_prob"] = snap_disp["win_prob"].map(lambda x: f"{float(x)*100:.1f}%")
snap_disp["is_loss"] = snap["is_loss"].map(lambda x: "LOSS" if x else "")

st.dataframe(
    snap_disp.style.apply(style_loss, axis=1),
    use_container_width=True
)

# line curves
line_curve = res_sel.groupby(["line_id","margin"], as_index=False)["expected_profit"].sum()

plot_df = line_curve.copy()
if view_mode == "정규화(라인별 0~1)":
    plot_df["expected_profit_norm"] = plot_df.groupby("line_id")["expected_profit"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    )

fig4, ax4 = plt.subplots(figsize=(9, 5))
for lid in sorted(plot_df["line_id"].unique()):
    d = plot_df[plot_df["line_id"] == lid]
    if view_mode == "정규화(라인별 0~1)":
        ax4.plot(d["margin"], d["expected_profit_norm"], marker="o", label=lid)
    else:
        ax4.plot(d["margin"], d["expected_profit"], marker="o", label=lid)

# Shade optimal band + best line
ax4.axvspan(band_min, band_max, alpha=0.12)
ax4.axvline(best_margin, linestyle="--")

ax4.set_xlabel("Margin Rate")
if view_mode == "정규화(라인별 0~1)":
    ax4.set_ylabel("Expected Profit (normalized)")
    ax4.set_title("LINES: Margin vs Expected Profit (Normalized)")
else:
    ax4.set_ylabel("Expected Profit (per line)")
    ax4.set_title("LINES: Margin vs Expected Profit")

if view_mode == "로그스케일(절대값)":
    ax4.set_yscale("log")
else:
    y_comma(ax4)

ax4.grid(True)
ax4.legend()
st.pyplot(fig4)

with st.expander("프로젝트 마진별 기대이익 테이블(선택 라인 합, 회계 스타일)"):
    proj_sel = res_sel.groupby("margin", as_index=False)["expected_profit"].sum()
    proj_sel_disp = proj_sel.copy()
    proj_sel_disp["expected_profit"] = proj_sel_disp["expected_profit"].map(fmt0)
    proj_sel_disp["margin"] = proj_sel_disp["margin"].map(lambda x: f"{float(x)*100:.2f}%")
    st.dataframe(proj_sel_disp, use_container_width=True)

st.divider()

# -------- 6) Export --------
st.header("6. 📥 결과 엑셀 Export (콤마 유지)")

export_bytes = make_download_excel(project_name, proj, df, proj_res, res, sim)
st.download_button(
    label="📥 RFQ_Analysis_Result.xlsx 다운로드",
    data=export_bytes,
    file_name="RFQ_Analysis_Result.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("※ Export 파일은 '표시용(콤마 포함 문자열)'로 저장됩니다. 추가 계산이 필요하면 원본 숫자 컬럼을 별도로 저장하도록 확장할 수도 있어요.")


st.caption("※ 이 대시보드는 '과거 RFQ 데이터로 학습된 마진-수주 패턴' + '현 프로젝트 원가 엔진'을 결합해 기대이익 최대 마진을 추천합니다.")

