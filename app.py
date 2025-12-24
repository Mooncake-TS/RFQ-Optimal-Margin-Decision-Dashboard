import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# =========================
# Number format helpers (accounting-style)
# =========================
def fmt_int(x):
    """Return integer with commas. Safe for NaN/None/strings."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        s = str(x).strip()
        # try to remove commas then parse
        try:
            return f"{int(round(float(s.replace(',', '')))):,}"
        except Exception:
            return s

def fmt_krw(x, digits=0):
    """Return KRW with commas."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        if digits == 0:
            return f"{float(x):,.0f}"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return str(x)

def fmt_pct(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{float(x)*100:.{digits}f}%"
    except Exception:
        return str(x)

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

# 부자재 규칙(너가 말한 기준 반영)
# plate = size * plate_factor
# sensor = sensor_cost
# cover = plate/2 (원하면 켜기)
# piece = sensor*piece_factor
# inner = inner_cost
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

# =========================
# Similar history filter: spec+size+product_type (BEST)
# =========================
current_keys = df[["spec","size","product_type"]].dropna().drop_duplicates()
sim = hist.merge(current_keys, on=["spec","size","product_type"], how="inner")

# =========================
# Margin sweep
# =========================
st.sidebar.header("Margin Sweep 설정")
m_min = st.sidebar.slider("최소 마진", 0.00, 0.30, 0.04, 0.005)
m_max = st.sidebar.slider("최대 마진", 0.00, 0.30, 0.15, 0.005)
step  = st.sidebar.selectbox("Step", [0.0025, 0.005, 0.01], index=1)

if m_max <= m_min:
    st.sidebar.error("최대 마진은 최소 마진보다 커야 해.")
    st.stop()

margins = np.arange(m_min, m_max + 1e-12, step)

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
            "margin": m,
            "line_id": r["line_id"],
            "win_prob": win_p,
            "raw_profit": raw_profit,
            "expected_profit": exp_profit
        })

res = pd.DataFrame(rows)
proj_res = res.groupby("margin", as_index=False)["expected_profit"].sum()
best_margin = float(proj_res.loc[proj_res["expected_profit"].idxmax(), "margin"])

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
- **문제 정의**: RFQ 프로젝트 내 라인별 Fixed Cost 상각과 Direct Material 변동, 수주 수량 변화에 맞추어 “얼마에 제출해야 수주와 이익을 동시에 잡는지”가 매번 감으로 결정됨  
- **해결 방향**: 과거 RFQ 데이터의 *수량-마진율-수주여부* 및 패턴을 학습하고, 신규 RFQ는 원가 엔진으로 단위 원가를 계산한 뒤 마진을 스윕하여 의사결정  
- **해결 방안**: 마진율을 최소 ~ 최대 범위로 변화시키며 **수주확률(모델) × 이익(원가 기반)**의 기대이익을 계산 → **기대이익 최대 마진**을 최적해로 선택  
- **결론**: 이번 프로젝트의 최적 마진은 **{best_margin*100:.1f}%** (Step5에서 마진 변화에 따른 결과를 직접 확인 가능)
"""
)

st.divider()

# -------- 2) Cost status (UPDATED: no pie, line summary) --------
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

st.subheader("라인별 원가 요약(한 줄씩)")
line_summary = df.copy()
line_summary["line_total_cost"] = line_summary["unit_cost"] * line_summary["lifetime_qty"]
line_summary["material_total"] = line_summary["material_cost_per_unit"] * line_summary["lifetime_qty"]
line_summary["processing_total"] = line_summary["processing_cost"] * line_summary["lifetime_qty"]
line_summary["amort_total"] = (line_summary["tool_amort_per_unit"] + line_summary["dev_amort_per_unit"]) * line_summary["lifetime_qty"]

show_cols = [
    "line_id","site","product_type","spec","size","lifetime_qty",
    "unit_cost","line_total_cost",
    "material_total","processing_total","amort_total"
]
st.dataframe(
    line_summary[show_cols].sort_values("line_total_cost", ascending=False),
    use_container_width=True
)

with st.expander("라인별 원가 상세(단가 구성요소)"):
    detail_cols = [
        "line_id","site","product_type","spec","size","lifetime_qty",
        "material_cost_per_unit","sub_parts_cost_per_unit","processing_cost","sga_cost_per_unit",
        "tool_amort_per_unit","dev_amort_per_unit","unit_cost"
    ]
    st.dataframe(df[detail_cols], use_container_width=True)

st.divider()

# -------- 3) Similar history --------
st.header("3. 과거 유사 프로젝트의 마진율 이력과 수주 여부")
st.caption("유사 기준: spec + size + product_type가 일치하는 과거 RFQ")

if len(sim) == 0:
    st.warning("유사 조건(spec+size+type)이 일치하는 과거 RFQ가 없습니다.")
else:
    colL, colR = st.columns(2, gap="large")

    # -----------------------------
    # (A) Margin vs Win/Lose
    # -----------------------------
    with colL:
        st.subheader("마진율 vs 수주 여부 (유사 RFQ)")

        # jitter for visibility
        y = sim["win_flag"] + (np.random.rand(len(sim)) - 0.5) * 0.12

        # Quantile-based "low/high" zones to highlight pattern
        q_low = float(sim["margin_rate"].quantile(0.25))
        q_high = float(sim["margin_rate"].quantile(0.75))

        low_zone = sim[sim["margin_rate"] <= q_low]
        high_zone = sim[sim["margin_rate"] >= q_high]

        win_low = float(low_zone["win_flag"].mean()) if len(low_zone) else np.nan
        win_high = float(high_zone["win_flag"].mean()) if len(high_zone) else np.nan

        # Visual: shaded zones + scatter + binned win-rate curve (thicker)
        fig_m, ax_m = plt.subplots()

        # Shade zones (left = low margin, right = high margin)
        ax_m.axvspan(sim["margin_rate"].min(), q_low, alpha=0.18)
        ax_m.axvspan(q_high, sim["margin_rate"].max(), alpha=0.18)

        ax_m.scatter(sim["margin_rate"], y, alpha=0.6)

        # Binned win-rate curve (shows "low margin -> higher win" pattern clearly)
        sim_tmp = sim[["margin_rate", "win_flag"]].dropna().copy()
        # Use quantile bins for stability
        qs = np.quantile(sim_tmp["margin_rate"], [0, .125, .25, .375, .5, .625, .75, .875, 1.0])
        # Make strictly increasing (handle duplicates)
        bins = np.unique(qs)
        if len(bins) >= 4:
            sim_tmp["bin"] = pd.cut(sim_tmp["margin_rate"], bins=bins, include_lowest=True)
            win_by_bin = sim_tmp.groupby("bin", observed=True)["win_flag"].mean()
            x_mid = np.array([(b.left + b.right) / 2 for b in win_by_bin.index])
            ax_m.plot(x_mid, win_by_bin.values, marker="o", linewidth=2.5)

        ax_m.set_yticks([0, 1])
        ax_m.set_yticklabels(["Lose", "Win"])
        ax_m.set_xlabel("Margin Rate")
        ax_m.set_ylabel("Win/Lose (jittered)")
        ax_m.set_title(f"Similar RFQs ({len(sim)} rows)")

        # (No on-plot callouts to avoid overlap)
        ax_m.grid(True)
        st.pyplot(fig_m)

        st.caption("해석 가이드: 좌측(저마진) 음영 구간의 평균수주율이 더 높게 나오면 \"낮은 마진일수록 수주가 잘 되는 경향\"을 보여줘요.")

        # Extra readability: show the key comparison as metrics
        cL1, cL2 = st.columns(2)
        cL1.metric("저마진 구간 평균 수주율", f"{win_low*100:.1f}%" if not np.isnan(win_low) else "-")
        cL2.metric("고마진 구간 평균 수주율", f"{win_high*100:.1f}%" if not np.isnan(win_high) else "-")

        st.caption("포인트: 음영(저마진/고마진) 구간의 **평균 수주율**과, 가운데의 **빈 평균선(굵은 선)** 흐름을 함께 보면 메시지가 한 눈에 들어와요.")

    # -----------------------------
    # (B) Quantity vs Win/Lose
    # -----------------------------
    with colR:
        st.subheader("발주수량 vs 수주 여부 (유사 RFQ)")
        # Use log scale on x for readability
        sim_q = sim.copy()
        sim_q["log_qty_plot"] = np.log1p(sim_q["lifetime_qty"].astype(float).fillna(0))
        y_qty = sim_q["win_flag"] + (np.random.rand(len(sim_q)) - 0.5) * 0.12

        ql = float(sim_q["log_qty_plot"].quantile(0.25))
        qh = float(sim_q["log_qty_plot"].quantile(0.75))

        fig_q, ax_q = plt.subplots()
        ax_q.axvspan(sim_q["log_qty_plot"].min(), ql, alpha=0.12)
        ax_q.axvspan(qh, sim_q["log_qty_plot"].max(), alpha=0.12)

        ax_q.scatter(sim_q["log_qty_plot"], y_qty, alpha=0.6)

        # Binned win-rate line (in log space)
        bins_q = np.linspace(sim_q["log_qty_plot"].min(), sim_q["log_qty_plot"].max(), 9)
        tmp = sim_q[["log_qty_plot", "win_flag"]].dropna().copy()
        tmp["bin"] = pd.cut(tmp["log_qty_plot"], bins=bins_q, include_lowest=True)
        win_by_bin_q = tmp.groupby("bin", observed=True)["win_flag"].mean()
        x_mid_q = np.array([(b.left + b.right) / 2 for b in win_by_bin_q.index])
        ax_q.plot(x_mid_q, win_by_bin_q.values, marker="o")

        ax_q.set_yticks([0, 1])
        ax_q.set_yticklabels(["Lose", "Win"])
        ax_q.set_xlabel("log(1 + Lifetime Quantity)")
        ax_q.set_ylabel("Win/Lose (jittered)")
        ax_q.set_title(f"Similar RFQs ({len(sim)} rows)")
        ax_q.grid(True)

        ax_q.text((sim_q["log_qty_plot"].min()+ql)/2, 1.05, "Low qty zone", ha="center", va="bottom")
        ax_q.text((qh+sim_q["log_qty_plot"].max())/2, 1.05, "High qty zone", ha="center", va="bottom")

        st.pyplot(fig_q)

        st.caption("해석 가이드: 우측(고수량) 음영 구간의 빈 평균선이 더 높게 나오면 '수량이 많을수록 수주가 잘 되는 경향'이 있음을 보여줘요.")

    with st.expander("유사 RFQ 테이블(상위 50개)"):
        sim_show = sim[["project","site","product_type","spec","size","lifetime_qty","margin_rate","win_lose"]].head(50).copy()
        sim_show["lifetime_qty"] = sim_show["lifetime_qty"].map(fmt_int)
        sim_show["margin_rate"] = sim_show["margin_rate"].map(lambda x: f"{float(x)*100:.2f}%")
        st.dataframe(sim_show, use_container_width=True)

st.divider()


# -------- 4) Optimal margin --------
st.header("4. 학습 결과 기반 현 프로젝트의 최적 마진율")
st.metric("최적 마진율(기대이익 최대) -> 수주 확률과 연계되기에 일정 마진율을 넘어가면 수주 확률이 내려가 기대이익이 줄어들어요.", f"{best_margin*100:.1f}%")

fig3, ax3 = plt.subplots()
ax3.plot(proj_res["margin"], proj_res["expected_profit"], marker="o")
ax3.axvline(best_margin, linestyle="--")
ax3.set_xlabel("Margin Rate")
ax3.set_ylabel("Project Expected Profit (sum of lines)")
ax3.set_title("PROJECT: Margin vs Expected Profit")
ax3.grid(True)
st.pyplot(fig3)

st.divider()

# -------- 5) Sweep dashboard (UPDATED: line selection + view modes) --------
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

picked_rows = res_sel[res_sel["margin"].round(6) == round(picked, 6)]
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

# line curves
line_curve = res_sel.groupby(["line_id","margin"], as_index=False)["expected_profit"].sum()

# Apply view mode transforms
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

ax4.grid(True)
ax4.legend()
st.pyplot(fig4)

with st.expander("프로젝트 마진별 기대이익 테이블(선택 라인 합)"):
    proj_sel = res_sel.groupby("margin", as_index=False)["expected_profit"].sum()
    st.dataframe(proj_sel, use_container_width=True)

st.caption("※ 이 대시보드는 '과거 RFQ 데이터로 학습된 마진-수주 패턴' + '현 프로젝트 원가 엔진'을 결합해 기대이익 최대 마진을 추천합니다.")
