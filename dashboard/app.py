
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Dashboard ABM Pola Tidur Mahasiswa",
    page_icon="😴",
    layout="wide"
)

st.title("😴 Dashboard ABM Pola Tidur dan Produktivitas Mahasiswa")
st.caption(
    "Dashboard interaktif untuk presentasi final project Pemodelan dan Simulasi."
)


# ==========================================================
# PARAMETER GLOBAL SESUAI NOTEBOOK
# ==========================================================

SIM_TIME_DEFAULT = 16
NUM_STUDENTS_DEFAULT = 30
N_ITERASI_DEFAULT = 1000
BASE_SEED = 2026

BOBOT_TIDUR = 0.28
BOBOT_KUALITAS = 0.22
BOBOT_REGULARITAS = 0.15
BOBOT_STRES = 0.25
BOBOT_INTERVENSI = 0.18
FAKTOR_PERUBAHAN = 0.22

IDEAL_SLEEP = 7.5
INTERVENSI_THRESHOLD = 0.55
PRODUKTIVITAS_RENDAH = 0.40
PRODUKTIVITAS_TINGGI = 0.70

EXAM_WEEKS = (7, 14)
EXAM_SPIKE = 0.15


BASELINE_PARAMS = {
    "PL_min": 0.40, "PL_max": 0.75,
    "SH_min": 4.00, "SH_max": 6.00,
    "SQ_min": 0.25, "SQ_max": 0.55,
    "SR_min": 0.15, "SR_max": 0.40,
    "AS_min": 0.60, "AS_max": 0.90,
    "D_min": 1.20, "D_max": 2.00,
    "HI_min": 0.65, "HI_max": 0.85,
}


def ubah_parameter(base, **perubahan):
    hasil = base.copy()
    hasil.update(perubahan)
    return hasil


SKENARIO_ALL = {
    "Tanpa Intervensi": ubah_parameter(
        BASELINE_PARAMS,
        mode="none",
        kategori="Eksperimen utama",
        deskripsi="Agen mengalami stressor akademik tanpa dukungan intervensi higiene tidur."
    ),
    "Intervensi Reaktif": ubah_parameter(
        BASELINE_PARAMS,
        mode="reaktif",
        kategori="Eksperimen utama",
        deskripsi="Intervensi aktif saat PL agen berada di bawah ambang 0,55."
    ),
    "Intervensi Preventif": ubah_parameter(
        BASELINE_PARAMS,
        mode="preventif",
        kategori="Eksperimen utama",
        deskripsi="Intervensi aktif secara rutin sejak awal simulasi."
    ),
    "Distorsi Kognitif Tinggi": ubah_parameter(
        BASELINE_PARAMS,
        mode="reaktif",
        D_min=2.00,
        D_max=3.00,
        kategori="Eksperimen utama",
        deskripsi="Agen memiliki distorsi kognitif tinggi sehingga stressor tidur berdampak lebih besar."
    ),
    "Kombinasi Ideal (Benchmark)": ubah_parameter(
        BASELINE_PARAMS,
        mode="preventif",
        SH_min=7.00, SH_max=8.20,
        SQ_min=0.75, SQ_max=0.95,
        SR_min=0.75, SR_max=0.95,
        AS_min=0.25, AS_max=0.50,
        D_min=0.80, D_max=1.20,
        HI_min=0.80, HI_max=1.00,
        kategori="Benchmark eksploratif",
        deskripsi="Kondisi pembanding terbaik dengan tidur lebih baik, stressor rendah, dan intervensi kuat."
    ),
}


# ==========================================================
# FUNGSI MODEL
# ==========================================================

def state_produktivitas(pl):
    if pl >= PRODUKTIVITAS_TINGGI:
        return "Tinggi"
    if pl >= PRODUKTIVITAS_RENDAH:
        return "Sedang"
    return "Rendah"


def hitung_sleep_score(sh):
    deviasi = abs(sh - IDEAL_SLEEP)
    score = 1 - min(deviasi / 4.5, 1)
    return max(0.0, min(score, 1.0))


@st.cache_data(show_spinner=False)
def buat_dataset_dummy_cached(params_tuple, n_mahasiswa, seed):
    params = dict(params_tuple)
    rng = random.Random(seed)
    data = []

    for i in range(1, n_mahasiswa + 1):
        data.append({
            "Nama": f"Mhs-{i}",
            "PL_awal": rng.uniform(params["PL_min"], params["PL_max"]),
            "SH_awal": rng.uniform(params["SH_min"], params["SH_max"]),
            "SQ_awal": rng.uniform(params["SQ_min"], params["SQ_max"]),
            "SR_awal": rng.uniform(params["SR_min"], params["SR_max"]),
            "AS": rng.uniform(params["AS_min"], params["AS_max"]),
            "D": rng.uniform(params["D_min"], params["D_max"]),
            "HI": rng.uniform(params["HI_min"], params["HI_max"]),
        })

    return pd.DataFrame(data)


def buat_dataset_dummy(params, n_mahasiswa, seed):
    clean = {k: v for k, v in params.items() if isinstance(v, (int, float, str))}
    return buat_dataset_dummy_cached(tuple(sorted(clean.items())), n_mahasiswa, seed)


def run_simulation(params, seed, n_mahasiswa, sim_time, collect_history=True):
    initial_data = buat_dataset_dummy(params, n_mahasiswa=n_mahasiswa, seed=seed)
    rng = random.Random(seed)

    agents = []
    for row in initial_data.itertuples(index=False):
        agents.append({
            "Nama": row.Nama,
            "PL": float(row.PL_awal),
            "SH": float(row.SH_awal),
            "SQ": float(row.SQ_awal),
            "SR": float(row.SR_awal),
            "AS": float(row.AS),
            "D": float(row.D),
            "HI": float(row.HI),
            "State_Sebelumnya": state_produktivitas(float(row.PL_awal)),
            "Jumlah_Intervensi": 0,
            "Jumlah_Minggu_Rendah": 0,
            "SleepScore": hitung_sleep_score(float(row.SH_awal)),
        })

    records = []

    for minggu in range(1, sim_time + 1):
        rng.shuffle(agents)

        for agent in agents:
            perubahan_jam = rng.uniform(-0.35, 0.35) * (1 - agent["SR"])
            perubahan_kualitas = rng.uniform(-0.04, 0.03)

            agent["SH"] = max(3.0, min(agent["SH"] + perubahan_jam, 10.0))
            agent["SQ"] = max(0.10, min(agent["SQ"] + perubahan_kualitas, 1.0))

            exam_spike = EXAM_SPIKE if minggu in EXAM_WEEKS else 0.0
            stres_akademik = min(1.0, agent["AS"] + exam_spike)
            sleep_score = hitung_sleep_score(agent["SH"])
            stres_tidur = (1 - sleep_score) * (1 - agent["SR"])

            manfaat_tidur = (
                BOBOT_TIDUR * sleep_score
                + BOBOT_KUALITAS * agent["SQ"]
                + BOBOT_REGULARITAS * agent["SR"]
            )

            dampak_stres = BOBOT_STRES * (
                stres_akademik + agent["D"] * stres_tidur
            )

            delta = FAKTOR_PERUBAHAN * (manfaat_tidur - dampak_stres)

            if delta >= 0:
                agent["PL"] += delta * (1 - agent["PL"])
            else:
                agent["PL"] += delta * agent["PL"]

            intervensi_aktif = False
            if params["mode"] == "preventif":
                intervensi_aktif = True
            elif params["mode"] == "reaktif" and agent["PL"] < INTERVENSI_THRESHOLD:
                intervensi_aktif = True

            if intervensi_aktif and agent["HI"] > 0:
                agent["Jumlah_Intervensi"] += 1
                efek_intervensi = BOBOT_INTERVENSI * agent["HI"]

                agent["PL"] += 0.08 * efek_intervensi * (1 - agent["PL"])
                agent["SH"] += (IDEAL_SLEEP - agent["SH"]) * 0.10 * agent["HI"]
                agent["SQ"] = min(1.0, agent["SQ"] + 0.025 * agent["HI"])
                agent["SR"] = min(1.0, agent["SR"] + 0.030 * agent["HI"])

            agent["PL"] = max(0.0, min(agent["PL"], 1.0))
            agent["SleepScore"] = hitung_sleep_score(agent["SH"])

            state_now = state_produktivitas(agent["PL"])
            pulih = agent["State_Sebelumnya"] == "Rendah" and state_now != "Rendah"

            if state_now == "Rendah":
                agent["Jumlah_Minggu_Rendah"] += 1

            if collect_history:
                records.append({
                    "Nama": agent["Nama"],
                    "Minggu": minggu,
                    "PL": agent["PL"],
                    "SH": agent["SH"],
                    "SQ": agent["SQ"],
                    "SR": agent["SR"],
                    "AS": agent["AS"],
                    "D": agent["D"],
                    "HI": agent["HI"],
                    "SleepScore": agent["SleepScore"],
                    "StresAkademik": stres_akademik,
                    "StresTidur": stres_tidur,
                    "Intervensi": intervensi_aktif,
                    "Produktivitas": state_now,
                    "Transisi": (
                        f"{agent['State_Sebelumnya']} → {state_now}"
                        if agent["State_Sebelumnya"] != state_now
                        else state_now
                    ),
                    "Pulih": pulih,
                })

            agent["State_Sebelumnya"] = state_now

    metrics = {
        "Avg_PL": np.mean([a["PL"] for a in agents]),
        "Pct_Rendah": np.mean([a["PL"] < PRODUKTIVITAS_RENDAH for a in agents]) * 100,
        "Pct_Tinggi": np.mean([a["PL"] >= PRODUKTIVITAS_TINGGI for a in agents]) * 100,
        "Avg_Intervensi": np.mean([a["Jumlah_Intervensi"] for a in agents]),
        "Avg_SleepScore": np.mean([a["SleepScore"] for a in agents]),
        "Avg_Minggu_Rendah": np.mean([a["Jumlah_Minggu_Rendah"] for a in agents]),
    }

    final_agents = pd.DataFrame(agents)
    history = pd.DataFrame(records)
    return metrics, initial_data, history, final_agents


@st.cache_data(show_spinner=True)
def run_monte_carlo(skenario_names, n_iterasi, n_mahasiswa, sim_time, seed_awal):
    rows = []

    for nama in skenario_names:
        params = SKENARIO_ALL[nama]

        for i in range(n_iterasi):
            metrics, _, _, _ = run_simulation(
                params=params,
                seed=seed_awal + i,
                n_mahasiswa=n_mahasiswa,
                sim_time=sim_time,
                collect_history=False
            )

            rows.append({
                "Skenario": nama,
                "Kategori": params["kategori"],
                "Iterasi": i + 1,
                "Seed": seed_awal + i,
                **metrics
            })

    df_iterasi = pd.DataFrame(rows)

    df_summary = (
        df_iterasi
        .groupby(["Skenario", "Kategori"], as_index=False)
        .agg(
            Avg_PL=("Avg_PL", "mean"),
            Std_PL=("Avg_PL", "std"),
            Pct_Produktivitas_Rendah=("Pct_Rendah", "mean"),
            Pct_Produktivitas_Tinggi=("Pct_Tinggi", "mean"),
            Avg_Intervensi=("Avg_Intervensi", "mean"),
            Avg_SleepScore=("Avg_SleepScore", "mean"),
            Avg_Minggu_Rendah=("Avg_Minggu_Rendah", "mean"),
        )
    )

    df_summary["SE_PL"] = df_summary["Std_PL"] / np.sqrt(n_iterasi)
    df_summary["CI95_PL"] = 1.96 * df_summary["SE_PL"]
    df_summary["CI95_Bawah"] = df_summary["Avg_PL"] - df_summary["CI95_PL"]
    df_summary["CI95_Atas"] = df_summary["Avg_PL"] + df_summary["CI95_PL"]

    order = list(SKENARIO_ALL.keys())
    df_summary["Skenario"] = pd.Categorical(
        df_summary["Skenario"],
        categories=order,
        ordered=True
    )
    df_summary = df_summary.sort_values("Skenario").reset_index(drop=True)

    baseline = df_summary.loc[
        df_summary["Skenario"].astype(str) == "Tanpa Intervensi",
        "Avg_PL"
    ]

    if not baseline.empty:
        baseline_pl = float(baseline.iloc[0])
        df_summary["Gain_vs_Tanpa_Intervensi"] = df_summary["Avg_PL"] - baseline_pl
        df_summary["Gain_PL_per_Intervensi"] = np.where(
            df_summary["Avg_Intervensi"] > 0,
            df_summary["Gain_vs_Tanpa_Intervensi"] / df_summary["Avg_Intervensi"],
            np.nan
        )

    return df_iterasi, df_summary


def format_float(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def buat_fig_line(df_weekly, y_col, y_label):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_weekly["Minggu"], df_weekly[y_col], marker="o")
    ax.set_xlabel("Minggu")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    return fig


def buat_fig_hist(series, x_label):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(series, bins=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Jumlah Agen")
    ax.grid(True, alpha=0.25)
    return fig


def buat_fig_bar(df, x_col, y_col, y_label):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    return fig


# ==========================================================
# SIDEBAR INPUT
# ==========================================================

with st.sidebar:
    st.header("Kontrol Simulasi")

    mode_dashboard = st.radio(
        "Mode Dashboard",
        [
            "Demo satu skenario",
            "Perbandingan semua skenario",
            "Eksperimen kustom"
        ]
    )

    n_mahasiswa = st.slider(
        "Jumlah agen mahasiswa",
        min_value=10,
        max_value=100,
        value=NUM_STUDENTS_DEFAULT,
        step=5
    )

    sim_time = st.slider(
        "Durasi simulasi",
        min_value=4,
        max_value=24,
        value=SIM_TIME_DEFAULT,
        step=1,
        help="Satuan waktu adalah minggu."
    )

    seed = st.number_input(
        "Seed simulasi",
        min_value=1,
        max_value=999999,
        value=BASE_SEED,
        step=1
    )

    n_iterasi = st.slider(
        "Jumlah iterasi Monte Carlo",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
        help="Gunakan 1000 untuk hasil final. Gunakan 100 sampai 200 saat demo agar lebih cepat."
    )

    if mode_dashboard == "Demo satu skenario":
        skenario_pilih = st.selectbox(
            "Pilih skenario",
            list(SKENARIO_ALL.keys())
        )
        params_aktif = SKENARIO_ALL[skenario_pilih].copy()

    elif mode_dashboard == "Eksperimen kustom":
        skenario_pilih = "Eksperimen Kustom"
        params_aktif = BASELINE_PARAMS.copy()
        params_aktif["kategori"] = "Eksperimen kustom"
        params_aktif["deskripsi"] = "Parameter diatur manual oleh pengguna."

        mode_intervensi = st.selectbox(
            "Mode intervensi",
            ["none", "reaktif", "preventif"],
            format_func=lambda x: {
                "none": "Tanpa intervensi",
                "reaktif": "Reaktif",
                "preventif": "Preventif"
            }[x]
        )
        params_aktif["mode"] = mode_intervensi

        st.markdown("**Rentang parameter awal agen**")
        params_aktif["AS_min"], params_aktif["AS_max"] = st.slider(
            "Academic Stress (AS)",
            0.0, 1.0, (0.60, 0.90), 0.05
        )
        params_aktif["D_min"], params_aktif["D_max"] = st.slider(
            "Cognitive Distortion (D)",
            0.5, 3.0, (1.20, 2.00), 0.05
        )
        params_aktif["HI_min"], params_aktif["HI_max"] = st.slider(
            "Hygiene Intervention (HI)",
            0.0, 1.0, (0.65, 0.85), 0.05
        )
        params_aktif["SR_min"], params_aktif["SR_max"] = st.slider(
            "Sleep Regularity (SR)",
            0.0, 1.0, (0.15, 0.40), 0.05
        )
        params_aktif["SH_min"], params_aktif["SH_max"] = st.slider(
            "Sleep Hours (SH)",
            3.0, 10.0, (4.0, 6.0), 0.25
        )
        params_aktif["SQ_min"], params_aktif["SQ_max"] = st.slider(
            "Sleep Quality (SQ)",
            0.1, 1.0, (0.25, 0.55), 0.05
        )

    else:
        skenario_pilih = "Semua Skenario"
        params_aktif = SKENARIO_ALL["Intervensi Preventif"].copy()

    st.divider()
    st.caption("Ambang state: rendah < 0,40 | sedang 0,40 sampai < 0,70 | tinggi ≥ 0,70")


# ==========================================================
# EKSEKUSI SIMULASI
# ==========================================================

if mode_dashboard in ["Demo satu skenario", "Eksperimen kustom"]:
    metrics, df_dummy, df_history, df_final = run_simulation(
        params=params_aktif,
        seed=int(seed),
        n_mahasiswa=int(n_mahasiswa),
        sim_time=int(sim_time),
        collect_history=True
    )

else:
    df_iterasi, df_summary = run_monte_carlo(
        skenario_names=list(SKENARIO_ALL.keys()),
        n_iterasi=int(n_iterasi),
        n_mahasiswa=int(n_mahasiswa),
        sim_time=int(sim_time),
        seed_awal=int(seed)
    )


# ==========================================================
# TAB DASHBOARD
# ==========================================================

if mode_dashboard in ["Demo satu skenario", "Eksperimen kustom"]:

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ringkasan",
        "Dataset Dummy",
        "Proses Simulasi",
        "Hasil Agen",
        "Cara Kerja Model"
    ])

    with tab1:
        st.subheader("Ringkasan Skenario")
        st.write(f"**Skenario aktif:** {skenario_pilih}")
        st.write(params_aktif.get("deskripsi", ""))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg PL Akhir", format_float(metrics["Avg_PL"]))
        c2.metric("Produktivitas Tinggi", f"{metrics['Pct_Tinggi']:.2f}%")
        c3.metric("Produktivitas Rendah", f"{metrics['Pct_Rendah']:.2f}%")
        c4.metric("Avg Intervensi", format_float(metrics["Avg_Intervensi"]))
        c5.metric("Avg SleepScore", format_float(metrics["Avg_SleepScore"]))

        st.markdown("#### Interpretasi Otomatis")
        if metrics["Avg_PL"] >= PRODUKTIVITAS_TINGGI:
            st.success(
                "Rata-rata produktivitas akhir berada pada kategori tinggi. "
                "Kondisi tidur dan intervensi cukup kuat untuk menahan dampak stressor."
            )
        elif metrics["Avg_PL"] >= PRODUKTIVITAS_RENDAH:
            st.warning(
                "Rata-rata produktivitas akhir berada pada kategori sedang. "
                "Stressor masih terlihat, tetapi agen belum dominan berada pada risiko rendah."
            )
        else:
            st.error(
                "Rata-rata produktivitas akhir berada pada kategori rendah. "
                "Stressor lebih dominan daripada manfaat tidur dan intervensi."
            )

        st.markdown("#### Parameter Skenario")
        tampil_params = pd.DataFrame([
            {"Parameter": k, "Nilai": v}
            for k, v in params_aktif.items()
            if k not in ["deskripsi"]
        ])
        st.dataframe(tampil_params, use_container_width=True)

    with tab2:
        st.subheader("Dataset Dummy Awal")
        st.write(
            "Dataset dummy dibuat secara sintetis. Setiap baris mewakili satu agen mahasiswa. "
            "Nilai awal diambil dari rentang parameter skenario."
        )

        st.dataframe(df_dummy.round(4), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Baris", df_dummy.shape[0])
        c2.metric("Jumlah Kolom", df_dummy.shape[1])
        c3.metric("Seed", int(seed))

        st.markdown("#### Statistik Dataset Awal")
        st.dataframe(df_dummy.describe().round(4), use_container_width=True)

        st.download_button(
            "Download dataset dummy CSV",
            df_dummy.to_csv(index=False).encode("utf-8"),
            "dataset_dummy_awal.csv",
            "text/csv"
        )

    with tab3:
        st.subheader("Proses Simulasi per Minggu")
        st.write(
            "Bagian ini menunjukkan perubahan rata-rata kondisi agen dari minggu ke minggu."
        )

        df_weekly = (
            df_history
            .groupby("Minggu", as_index=False)
            .agg(
                Avg_PL=("PL", "mean"),
                Avg_SH=("SH", "mean"),
                Avg_SQ=("SQ", "mean"),
                Avg_SR=("SR", "mean"),
                Avg_SleepScore=("SleepScore", "mean"),
                Avg_StresAkademik=("StresAkademik", "mean"),
                Avg_StresTidur=("StresTidur", "mean"),
                Pct_Intervensi=("Intervensi", lambda x: x.mean() * 100),
                Pct_Pulih=("Pulih", lambda x: x.mean() * 100),
            )
        )

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(buat_fig_line(df_weekly, "Avg_PL", "Rata-rata PL"))
        with c2:
            st.pyplot(buat_fig_line(df_weekly, "Avg_SleepScore", "Rata-rata SleepScore"))

        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(buat_fig_line(df_weekly, "Avg_StresAkademik", "Rata-rata Stres Akademik"))
        with c4:
            st.pyplot(buat_fig_line(df_weekly, "Pct_Intervensi", "Intervensi Aktif (%)"))

        st.markdown("#### Tabel Ringkasan Mingguan")
        st.dataframe(df_weekly.round(4), use_container_width=True)

    with tab4:
        st.subheader("Hasil Akhir Agen")
        st.write(
            "Bagian ini menampilkan kondisi akhir seluruh agen setelah simulasi selesai."
        )

        final_view = df_final.copy()
        final_view["Produktivitas"] = final_view["PL"].apply(state_produktivitas)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(buat_fig_hist(final_view["PL"], "Productivity Level Akhir"))
        with c2:
            state_count = final_view["Produktivitas"].value_counts().reset_index()
            state_count.columns = ["State", "Jumlah Agen"]
            st.pyplot(buat_fig_bar(state_count, "State", "Jumlah Agen", "Jumlah Agen"))

        kolom_tampil = [
            "Nama", "PL", "SH", "SQ", "SR", "AS", "D", "HI",
            "SleepScore", "Jumlah_Intervensi", "Jumlah_Minggu_Rendah",
            "Produktivitas"
        ]

        st.dataframe(final_view[kolom_tampil].round(4), use_container_width=True)

        st.download_button(
            "Download hasil akhir agen CSV",
            final_view[kolom_tampil].to_csv(index=False).encode("utf-8"),
            "hasil_akhir_agen.csv",
            "text/csv"
        )

        st.download_button(
            "Download data longitudinal CSV",
            df_history.to_csv(index=False).encode("utf-8"),
            "hasil_longitudinal_simulasi.csv",
            "text/csv"
        )

    with tab5:
        st.subheader("Cara Kerja Model di Dalam Dashboard")

        st.markdown("#### 1. Pembuatan Dataset Dummy")
        st.code(
            """Setiap agen dibuat dari distribusi uniform:
PL_awal ∼ U(PL_min, PL_max)
SH_awal ∼ U(SH_min, SH_max)
SQ_awal ∼ U(SQ_min, SQ_max)
SR_awal ∼ U(SR_min, SR_max)
AS      ∼ U(AS_min, AS_max)
D       ∼ U(D_min, D_max)
HI      ∼ U(HI_min, HI_max)""",
            language="text"
        )

        st.markdown("#### 2. Perhitungan SleepScore")
        st.latex(r"SleepScore = 1 - \min\left(\frac{|SH - 7.5|}{4.5}, 1\right)")

        st.markdown("#### 3. Perhitungan Stressor")
        st.latex(r"StresTidur = (1 - SleepScore) \times (1 - SR)")
        st.latex(r"DampakStres = 0.25 \times (AS + D \times StresTidur)")

        st.markdown("#### 4. Perhitungan Manfaat Tidur")
        st.latex(r"ManfaatTidur = 0.28(SleepScore) + 0.22(SQ) + 0.15(SR)")

        st.markdown("#### 5. Perubahan Produktivitas")
        st.latex(r"\Delta = 0.22 \times (ManfaatTidur - DampakStres)")

        st.markdown("#### 6. Aturan Intervensi")
        st.code(
            """Tanpa Intervensi:
    intervensi tidak aktif

Intervensi Reaktif:
    intervensi aktif jika PL < 0.55

Intervensi Preventif:
    intervensi aktif setiap minggu""",
            language="text"
        )

        st.markdown("#### 7. Dampak Intervensi")
        st.write(
            "Jika intervensi aktif, dashboard memperbaiki PL, mendekatkan SH ke jam tidur ideal, "
            "meningkatkan SQ, dan meningkatkan SR."
        )

else:

    tab1, tab2, tab3 = st.tabs([
        "Perbandingan Skenario",
        "Monte Carlo",
        "Parameter Skenario"
    ])

    with tab1:
        st.subheader("Perbandingan Semua Skenario")

        c1, c2, c3, c4 = st.columns(4)

        best = df_summary.loc[df_summary["Avg_PL"].idxmax()]
        worst = df_summary.loc[df_summary["Avg_PL"].idxmin()]

        c1.metric("Iterasi per Skenario", int(n_iterasi))
        c2.metric("Skenario Terbaik", str(best["Skenario"]))
        c3.metric("Avg PL Tertinggi", format_float(best["Avg_PL"]))
        c4.metric("Avg PL Terendah", format_float(worst["Avg_PL"]))

        st.markdown("#### Tabel Ringkasan")
        st.dataframe(df_summary.round(4), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(buat_fig_bar(df_summary, "Skenario", "Avg_PL", "Rata-rata PL"))
        with c2:
            st.pyplot(buat_fig_bar(df_summary, "Skenario", "Pct_Produktivitas_Rendah", "Produktivitas Rendah (%)"))

        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(buat_fig_bar(df_summary, "Skenario", "Avg_Intervensi", "Rata-rata Intervensi"))
        with c4:
            st.pyplot(buat_fig_bar(df_summary, "Skenario", "Avg_SleepScore", "Rata-rata SleepScore"))

        st.download_button(
            "Download ringkasan skenario CSV",
            df_summary.to_csv(index=False).encode("utf-8"),
            "ringkasan_skenario_simulasi_tidur.csv",
            "text/csv"
        )

    with tab2:
        st.subheader("Data Monte Carlo")
        st.write(
            "Setiap skenario dijalankan berulang kali untuk mengurangi ketergantungan pada satu seed simulasi."
        )

        st.dataframe(df_iterasi.head(200).round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        data_box = [
            df_iterasi.loc[df_iterasi["Skenario"] == nama, "Avg_PL"].values
            for nama in SKENARIO_ALL.keys()
        ]
        ax.boxplot(data_box, labels=list(SKENARIO_ALL.keys()), showmeans=True)
        ax.set_ylabel("Avg PL")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        st.pyplot(fig)

        st.download_button(
            "Download hasil iterasi CSV",
            df_iterasi.to_csv(index=False).encode("utf-8"),
            "hasil_iterasi_simulasi_tidur.csv",
            "text/csv"
        )

    with tab3:
        st.subheader("Parameter Semua Skenario")

        rows = []
        for nama, params in SKENARIO_ALL.items():
            row = {"Skenario": nama}
            row.update({
                k: v for k, v in params.items()
                if k not in ["deskripsi"]
            })
            rows.append(row)

        df_params = pd.DataFrame(rows)
        st.dataframe(df_params, use_container_width=True)

        st.download_button(
            "Download parameter skenario CSV",
            df_params.to_csv(index=False).encode("utf-8"),
            "parameter_skenario_simulasi_tidur.csv",
            "text/csv"
        )


st.divider()
st.caption(
    "Catatan: Model ini bersifat simulatif. Parameter belum dikalibrasi dengan data survei klinis. "
    "Dashboard digunakan untuk eksperimen pemodelan, bukan diagnosis kesehatan."
)
