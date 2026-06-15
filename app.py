import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Prediksi Performa Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - TAMPILAN PROFESIONAL
# ============================================================
st.markdown("""
<style>
    /* Font & warna dasar */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0;
    }

    /* Kartu metrik */
    .metric-card {
        background: white;
        border: 1px solid #e8edf2;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .metric-card .label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }

    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a3a5c;
        line-height: 1;
    }

    .metric-card .sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* Badge hasil prediksi */
    .result-high {
        background: #dcfce7;
        border: 1.5px solid #16a34a;
        color: #15803d;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .result-medium {
        background: #fef9c3;
        border: 1.5px solid #ca8a04;
        color: #92400e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .result-low {
        background: #fee2e2;
        border: 1.5px solid #dc2626;
        color: #991b1b;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Section header */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a3a5c;
        border-left: 4px solid #2d6a9f;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Tabel rekomendasi */
    .rec-table {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border: 1px solid #e2e8f0;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #1e40af;
        font-size: 0.9rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f1f5f9;
    }

    /* Tombol utama */
    .stButton > button[kind="primary"] {
        background: #2d6a9f;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        letter-spacing: 0.2px;
    }

    /* Hide watermark */
    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "Artifak Model Student Performance"

@st.cache_resource
def load_artifacts():
    model    = pickle.load(open(MODEL_DIR / "model_student_performance.pkl",  "rb"))
    features = pickle.load(open(MODEL_DIR / "feature_columns.pkl", "rb"))
    encoder  = pickle.load(open(MODEL_DIR / "label_encoder.pkl",   "rb"))
    if not isinstance(features, list):
        features = list(features)
    return model, features, encoder

try:
    model, features, encoder = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR   = str(e)

# ============================================================
# KONSTANTA INPUT
# ============================================================
GENDER_OPT        = ["Male", "Female"]
YESNO_OPT         = ["Yes", "No"]
LOW_MED_HIGH      = ["Low", "Medium", "High"]
SCHOOL_TYPE_OPT   = ["Public", "Private"]
PARENT_EDU_OPT    = ["High School", "College", "Postgraduate"]
PEER_OPT          = ["Positive", "Neutral", "Negative"]
DISTANCE_OPT      = ["Near", "Moderate", "Far"]

# ============================================================
# HEADER UTAMA
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎓 Sistem Prediksi Performa Akademik Siswa</h1>
    <p>Berbasis Machine Learning · Metode CRISP-DM · Analisis Faktor Akademik & Non-Akademik</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - NAVIGASI
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=72)
    st.markdown("### Menu Utama")

    menu = st.radio(
        label="Navigasi",
        options=[
            "🏠  Beranda",
            "🔮  Prediksi Performa",
            "📊  Analisis Fitur",
            "📈  Simulasi Skenario",
            "📋  Panduan & Metodologi"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    if MODEL_LOADED:
        st.success(f"Model aktif: **{type(model).__name__}**")
        st.caption(f"Jumlah fitur: {len(features)}")
        st.caption(f"Kelas target: {', '.join(encoder.classes_)}")
    else:
        st.error("Model gagal dimuat")
        st.code(LOAD_ERROR, language="text")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_input_dataframe(raw: dict) -> pd.DataFrame:
    """Konversi dict input mentah → DataFrame dengan OHE, sesuai kolom model."""
    df = pd.DataFrame([raw])
    df_enc = pd.get_dummies(df)
    df_enc = df_enc.reindex(columns=features, fill_value=0)
    return df_enc

def predict(df_enc: pd.DataFrame):
    """Jalankan prediksi, kembalikan label string."""
    pred  = model.predict(df_enc)[0]
    label = encoder.inverse_transform([int(pred)])[0]
    return label

def get_prediction_proba(df_enc: pd.DataFrame, fallback_label: str) -> dict:
    """Ambil probabilitas asli model Logistic Regression.

    Logistic Regression memiliki method predict_proba(), sehingga grafik
    probabilitas tidak perlu disimulasikan lagi. Jika suatu saat model
    diganti ke model yang tidak mendukung probabilitas, fungsi ini tetap
    aman dengan fallback sederhana.
    """
    class_labels = [str(label) for label in encoder.classes_]
    preferred_order = ["High", "Medium", "Low"]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_enc)[0]
        raw_proba = {str(label): float(prob) for label, prob in zip(class_labels, probs)}
    else:
        # Fallback keamanan jika model tidak memiliki predict_proba()
        raw_proba = {str(label): 0.0 for label in class_labels}
        raw_proba[str(fallback_label)] = 1.0

    # Urutkan agar visualisasi selalu konsisten: High → Medium → Low
    ordered = {label: raw_proba[label] for label in preferred_order if label in raw_proba}
    ordered.update({label: raw_proba[label] for label in raw_proba if label not in ordered})
    return ordered

def risk_score(label: str) -> int:
    return {"High": 15, "Medium": 45, "Low": 80}.get(label, 50)

def recommendation(label: str) -> list:
    base = [
        ("Kehadiran", "Pertahankan kehadiran ≥ 85% setiap bulan."),
        ("Istirahat",  "Tidur 7–9 jam/malam untuk konsolidasi memori."),
    ]
    if label == "Low":
        return base + [
            ("Bimbingan Belajar", "Ikuti sesi tutoring minimal 2×/minggu."),
            ("Motivasi",          "Konsultasi dengan guru BK untuk pemetaan minat."),
            ("Jam Belajar",       "Tingkatkan jam belajar mandiri ke 20–25 jam/minggu."),
        ]
    elif label == "Medium":
        return base + [
            ("Konsistensi",  "Jadwalkan ulang jadwal belajar agar lebih terstruktur."),
            ("Pemanfaatan", "Optimalkan akses internet untuk sumber belajar digital."),
        ]
    else:
        return base + [
            ("Pengembangan", "Ikuti kompetisi atau olympiade sebagai tantangan lebih."),
            ("Mentoring",    "Pertimbangkan menjadi tutor sebaya untuk memperkuat pemahaman."),
        ]

def gauge_chart(value: int, label: str):
    color = "#16a34a" if value < 30 else "#ca8a04" if value < 65 else "#dc2626"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": label, "font": {"size": 14}},
        gauge = {
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,  30], "color": "#dcfce7"},
                {"range": [30, 65], "color": "#fef9c3"},
                {"range": [65, 100],"color": "#fee2e2"},
            ],
            "threshold": {
                "line":  {"color": "#1a3a5c", "width": 3},
                "thickness": 0.8,
                "value": value
            }
        },
        number={"suffix": "/100"}
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ============================================================
# HALAMAN: BERANDA
# ============================================================
if menu == "🏠  Beranda":

    if not MODEL_LOADED:
        st.error(f"❌ Gagal memuat model: {LOAD_ERROR}")
        st.stop()

    # Metrik ringkasan
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Algoritma Model",   type(model).__name__, "Dari file pickle"),
        ("Jumlah Fitur",      str(len(features)),   "Input variabel"),
        ("Kelas Target",      str(len(encoder.classes_)), "High / Medium / Low"),
        ("Status Sistem",     "Aktif",              "Siap digunakan"),
    ]
    for col, (lbl, val, sub) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{lbl}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Dua kolom info sistem
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Tentang Sistem</div>', unsafe_allow_html=True)
        st.write("""
        Sistem ini memprediksi performa akademik siswa berdasarkan kombinasi
        faktor perilaku belajar, kondisi lingkungan, dan data akademik historis.
        Prediksi dikelompokkan ke dalam tiga kategori: **High**, **Medium**, dan **Low**.
        """)
        st.markdown("""
        **Faktor yang dianalisis:**
        - Perilaku belajar: jam belajar, kehadiran, sesi tutoring
        - Kesehatan & gaya hidup: jam tidur, aktivitas fisik
        - Lingkungan: dukungan orang tua, akses internet, pengaruh teman
        - Latar belakang: jenis sekolah, pendidikan orang tua, jarak rumah
        """)

    with col_b:
        st.markdown('<div class="section-title">Fitur Sistem</div>', unsafe_allow_html=True)
        fitur_list = {
            "🔮 Prediksi Performa":     "Input data siswa dan dapatkan prediksi kategori performa.",
            "📊 Analisis Fitur":         "Visualisasi kontribusi setiap variabel terhadap model.",
            "📈 Simulasi Skenario":      "Bandingkan dua skenario input untuk melihat perbedaan prediksi.",
            "📋 Panduan & Metodologi":   "Dokumentasi alur CRISP-DM dan cara penggunaan sistem.",
        }
        for judul, deskripsi in fitur_list.items():
            st.markdown(f"**{judul}** — {deskripsi}")

    st.markdown("---")

    # Distribusi kelas encoder sebagai informasi
    st.markdown('<div class="section-title">Kategori Output Prediksi</div>', unsafe_allow_html=True)
    cols_cls = st.columns(3)
    class_info = {
        "High":   ("🟢", "#dcfce7", "#15803d", "Siswa berprestasi tinggi. Faktor akademik dan non-akademik mendukung secara optimal."),
        "Medium": ("🟡", "#fef9c3", "#92400e", "Performa cukup. Masih ada ruang peningkatan pada beberapa faktor kunci."),
        "Low":    ("🔴", "#fee2e2", "#991b1b", "Performa rendah. Membutuhkan intervensi dan pendampingan lebih intensif."),
    }
    for col, (kls, (icon, bg, clr, desc)) in zip(cols_cls, class_info.items()):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border-radius:10px;padding:1rem 1.2rem;border:1px solid {clr}20;">
                <div style="font-size:1.5rem;margin-bottom:0.3rem;">{icon} <strong style="color:{clr}">{kls}</strong></div>
                <div style="font-size:0.85rem;color:#374151;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# HALAMAN: PREDIKSI PERFORMA
# ============================================================
elif menu == "🔮  Prediksi Performa":

    if not MODEL_LOADED:
        st.error("Model belum dimuat. Periksa file di folder model.")
        st.stop()

    st.markdown("## 🔮 Input Data Siswa")
    st.caption("Lengkapi seluruh parameter berikut untuk menghasilkan prediksi yang akurat.")

    with st.form("form_prediksi"):

        tab1, tab2, tab3 = st.tabs(["📊 Data Akademik & Perilaku", "👥 Latar Belakang Sosial", "🏫 Konteks Sekolah & Lingkungan"])

        # ── Tab 1: Akademik & Perilaku ──────────────────────────────
        with tab1:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown('<div class="section-title">Perilaku Belajar</div>', unsafe_allow_html=True)
                hours_studied      = st.slider("Jam Belajar per Minggu", 0, 50, 20,
                                               help="Total jam belajar mandiri dalam satu minggu.")
                attendance         = st.slider("Tingkat Kehadiran (%)", 0, 100, 85,
                                               help="Persentase kehadiran siswa di sekolah.")
                tutoring_sessions  = st.slider("Sesi Bimbingan Belajar per Bulan", 0, 10, 2,
                                               help="Jumlah sesi les/tutoring per bulan.")
                extracurricular    = st.selectbox("Kegiatan Ekstrakurikuler", YESNO_OPT,
                                                  help="Apakah siswa aktif di ekstrakurikuler?")

            with c2:
                st.markdown('<div class="section-title">Kesehatan & Gaya Hidup</div>', unsafe_allow_html=True)
                sleep_hours        = st.slider("Jam Tidur per Hari", 3, 12, 7,
                                               help="Rata-rata jam tidur malam per hari.")
                physical_activity  = st.slider("Aktivitas Fisik per Minggu (jam)", 0, 20, 4,
                                               help="Jumlah jam olahraga/aktivitas fisik dalam seminggu.")
                prev_scores        = st.slider("Nilai Ujian Sebelumnya (0–100)", 0, 100, 70,
                                               help="Rata-rata nilai ujian semester sebelumnya.")

        # ── Tab 2: Latar Belakang Sosial ────────────────────────────
        with tab2:
            c3, c4 = st.columns(2)

            with c3:
                st.markdown('<div class="section-title">Data Demografis</div>', unsafe_allow_html=True)
                gender             = st.selectbox("Jenis Kelamin", GENDER_OPT)
                parental_education = st.selectbox("Pendidikan Orang Tua", PARENT_EDU_OPT,
                                                  help="Tingkat pendidikan terakhir orang tua/wali.")
                family_income      = st.selectbox("Tingkat Pendapatan Keluarga", LOW_MED_HIGH)
                learning_disab     = st.selectbox("Memiliki Kesulitan Belajar (Learning Disability)", YESNO_OPT)

            with c4:
                st.markdown('<div class="section-title">Dukungan & Motivasi</div>', unsafe_allow_html=True)
                parental_invol     = st.selectbox("Keterlibatan Orang Tua dalam Pendidikan", LOW_MED_HIGH)
                motivation_level   = st.selectbox("Tingkat Motivasi Belajar", LOW_MED_HIGH)
                peer_influence     = st.selectbox("Pengaruh Teman Sebaya", PEER_OPT,
                                                  help="Seberapa positif lingkungan pertemanan siswa.")
                internet_access    = st.selectbox("Akses Internet di Rumah", YESNO_OPT)

        # ── Tab 3: Konteks Sekolah & Lingkungan ─────────────────────
        with tab3:
            c5, c6 = st.columns(2)

            with c5:
                st.markdown('<div class="section-title">Profil Sekolah</div>', unsafe_allow_html=True)
                school_type        = st.selectbox("Jenis Sekolah", SCHOOL_TYPE_OPT)
                teacher_quality    = st.selectbox("Kualitas Pengajaran Guru", LOW_MED_HIGH,
                                                  help="Penilaian umum atas kualitas pengajaran di sekolah.")
                access_resources   = st.selectbox("Akses Sumber Belajar di Sekolah", LOW_MED_HIGH,
                                                  help="Ketersediaan buku, laboratorium, komputer, dll.")

            with c6:
                st.markdown('<div class="section-title">Kondisi Geografis</div>', unsafe_allow_html=True)
                distance_home      = st.selectbox("Jarak Rumah ke Sekolah", DISTANCE_OPT,
                                                  help="Estimasi jarak/waktu tempuh ke sekolah.")

        st.markdown("---")
        submitted = st.form_submit_button("🔮 Jalankan Prediksi", use_container_width=True, type="primary")

    # ── Proses Prediksi ─────────────────────────────────────────────
    if submitted:
        raw = {
            "Hours_Studied":              hours_studied,
            "Attendance":                 attendance,
            "Sleep_Hours":                sleep_hours,
            "Previous_Scores":            prev_scores,
            "Tutoring_Sessions":          tutoring_sessions,
            "Physical_Activity":          physical_activity,
            "Gender":                     gender,
            "Parental_Involvement":       parental_invol,
            "Internet_Access":            internet_access,
            "Extracurricular_Activities": extracurricular,
            "Motivation_Level":           motivation_level,
            "School_Type":                school_type,
            "Access_to_Resources":        access_resources,
            "Family_Income":              family_income,
            "Teacher_Quality":            teacher_quality,
            "Peer_Influence":             peer_influence,
            "Learning_Disabilities":      learning_disab,
            "Parental_Education_Level":   parental_education,
            "Distance_from_Home":         distance_home,
        }

        df_enc = build_input_dataframe(raw)
        hasil  = predict(df_enc)
        risk   = risk_score(hasil)
        proba  = get_prediction_proba(df_enc, hasil)
        recs   = recommendation(hasil)

        st.markdown("---")
        st.markdown("## 📋 Hasil Prediksi")

        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

        with res_col1:
            css_cls = f"result-{hasil.lower()}"
            icon_map = {"High": "🟢 Tinggi (High)", "Medium": "🟡 Sedang (Medium)", "Low": "🔴 Rendah (Low)"}
            st.markdown(f'<div class="{css_cls}">Kategori Performa: {icon_map[hasil]}</div>', unsafe_allow_html=True)
            st.caption("Prediksi dihasilkan oleh model machine learning berdasarkan kombinasi seluruh input.")

        with res_col2:
            st.plotly_chart(gauge_chart(risk, "Indeks Risiko Akademik"), use_container_width=True)

        with res_col3:
            # Donut chart distribusi probabilitas
            fig_donut = go.Figure(data=[go.Pie(
                labels=list(proba.keys()),
                values=list(proba.values()),
                hole=0.5,
                marker_colors=["#16a34a", "#ca8a04", "#dc2626"]
            )])
            fig_donut.update_layout(
                title={"text": "Distribusi Probabilitas Kelas", "font": {"size": 13}},
                height=220,
                margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.15)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Bar probabilitas horizontal
        st.markdown('<div class="section-title">Probabilitas per Kategori</div>', unsafe_allow_html=True)
        df_prob = pd.DataFrame({
            "Kategori": list(proba.keys()),
            "Probabilitas (%)": [round(v * 100, 1) for v in proba.values()]
        })
        fig_bar = px.bar(
            df_prob, x="Probabilitas (%)", y="Kategori", orientation="h",
            color="Kategori",
            color_discrete_map={"High": "#16a34a", "Medium": "#ca8a04", "Low": "#dc2626"},
            text="Probabilitas (%)"
        )
        fig_bar.update_layout(height=200, margin=dict(t=20, b=20, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Rekomendasi
        st.markdown('<div class="section-title">Rekomendasi Tindak Lanjut</div>', unsafe_allow_html=True)
        for i, (aspek, saran) in enumerate(recs, 1):
            st.markdown(f"**{i}. {aspek}** — {saran}")

        # Ringkasan input
        with st.expander("📄 Lihat Ringkasan Data Input"):
            df_display = pd.DataFrame(raw.items(), columns=["Parameter", "Nilai"])
            st.dataframe(df_display, use_container_width=True, hide_index=True)

# ============================================================
# HALAMAN: ANALISIS FITUR
# ============================================================
elif menu == "📊  Analisis Fitur":

    if not MODEL_LOADED:
        st.error("Model belum dimuat.")
        st.stop()

    st.markdown("## 📊 Analisis Kepentingan Fitur")
    st.caption("Visualisasi seberapa besar kontribusi masing-masing variabel terhadap keputusan model.")

    # Logistic Regression tidak memiliki feature_importances_.
    # Untuk Logistic Regression, kontribusi fitur dibaca dari koefisien model.
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        metode_importance = "Feature Importance bawaan model"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            # Multiclass Logistic Regression: ambil rata-rata nilai absolut koefisien setiap fitur
            importances = np.mean(np.abs(coef), axis=0)
        metode_importance = "Rata-rata nilai absolut koefisien Logistic Regression"
    else:
        st.warning("Model ini belum menyediakan atribut untuk analisis fitur.")
        st.stop()

    if importances.sum() > 0:
        importances = importances / importances.sum()

    df_imp = pd.DataFrame({
        "Fitur":        features,
        "Kepentingan":  importances
    }).sort_values("Kepentingan", ascending=False)

    st.info(f"Metode analisis fitur: {metode_importance}. Nilai ditampilkan dalam bentuk proporsi agar mudah dibandingkan.")

    # Top-10 fitur
    top10 = df_imp.head(10).copy().sort_values("Kepentingan", ascending=True)

    col_chart, col_tabel = st.columns([3, 2])

    with col_chart:
        st.markdown('<div class="section-title">10 Fitur Paling Berpengaruh</div>', unsafe_allow_html=True)
        fig_imp = px.bar(
            top10, x="Kepentingan", y="Fitur", orientation="h",
            color="Kepentingan",
            color_continuous_scale=["#bfdbfe", "#2d6a9f"],
            text=top10["Kepentingan"].map(lambda x: f"{x:.3f}")
        )
        fig_imp.update_coloraxes(showscale=False)
        fig_imp.update_layout(height=420, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_tabel:
        st.markdown('<div class="section-title">Tabel Kepentingan Fitur</div>', unsafe_allow_html=True)
        df_show = df_imp.reset_index(drop=True)
        df_show.index = df_show.index + 1
        df_show["Kepentingan (%)"] = (df_show["Kepentingan"] * 100).round(2)
        st.dataframe(
            df_show[["Fitur", "Kepentingan (%)"]],
            use_container_width=True,
            height=400
        )

    st.markdown("---")

    # Pie distribusi kepentingan (grouped: top5 vs lainnya)
    st.markdown('<div class="section-title">Distribusi Kepentingan: Top 5 vs Lainnya</div>', unsafe_allow_html=True)
    top5_sum   = df_imp["Kepentingan"].head(5).sum()
    other_sum  = df_imp["Kepentingan"].iloc[5:].sum()
    top5_names = ", ".join(df_imp["Fitur"].head(5).tolist())

    fig_pie = go.Figure(data=[go.Pie(
        labels=[f"Top 5 Fitur\n({top5_names[:40]}...)", "Fitur Lainnya"],
        values=[top5_sum, other_sum],
        hole=0.4,
        marker_colors=["#2d6a9f", "#e2e8f0"]
    )])
    fig_pie.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
    ℹ️ Lima fitur teratas (<strong>{top5_names[:60]}...</strong>) secara kolektif berkontribusi
    sebesar <strong>{top5_sum*100:.1f}%</strong> dari total kepentingan model.
    Fokus intervensi sebaiknya diarahkan pada fitur-fitur tersebut.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HALAMAN: SIMULASI SKENARIO
# ============================================================
elif menu == "📈  Simulasi Skenario":

    if not MODEL_LOADED:
        st.error("Model belum dimuat.")
        st.stop()

    st.markdown("## 📈 Simulasi & Perbandingan Skenario")
    st.caption("Masukkan dua kondisi siswa yang berbeda untuk membandingkan hasil prediksi secara berdampingan.")

    def input_skenario(prefix: str, label: str, col):
        with col:
            st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)
            hrs   = st.slider(f"Jam Belajar/Minggu [{prefix}]",  0, 50, 15 if prefix=="A" else 30)
            att   = st.slider(f"Kehadiran (%) [{prefix}]",        0, 100, 70 if prefix=="A" else 95)
            slp   = st.slider(f"Jam Tidur/Hari [{prefix}]",       3, 12,  5  if prefix=="A" else 8)
            prev  = st.slider(f"Nilai Ujian Sebelumnya [{prefix}]", 0, 100, 55 if prefix=="A" else 85)
            tut   = st.slider(f"Sesi Tutoring/Bulan [{prefix}]",  0, 10,  0  if prefix=="A" else 3)
            phys  = st.slider(f"Aktivitas Fisik/Minggu (jam) [{prefix}]", 0, 20, 2 if prefix=="A" else 6)

            mot   = st.selectbox(f"Motivasi [{prefix}]",          LOW_MED_HIGH, index=0 if prefix=="A" else 2)
            par   = st.selectbox(f"Keterlibatan Orang Tua [{prefix}]", LOW_MED_HIGH, index=0 if prefix=="A" else 2)
            inet  = st.selectbox(f"Akses Internet [{prefix}]",     YESNO_OPT,    index=1 if prefix=="A" else 0)
            ekskul= st.selectbox(f"Ekstrakurikuler [{prefix}]",    YESNO_OPT,    index=1 if prefix=="A" else 0)
            peer  = st.selectbox(f"Pengaruh Teman [{prefix}]",     PEER_OPT,     index=2 if prefix=="A" else 0)
            gen   = st.selectbox(f"Jenis Kelamin [{prefix}]",      GENDER_OPT)

            return {
                "Hours_Studied": hrs, "Attendance": att, "Sleep_Hours": slp,
                "Previous_Scores": prev, "Tutoring_Sessions": tut, "Physical_Activity": phys,
                "Motivation_Level": mot, "Parental_Involvement": par,
                "Internet_Access": inet, "Extracurricular_Activities": ekskul,
                "Peer_Influence": peer, "Gender": gen,
                "School_Type": "Public", "Access_to_Resources": "Medium",
                "Family_Income": "Medium", "Teacher_Quality": "Medium",
                "Learning_Disabilities": "No", "Parental_Education_Level": "High School",
                "Distance_from_Home": "Near",
            }

    colS1, colS2 = st.columns(2)
    raw_a = input_skenario("A", "Skenario A", colS1)
    raw_b = input_skenario("B", "Skenario B", colS2)

    if st.button("⚖️  Bandingkan Kedua Skenario", use_container_width=True, type="primary"):

        df_enc_a = build_input_dataframe(raw_a)
        df_enc_b = build_input_dataframe(raw_b)
        hasil_a  = predict(df_enc_a)
        hasil_b  = predict(df_enc_b)
        proba_a  = get_prediction_proba(df_enc_a, hasil_a)
        proba_b  = get_prediction_proba(df_enc_b, hasil_b)
        risk_a   = risk_score(hasil_a)
        risk_b   = risk_score(hasil_b)

        st.markdown("---")
        st.markdown("### 📊 Perbandingan Hasil Prediksi")

        m1, m2 = st.columns(2)
        icon_map = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}

        with m1:
            st.metric("Skenario A", f"{icon_map[hasil_a]} {hasil_a}", delta=f"Risiko: {risk_a}/100")
        with m2:
            st.metric("Skenario B", f"{icon_map[hasil_b]} {hasil_b}", delta=f"Risiko: {risk_b}/100")

        # Gauge berdampingan
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(gauge_chart(risk_a, "Risiko Akademik — Skenario A"), use_container_width=True)
        with g2:
            st.plotly_chart(gauge_chart(risk_b, "Risiko Akademik — Skenario B"), use_container_width=True)

        # Grouped bar probabilitas
        st.markdown('<div class="section-title">Perbandingan Distribusi Probabilitas</div>', unsafe_allow_html=True)
        classes = ["High", "Medium", "Low"]
        fig_comp = go.Figure(data=[
            go.Bar(name="Skenario A", x=classes, y=[proba_a[k]*100 for k in classes],
                   marker_color="#2d6a9f", text=[f"{proba_a[k]*100:.0f}%" for k in classes], textposition="outside"),
            go.Bar(name="Skenario B", x=classes, y=[proba_b[k]*100 for k in classes],
                   marker_color="#16a34a", text=[f"{proba_b[k]*100:.0f}%" for k in classes], textposition="outside"),
        ])
        fig_comp.update_layout(
            barmode="group", height=320,
            yaxis_title="Probabilitas (%)",
            margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Delta perubahan fitur kunci
        st.markdown('<div class="section-title">Perbedaan Nilai Input Kunci</div>', unsafe_allow_html=True)
        numerik_keys = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]
        delta_rows = []
        for k in numerik_keys:
            delta_rows.append({
                "Variabel": k.replace("_", " "),
                "Skenario A": raw_a[k],
                "Skenario B": raw_b[k],
                "Selisih (B–A)": raw_b[k] - raw_a[k]
            })
        df_delta = pd.DataFrame(delta_rows)
        st.dataframe(df_delta, use_container_width=True, hide_index=True)

# ============================================================
# HALAMAN: PANDUAN & METODOLOGI
# ============================================================
elif menu == "📋  Panduan & Metodologi":

    st.markdown("## 📋 Panduan Penggunaan & Metodologi")

    tab_p, tab_m = st.tabs(["📖 Panduan Pengguna", "🔬 Metodologi CRISP-DM"])

    with tab_p:
        st.markdown("""
        ### Cara Menggunakan Sistem

        **1. Prediksi Performa**
        Buka menu *Prediksi Performa* dan isi seluruh parameter pada tiga tab yang tersedia:
        - **Data Akademik & Perilaku**: jam belajar, kehadiran, sesi tutoring, aktivitas fisik, dll.
        - **Latar Belakang Sosial**: data demografis, dukungan keluarga, motivasi.
        - **Konteks Sekolah**: jenis sekolah, kualitas guru, akses sumber belajar.

        Setelah mengisi semua field, klik tombol **Jalankan Prediksi** untuk mendapatkan hasil.

        **2. Analisis Fitur**
        Gunakan menu *Analisis Fitur* untuk memahami variabel mana yang paling memengaruhi
        prediksi model. Visualisasi ini membantu guru dan konselor menentukan fokus intervensi.

        **3. Simulasi Skenario**
        Menu *Simulasi Skenario* memungkinkan perbandingan dua kondisi siswa secara berdampingan.
        Berguna untuk mengevaluasi "bagaimana jika" (*what-if analysis*) perubahan faktor tertentu.

        ---

        ### Interpretasi Output

        | Kategori | Deskripsi | Tindakan yang Disarankan |
        |----------|-----------|--------------------------|
        | **High** | Performa akademik baik | Pertahankan pola belajar; pertimbangkan program pengayaan |
        | **Medium** | Performa cukup | Identifikasi faktor penghambat; bimbingan terarah |
        | **Low** | Performa rendah | Intervensi intensif; libatkan orang tua dan konselor BK |

        ---

        ### Catatan Penting
        - Prediksi bersifat probabilistik dan bukan diagnosis absolut.
        - Sistem ini merupakan alat bantu pengambilan keputusan, bukan pengganti penilaian guru/konselor.
        - Akurasi prediksi bergantung pada kualitas dan kelengkapan data input.
        """)

    with tab_m:
        st.markdown("""
        ### Metodologi CRISP-DM

        Pengembangan model prediksi ini mengikuti kerangka kerja **CRISP-DM**
        (*Cross-Industry Standard Process for Data Mining*), yang terdiri dari enam fase:
        """)

        fases = [
            ("1. Business Understanding",
             "Tujuan: membangun sistem yang dapat mengidentifikasi siswa berisiko rendah "
             "secara dini agar intervensi dapat dilakukan sebelum nilai akhir menurun."),
            ("2. Data Understanding",
             "Dataset mencakup variabel akademik (jam belajar, kehadiran, nilai ujian) dan "
             "non-akademik (kondisi keluarga, akses teknologi, motivasi, gaya hidup)."),
            ("3. Data Preparation",
             "Variabel kategorikal dikonversi dengan One-Hot Encoding (OHE). Nilai hilang "
             "diimputasi, dan data diseimbangkan sebelum pelatihan."),
            ("4. Modeling",
             "Model dilatih menggunakan algoritma Logistic Regression. Pemilihan model "
             "didasarkan pada perbandingan beberapa algoritma melalui LazyPredict dan "
             "evaluasi skenario pembagian data train-test."),
            ("5. Evaluation",
             "Model dievaluasi menggunakan data uji terpisah. Metrik yang digunakan: "
             "akurasi, precision, recall, dan F1-score per kelas."),
            ("6. Deployment",
             "Model yang telah terlatih disimpan sebagai file pickle dan diintegrasikan "
             "ke dalam antarmuka Streamlit ini untuk digunakan oleh pengguna akhir."),
        ]

        for judul, isi in fases:
            with st.expander(judul):
                st.write(isi)

        st.markdown("""
        ---
        ### File Artefak Model

        | File | Keterangan |
        |------|-----------|
        | `model_student_performance.pkl` | Model machine learning terlatih |
        | `feature_columns.pkl` | Daftar nama kolom fitur setelah OHE |
        | `label_encoder.pkl` | Encoder untuk mengonversi angka ke label kelas |

        Semua file disimpan di folder `Artifak Model Student Performance/`
        dan dimuat secara otomatis saat aplikasi dijalankan.
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Sistem Prediksi Performa Akademik Siswa · Dikembangkan dengan Streamlit & Scikit-learn")
