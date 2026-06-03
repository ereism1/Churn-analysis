import streamlit as st
import joblib
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt




modelo = joblib.load("models/modelo_churn.pkl")
scaler = joblib.load("models/scaler.pkl")
colunas = joblib.load("models/colunas.pkl")

df = pd.read_excel(
    "data/Telco_customer_churn.xlsx"
)

st.title("Predição de Churn")

tenure = st.number_input(
    "Tempo de permanência (meses)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Mensalidade",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Valor total gasto",
    min_value=0.0,
    value=500.0
)

contract = st.selectbox(
    "Tipo de contrato",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

dependents = st.selectbox(
    "Possui dependentes?",
    ["No", "Yes"]
)

tech_support = st.selectbox(
    "Possui suporte técnico?",
    ["No", "Yes"]
)

online_security = st.selectbox(
    "Possui segurança online?",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Método de pagamento",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

if st.button("Prever Churn"):

    dados = pd.DataFrame(
        0,
        index=[0],
        columns=colunas
    )

    # preencher dados...

    dados_escalados = scaler.transform(dados)

    probabilidade = modelo.predict_proba(
        dados_escalados
    )[0][1]

    prob = probabilidade * 100

    st.subheader("Resultado")

    st.metric(
        "Probabilidade de Churn",
        f"{prob:.2f}%"
    )

    st.progress(prob / 100)

    if prob >= 70:
        st.error("⚠️ Alto risco de cancelamento")

    elif prob >= 40:
        st.warning("⚡ Risco moderado")

    else:
        st.success("✅ Baixo risco")



st.header("Fatores de Churn")

top_fatores = pd.DataFrame({
    "Variável":[
        "Total Charges",
        "Fiber Optic",
        "Paperless Billing",
        "Partner"
    ],
    "Impacto":[
        0.72,
        0.27,
        0.19,
        0.15
    ]
})

st.bar_chart(
    top_fatores.set_index("Variável")
)