import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def show_classification_report_table(y_true, y_pred):
    """Exibe o classification_report do sklearn como tabela colorida no Streamlit."""
    import pandas as pd
    from sklearn.metrics import classification_report

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    if 'accuracy' in df_report.index:
        df_report = df_report.drop('accuracy')
    styled = df_report.style.background_gradient(
        subset=['precision', 'recall', 'f1-score'],
        cmap='RdYlGn'
    ).format("{:.2f}")
    st.write("### Relatório de Classificação")
    st.dataframe(styled, use_container_width=True)

def show_feature_importance_rf():
    st.subheader("Feature Importance - Random Forest")
    st.write("Importância das variáveis segundo o modelo Random Forest.")
    st.image(r"data/Feature_importance-RandomForest.png", use_container_width=True)

def show_force_plot_rf():
    st.subheader("Force Plot - Random Forest")
    st.image(r"data/Force plot-Random Forest.png", use_container_width=True)

def show_shap_summary_rf():
    st.subheader("SHAP Summary Plot - Random Forest")
    st.image(r"data/Shap-RandomForest.png", use_container_width=True)

def run_classification():
    st.title("Resultados dos Modelos de Regressão")

    st.sidebar.subheader("Tópicos de Classificação")
    topics = [
        "Resultados dos Modelos",
        "Feature Importance Random Forest",
        "Force Plot Random Forest",
        "Shap Random Forest"
    ]
    selected_topic = st.sidebar.radio("Escolha um tópico:", topics, key="classification_topic")

    if selected_topic == "Resultados dos Modelos":
        resultados = [
            {"Modelo": "SVM (StandardScaler)", "MSE": 3.941641e+16, "R²": -0.129421},
            {"Modelo": "SVM (StandardScaler com log)", "MSE": 7.976347e+15, "R²": 0.771449},
            {"Modelo": "KNN (StandardScaler)", "MSE": 7.062214e+15, "R²": 0.797642},
            {"Modelo": "KNN (StandardScaler com log)", "MSE": 1.041102e+16, "R²": 0.701687},
            {"Modelo": "Random Forest (StandardScaler)", "MSE": 1.305475e+15, "R²": 0.962593},
            {"Modelo": "Random Forest (StandardScaler com log)", "MSE": 3.965721e+16, "R²": -0.136320},
        ]
        df_resultados = pd.DataFrame(resultados)
        styled = df_resultados.style\
            .background_gradient(subset=['R²'], cmap='RdYlGn')\
            .background_gradient(subset=['MSE'], cmap='RdYlGn_r')\
            .format({'MSE': '{:.2e}', 'R²': '{:.6f}'})
        st.write("### Tabela de Resultados dos Modelos")
        st.dataframe(styled, use_container_width=True)

    elif selected_topic == "Feature Importance Random Forest":
        show_feature_importance_rf()

    elif selected_topic == "Force Plot Random Forest":
        show_force_plot_rf()

    elif selected_topic == "Shap Random Forest":
        show_shap_summary_rf()
