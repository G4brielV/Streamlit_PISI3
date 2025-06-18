import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

def explode_genre_names(df):
    # Divide a coluna 'genre_names' em listas e explode para linhas
    if 'genre_names' in df.columns:
        return df.assign(genre_names=df['genre_names'].str.split(',')).explode('genre_names')
    else:
        return df

def plot_elbow_method(df):
    """Exibe o gráfico do método do cotovelo para escolha do K ideal."""
    st.subheader("Método do Cotovelo")
    features = ['popularity', 'vote_average', 'revenue']
    df_kmeans = df[features].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_kmeans)

    wss = []
    k_values_elbow = list(range(1, 11))
    for k in k_values_elbow:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wss.append(kmeans.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values_elbow, y=wss, mode='lines+markers', marker=dict(size=8)))
    fig.update_layout(
        title='Método do Cotovelo',
        xaxis_title='Número de Clusters (K)',
        yaxis_title='Soma dos Quadrados Intra-cluster (WSS)',
        hovermode='x unified',
        dragmode='zoom',
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

def plot_silhouette_plotly(X_scaled, labels):
    """Gráfico de silhueta interativo com Plotly."""
    from sklearn.metrics import silhouette_samples
    import plotly.graph_objects as go
    k = len(np.unique(labels))
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    y_lower = 10
    traces = []
    yticks = []
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        traces.append(go.Bar(
            x=ith_cluster_silhouette_values,
            y=np.arange(y_lower, y_upper),
            orientation='h',
            marker=dict(color=cm.nipy_spectral(float(i) / k)),
            name=f'Cluster {i}',
            showlegend=True if i == 0 else False,
            hoverinfo='x+y'
        ))
        yticks.append(y_lower + 0.5 * size_cluster_i)
        y_lower = y_upper + 10

    silhouette_avg = np.mean(sample_silhouette_values)
    layout = go.Layout(
        title="Gráfico de Silhueta para cada Cluster",
        xaxis=dict(title="Coeficiente de Silhueta"),
        yaxis=dict(title="Amostras", showticklabels=False),
        shapes=[
            dict(
                type='line',
                x0=silhouette_avg, x1=silhouette_avg,
                y0=0, y1=y_lower,
                line=dict(color='red', dash='dash')
            )
        ],
        height=300,
        dragmode='zoom'
    )
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Score médio da silhueta:** {silhouette_avg:.3f}")

def plot_silhouette_matplotlib(X_scaled, labels):
    """Gráfico de silhueta tradicional (matplotlib, compacto)."""
    from sklearn.metrics import silhouette_samples
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    k = len(np.unique(labels))
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    silhouette_avg = np.mean(sample_silhouette_values)
    y_lower = 10

    fig, ax = plt.subplots(figsize=(4, 2.5))
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=8)
        y_lower = y_upper + 10

    ax.set_title("Silhueta por Cluster", fontsize=10)
    ax.set_xlabel("Coef. Silhueta", fontsize=9)
    ax.set_ylabel("Clusters", fontsize=9)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    st.write(f"**Score médio da silhueta:** {silhouette_avg:.3f}")

def plot_kmeans_single(df):
    """Executa e exibe análise K-Means para um valor de K escolhido."""
    st.subheader("Análise com K-Means")
    k = st.slider("Número de clusters (K)", min_value=2, max_value=10, value=4, key="kmeans_single")

    if st.button("Rodar K-Means"):
        features = ['popularity', 'vote_average', 'revenue']
        df_kmeans = df[features].dropna().copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_kmeans)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        df_kmeans['cluster'] = labels

        st.success("K-Means aplicado com sucesso!")

        # Gráfico: Popularidade média por cluster
        st.write("#### Popularidade média por Cluster")
        pop_means = df_kmeans.groupby('cluster')['popularity'].mean()
        fig1 = px.bar(
            x=pop_means.index, y=pop_means.values,
            labels={'x': 'Cluster', 'y': 'Popularidade média'},
            title="Popularidade por Cluster"
        )
        fig1.update_layout(dragmode='zoom')
        fig1.update_xaxes(fixedrange=False)
        fig1.update_yaxes(fixedrange=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Gráfico: Distribuição de gêneros por cluster
        st.write("#### Distribuição de Gêneros por Cluster")
        df_with_id = df.copy()
        df_with_id = df_with_id.loc[df_kmeans.index]
        df_with_id['cluster'] = labels
        df_exploded_k = explode_genre_names(df_with_id)
        if 'genre_names' in df_exploded_k.columns:
            dist = df_exploded_k.groupby(['cluster','genre_names']).size().unstack(fill_value=0)
            prop = dist.div(dist.sum(axis=1), axis=0)
            fig2 = px.imshow(
                prop.values,
                labels=dict(x="Gênero", y="Cluster", color="Proporção"),
                x=prop.columns,
                y=prop.index,
                aspect="auto",
                color_continuous_scale="viridis",
                title="Proporção de Gêneros por Cluster"
            )
            fig2.update_layout(dragmode='zoom')
            fig2.update_xaxes(fixedrange=False)
            fig2.update_yaxes(fixedrange=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Gráfico: Silhueta
        st.write("#### Análise de Silhueta")
        plot_silhouette_matplotlib(X_scaled, labels)

        # Gráficos de Caixa por Cluster
        st.write("#### 📊 Gráficos de Caixa por Cluster")
        for feature in features:
            fig_box, ax_box = plt.subplots(figsize=(5, 3), dpi=80)
            sns.boxplot(x='cluster', y=feature, data=df_kmeans, palette='Set2', ax=ax_box)
            ax_box.set_title(f'Distribuição de {feature.capitalize()} por Cluster', fontsize=10)
            ax_box.set_xlabel("Cluster", fontsize=9)
            ax_box.set_ylabel(feature.capitalize(), fontsize=9)
            ax_box.tick_params(labelsize=8)
            ax_box.grid(True, linestyle="--", alpha=0.5)
            col1, col2, col3 = st.columns([1, 2, 1])  # centralizar
            with col2:
                st.pyplot(fig_box)

    

def plot_regression_popularity_rating(df):
    """Regressão linear entre popularidade e média de votos (matplotlib)."""
    st.subheader("Regressão Linear: Popularidade vs. Média de Votos (Matplotlib)")
    st.write(
        "Analisamos a relação entre a popularidade de um filme e sua nota média. "
        "Para evitar que filmes extremamente populares (outliers) distorçam a visualização, "
        "filtramos os dados para incluir apenas os 95% dos filmes menos populares."
    )
    limite_superior = df['popularity'].quantile(0.95)
    df_filtrado = df[df['popularity'] <= limite_superior]
    X = df_filtrado[['popularity']]
    y = df_filtrado['vote_average']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("R² (R-quadrado)", f"{r2:.3f}")
    col2.metric("MSE (Erro Quadrático Médio)", f"{mse:.3f}")
    col3.metric("Nº de Filmes Analisados", str(len(df_filtrado)))
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_idx = X_test['popularity'].argsort()
    X_sorted = X_test.iloc[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]
    ax.scatter(X_test, y_test, color='blue', alpha=0.4, label='Dados Reais')
    ax.plot(X_sorted, y_pred_sorted, color='red', linewidth=2.5, label='Regressão Linear')
    ax.set_title('Popularidade vs. Média de Votos (com Regressão Linear)')
    ax.set_xlabel('Popularidade (95% dos dados, sem outliers)')
    ax.set_ylabel('Média de Votos')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_regression_popularity_rating2(df):
    """Regressão linear entre popularidade e média de votos (plotly interativo)."""
    st.subheader("Regressão Linear: Popularidade vs. Média de Votos (Plotly)")
    limite_superior = df['popularity'].quantile(0.95)
    df_filtrado = df[df['popularity'] <= limite_superior]
    X = df_filtrado[['popularity']]
    y = df_filtrado['vote_average']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    fig_plotly = px.scatter(
        df_filtrado,
        x='popularity',
        y='vote_average',
        trendline="ols",
        trendline_color_override="red",
        title='Passe o mouse sobre os pontos para ver os detalhes. Você também pode dar zoom (scroll ou arrasto).',
        labels={'popularity': 'Popularidade (filtrada, 95%)', 'vote_average': 'Média de Votos'},
        opacity=0.5,
        hover_data=['title'] if 'title' in df_filtrado.columns else None
    )
    fig_plotly.update_layout(
        xaxis_title="Popularidade",
        yaxis_title="Média de Votos",
        title_font_size=18,
        dragmode="zoom",
    )
    fig_plotly.update_xaxes(fixedrange=False)
    fig_plotly.update_yaxes(fixedrange=False)
    st.plotly_chart(fig_plotly, use_container_width=True)

def run_clustering():
    st.title("Clusterização de Filmes")
    data_path = 'data/filmes_filtrados_sem_nulos.csv'
    df = load_data(data_path)

    # Navegação dos tópicos de clusterização
    st.sidebar.subheader("Tópicos de Clusterização")
    topics = [
        "Método do Cotovelo",
        "K-Means (1 valor de K)",
        "Regressão Linear (Matplotlib)",
        "Regressão Linear (Plotly)"
    ]
    selected_topic = st.sidebar.radio("Escolha um tópico:", topics, key="clustering_topic")

    if selected_topic == "Método do Cotovelo":
        plot_elbow_method(df)
    elif selected_topic == "K-Means (1 valor de K)":
        plot_kmeans_single(df)
    elif selected_topic == "Regressão Linear (Matplotlib)":
        plot_regression_popularity_rating(df)
    elif selected_topic == "Regressão Linear (Plotly)":
        plot_regression_popularity_rating2(df)