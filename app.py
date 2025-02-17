import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from fpdf import FPDF

# Dados de entrada
#x76 = 0 # Deslocamento da T76
#hu76 = 24.0 # Altura útil da T76

# Definir os dados do perfil do terreno
terrain_data = pd.DataFrame({
    "x": [0.00, 8.58, 28.67, 47.76, 68.97, 89.57, 120.89, 135.97, 196.73, 225.22, 240.41, 265.32, 283.00, 306.42, 319.21, 334.10, 352.09, 368.54, 391.36, 403.65, 416.46, 430.78, 436.33, 449.48, 467.72, 474.79, 488.53, 511.26, 537.31, 553.84, 575.53, 592.26, 611.88, 635.54, 654.36, 672.24, 698.24, 700.95, 721.57, 730.79, 757.14, 761.32, 767.02, 773.35, 784.04, 818.21, 822.51, 823.46, 829.77],
    "y": [22.55, 22.77, 21.87, 21.84, 21.88, 22.03, 22.36, 22.52, 22.65, 22.90, 22.95, 22.81, 23.23, 23.46, 23.85, 24.00, 23.99, 24.24, 24.28, 24.24, 24.25, 23.95, 24.03, 24.23, 24.62, 24.57, 24.02, 23.90, 23.31, 23.25, 23.84, 24.05, 23.61, 22.88, 23.00, 22.68, 21.77, 21.68, 21.86, 22.21, 22.81, 22.98, 22.95, 23.35, 23.67, 24.70, 24.83, 24.75, 24.86]
})
# Interpolação do perfil do terreno
x_terreno = terrain_data["x"].values
y_terreno = terrain_data["y"].values
cs = CubicSpline(x_terreno, y_terreno)
x_fino = np.linspace(x_terreno[0], x_terreno[-1], 300)
y_fino = cs(x_fino)

# Função para gerar os cálculos e tabelas de saída
def calcular_resultados(x76, hu76):
    # Definir dados das estruturas
    estruturas = {
        "75": {"x": 0.0, "hu": 25.5},
        "76": {"x": 436.33+x76, "hu": hu76},
        "77": {"x": 829.77, "hu": 18.0},
    }
    
    # Definir vãos entre estruturas
    vãos = {
        "75-76": estruturas["76"]["x"] - estruturas["75"]["x"],
        "76-77": estruturas["77"]["x"] - estruturas["76"]["x"],
    }
    
    # Cálculo do vão médio
    vao_medio = {
        "75": round((vãos["75-76"] / 2) + 153.88, 2),
        "76": round((vãos["75-76"] / 2) + (vãos["76-77"] / 2), 2),
        "77": round((vãos["76-77"] / 2) + 93.05, 2),
    }
    
    # Dados da tensão e massa linear do cabo
    T0_daN = 2391  # daN (decaNewton)
    T0_kgf = T0_daN * 1.01972  # Conversão para kgf
    p = 1.034  # kg/m
    
    g = 9.81  # Gravidade (m/s²)
    
    # Cálculo das flechas
    flechas = {
        "75-76": round(vãos["75-76"]**2 * p / (8 * T0_kgf)+vãos["75-76"]**4 * p**3 / (384 * T0_kgf**3), 2),
        "76-77": round(vãos["76-77"]**2 * p / (8 * T0_kgf)+vãos["76-77"]**4 * p**3 / (384 * T0_kgf**3), 2),
    }
    
    
    # Cálculo dos pontos de suporte e meio dos vãos
    pontos_suporte = {}
    pontos_meio_vão = {}
    
    for key, vão in vãos.items():
        torre1, torre2 = key.split('-')
        x_torre1, x_torre2 = estruturas[torre1]["x"], estruturas[torre2]["x"]
        hu_torre1, hu_torre2 = estruturas[torre1]["hu"], estruturas[torre2]["hu"]
        
        # Cálculo das cotas dos suportes
        cota_suporte1 = cs(x_torre1) + hu_torre1
        cota_suporte2 = cs(x_torre2) + hu_torre2
        pontos_suporte[key] = [(x_torre1, cota_suporte1), (x_torre2, cota_suporte2)]
        
        # Cálculo do ponto médio do vão
        x_meio = (x_torre1 + x_torre2) / 2
        cota_meio = ((cota_suporte1 + cota_suporte2) / 2) - flechas[key]
        pontos_meio_vão[key] = (x_meio, cota_meio)
    
    # Função alternativa da catenária considerando tensão horizontal T0
    def catenary(x, T0, y0, x0):
        return y0 + (T0 / (p * g)) * (np.cosh((p * g / T0) * (x - x0)) - 1)
    
    # Ajuste separado para cada catenária
    catenarias = {}
    params_catenarias = {}
    
    for key in vãos:
        x_torre1, x_torre2 = pontos_suporte[key][0][0], pontos_suporte[key][1][0]
        y_torre1, y_torre2 = pontos_suporte[key][0][1], pontos_suporte[key][1][1]
        x_range = np.linspace(x_torre1, x_torre2, 200)
    
        # Definir os 3 pontos para ajuste
        x_vals = np.array([x_torre1, pontos_meio_vão[key][0], x_torre2])
        y_vals = np.array([y_torre1, pontos_meio_vão[key][1], y_torre2])
        
        # Melhorando a estimativa inicial para curve_fit
        p0 = [T0_kgf, np.mean(y_vals), np.mean(x_vals)]
        
        # Ajuste usando a equação alternativa da catenária
        try:
            params, _ = curve_fit(catenary, x_vals, y_vals, p0=p0, maxfev=10000)
            params_catenarias[key] = params  # Salvar parâmetros
            y_catenaria = catenary(x_range, *params)
            catenarias[key] = (x_range, y_catenaria)
        except RuntimeError:
            print(f"Falha no ajuste da catenária para {key}. Usando aproximação linear.")
            y_catenaria = np.interp(x_range, x_vals, y_vals)
            catenarias[key] = (x_range, y_catenaria)
    
    # Parâmetros do cabo
    S = 375.35  # Área da seção transversal (mm²)
    E = 62e3 * 0.10197  # Módulo de elasticidade (kgf/mm²)
    alpha_t = 23e-6  # Coeficiente de dilatação térmica linear (1/°C)
    
    # Temperaturas
    t1 = 25  # Temperatura inicial (°C)
    t2 = 75  # Temperatura final (°C)
    
    # Tração inicial a 25°C
    T01 = T0_kgf
    
    # Cálculo do vão regulador Ar
    vao_componentes = [307.77, vãos["75-76"], vãos["76-77"], 186.10]
    vao_regulador = np.sqrt(sum(a**3 for a in vao_componentes) / sum(vao_componentes))
    
    # Resolver a equação de mudança de estado para cada vão com o vão regulador
    for key, A in vãos.items():
        A_eq = vao_regulador  # Usando o vão regulador na equação
        a = 1
        b = (E * S * p**2 * A_eq**2) / (24 * T01**2) + E * S * alpha_t * (t2 - t1) - T01
        c = 0
        d = - (E * S * p**2 * A_eq**2) / 24
        
        def equacao_tracao(T02):
            return T02**3 * a + T02**2 * b + T02 * c + d
        
        chute_inicial = max(T01, 1000)
        T02_solucao = fsolve(equacao_tracao, chute_inicial)[0]
    
    # Calcular novas flechas
    flechas_novas = {
        "75-76": round(vãos["75-76"]**2 * p / (8 * T02_solucao)+vãos["75-76"]**4 * p**3 / (384 * T02_solucao**3), 2),
        "76-77": round(vãos["76-77"]**2 * p / (8 * T02_solucao)+vãos["76-77"]**4 * p**3 / (384 * T02_solucao**3), 2),
    }
     
    # Calcular novos pontos no meio do vão utilizando pontos de suporte
    pontos_meio_vão_novos = {}
    for key, A in vãos.items():
        x_meio = (pontos_suporte[key][0][0] + pontos_suporte[key][1][0]) / 2
        y_meio = (pontos_suporte[key][0][1] + pontos_suporte[key][1][1]) / 2 - flechas_novas[key]
        pontos_meio_vão_novos[key] = (x_meio, y_meio)
    
    catenarias_novas = {}
    for key, A in vãos.items():
        x_vals = np.array([pontos_suporte[key][0][0], pontos_meio_vão_novos[key][0], pontos_suporte[key][1][0]])
        y_vals = np.array([pontos_suporte[key][0][1], pontos_meio_vão_novos[key][1], pontos_suporte[key][1][1]])
        p0 = [max(T02_solucao, 500), np.mean(x_vals), np.mean(y_vals)]
        popt, _ = curve_fit(catenary, x_vals, y_vals, p0=p0)
        x_range = np.linspace(pontos_suporte[key][0][0], pontos_suporte[key][1][0], 100)
        y_range = catenary(x_range, *popt)
        catenarias_novas[key] = (x_range, y_range)
    
    
    # Encontrar a menor distância entre o cabo e o solo
    def menor_distancia_cabo_solo(catenarias, cs):
        distancias_minimas = {}
        for key, (x_catenaria, y_catenaria) in catenarias.items():
            distancias = np.abs(y_catenaria - cs(x_catenaria))
            idx_min = np.argmin(distancias)
            distancias_minimas[key] = (x_catenaria[idx_min], distancias[idx_min])
        return distancias_minimas
    
    # Encontrar os vértices (pontos de mínimo) das catenárias
    def encontrar_vertices(catenarias):
        vertices = {}
        for key, (x_catenaria, y_catenaria) in catenarias.items():
            idx_min = np.argmin(y_catenaria)
            vertices[key] = (x_catenaria[idx_min], y_catenaria[idx_min])
        return vertices
    
    # Calcular a menor distância para as catenárias a 75°C
    distancias_minimas_75 = menor_distancia_cabo_solo(catenarias_novas, cs)
    
    # Calcular os vértices das catenárias a 75°C
    vertices_catenarias_75 = encontrar_vertices(catenarias_novas)
    
    # Calcular o vão gravante
    vao_gravante = {
        "75": round((vertices_catenarias_75["75-76"][0] - 0) + 243.63, 2),
        "76": round(vertices_catenarias_75["76-77"][0] - vertices_catenarias_75["75-76"][0], 2),
        "77": round((829.77 - vertices_catenarias_75["76-77"][0]) + 278.14, 2)
    }
    total_vao_gravante = [vao_gravante["75"], vao_gravante["76"], vao_gravante["77"]]
    
    # Criar e exibir a tabela de resultados
    resultados_df = pd.DataFrame({
        "Estrutura": ["75", "76", "77"],
        "Progressiva": [estruturas["75"]["x"], estruturas["76"]["x"], estruturas["77"]["x"]],
        "Vão à Frente": [round(vãos["75-76"], 2), round(vãos["76-77"], 2), 186.10],
        "Vão Médio": [vao_medio["75"], vao_medio["76"], vao_medio["77"]],
        "Vão Gravante": total_vao_gravante,
        "Cota": [round(float(cs(estruturas["75"]["x"])), 2), round(float(cs(estruturas["76"]["x"])), 2), round(float(cs(estruturas["77"]["x"])), 2)],
        "Altura": [estruturas["75"]["hu"], estruturas["76"]["hu"], estruturas["77"]["hu"]],
    })
    
    # Exibir resultado do vão regulador
    print(f"\nVão Regulador: {vao_regulador:.2f} m")
    
    # Calcular a tensão horizontal em diferentes temperaturas
    temperaturas = np.arange(0, 55, 5)
    T02_temperaturas = []
    
    for t2 in temperaturas:
        def equacao_tracao(T02):
            return T02**3 + T02**2 * ((E * S * p**2 * vao_regulador**2) / (24 * T01**2) + E * S * alpha_t * (t2 - 25) - T01) - (E * S * p**2 * vao_regulador**2) / 24
        
        T02_solucao = round(fsolve(equacao_tracao, T01)[0], 2)
        T02_temperaturas.append(T02_solucao)
    
    
    # Calcular flechas para cada temperatura
    flechas_temperaturas = []
    
    for T02 in T02_temperaturas:
        flechas = {
            "75-76": round(vãos["75-76"]**2 * p / (8 * T02) + vãos["75-76"]**4 * p**3 / (384 * T02**3), 2),
            "76-77": round(vãos["76-77"]**2 * p / (8 * T02) + vãos["76-77"]**4 * p**3 / (384 * T02**3), 2),
        }
        flechas_temperaturas.append(flechas)
    
    # Criar tabela com os resultados de flechas
    flechas_df = pd.DataFrame(flechas_temperaturas, columns=["75-76", "76-77"], index=[f"{t} °C" for t in temperaturas]).T
    
    # Criar tabela com os resultados
    tensao_df = pd.DataFrame([T02_temperaturas], columns=[f"{t} °C" for t in temperaturas])

    return resultados_df, tensao_df, flechas_df, catenarias, catenarias_novas, distancias_minimas_75, estruturas, cs


def gerar_relatorio(x76, hu76, resultados_df, tensao_df, flechas_df):
    # Criar o objeto PDF
    pdf = FPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título do relatório
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "Análise da Estrutura 76", ln=True, align='C')

    # Seção de Dados de Entrada
    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=11)
    pdf.cell(0, 10, "Dados de Entrada:", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, f"Movimentação: {x76} m\n"
                         f"Altura útil: {hu76} m\n")

    # Adicionar Resultados Obtidos
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=11)
    pdf.cell(0, 10, "Resultados Obtidos:", ln=True)

    # Função para formatar tabelas no PDF
    def add_table_to_pdf(pdf, df, include_index=False):
        pdf.set_font("Arial", size=8)
        col_width = pdf.w / (len(df.columns) + (1 if include_index else 0) + 1)
        
        # Cabeçalhos
        pdf.set_font("Arial", style='B', size=8)
        if include_index:
            pdf.cell(col_width, 6, df.index.name if df.index.name else "Vão", border=1, align='C')
        for col in df.columns:
            pdf.cell(col_width, 6, col, border=1, align='C')
        pdf.ln()
        
        # Linhas
        pdf.set_font("Arial", size=8)
        for i in range(len(df)):
            if include_index:
                pdf.cell(col_width, 6, str(df.index[i]), border=1, align='C')
            for col in df.columns:
                pdf.cell(col_width, 6, str(df.iloc[i][col]), border=1, align='C')
            pdf.ln()

    # Adicionar gráfico
    pdf.set_font("Arial", style='B', size=10)
    
    # Salvar e inserir no PDF
    pdf.image(graph_path, x=10, w=180)
    
    # Tabela de Menores Distâncias Cabo-Solo
    pdf.ln(1)
    pdf.set_font("Arial", style='B', size=9)
    pdf.cell(0, 10, "Menores Distâncias entre Cabo e Solo:", ln=True)
    pdf.set_font("Arial", size=8)

    distancias_text = "\n".join([f"{key}: Distância progressiva = {val[0]:.2f} m, Distância cabo-solo = {val[1]:.2f} m" for key, val in distancias_minimas_75.items()])
    pdf.multi_cell(0, 5, distancias_text)

    resultados_df_fmt = resultados_df.copy()
    tensao_df_fmt = tensao_df.copy()
    flechas_df_fmt = flechas_df.copy()
    
    # Aplicar formatação apenas em colunas numéricas
    for df in [resultados_df_fmt, tensao_df_fmt, flechas_df_fmt]:
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}")  # Formata como string com duas casas decimais

    # Adicionar tabelas ao relatório
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(0, 10, "Tabela de Locação (m)", ln=True)
    add_table_to_pdf(pdf, resultados_df_fmt)


    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(0, 10, "Tração Horizontal (kgf)", ln=True)
    add_table_to_pdf(pdf, tensao_df_fmt)

    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(0, 10, "Flechas (m)", ln=True)
    add_table_to_pdf(pdf, flechas_df_fmt, include_index=True)

    # Salvar PDF
    pdf_path = f"./analise_T76_{x76}_{hu76}.pdf"
    pdf.output(pdf_path)

    # Exibir o caminho do PDF
    txt = f"Relatório gerado: {pdf_path}"
    print(txt)
    return pdf_path

st.set_page_config(page_title="Análise da Estrutura 76", page_icon="🏗️")
# Configuração da interface do Streamlit
st.title("Análise da Estrutura 76")

# Sliders para entrada dos parâmetros
x76 = st.slider("Deslocamento (m)", -30, 30, value=0, step=5)
hu76 = st.slider("Altura útil (m)", 18.0, 36.0, value=24.0, step=0.5)


# Executa os cálculos
resultados_df, tensao_df, flechas_df, catenarias, catenarias_novas, distancias_minimas_75, estruturas, cs = calcular_resultados(x76, hu76)


# Exibir gráfico de flechas
st.subheader("Perfil")
fig, ax = plt.subplots(figsize=(12, 6))
# Exibir gráfico atualizado
plt.plot(x_fino, y_fino, color='brown')
for key, (x_catenaria, y_catenaria) in catenarias.items():
    plt.plot(x_catenaria, y_catenaria, linestyle='-', color='orange')
for key, (x_catenaria, y_catenaria) in catenarias_novas.items():
    plt.plot(x_catenaria, y_catenaria, linestyle='-', color='blue')
    plt.scatter(distancias_minimas_75[key][0], cs(distancias_minimas_75[key][0]), color='red')
    
# Adicionar linhas verticais para representar cada estrutura
for key, estrutura in estruturas.items():
    x_estrutura = estrutura["x"]
    y_terreno = cs(x_estrutura)
    y_suporte = y_terreno + estrutura["hu"]

    plt.vlines(x=x_estrutura, ymin=y_terreno, ymax=y_suporte, color='green', linestyles='-')
    #plt.scatter([x_estrutura], [y_suporte], color='black', marker='o')
    # Adicionar etiqueta (número da estrutura) acima do ponto do suporte
    plt.text(x_estrutura, y_suporte + 1, f"{key}", fontsize=12, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

# Criar legenda genérica para as temperaturas
plt.plot([], [], color='orange', linestyle='-', label="25°C")
plt.plot([], [], color='blue', linestyle='-', label="75°C")


plt.xlabel("Distância Progressiva (m)")
plt.ylabel("Altitude (m)")
plt.legend(loc='lower right')  # Posiciona a legenda no canto inferior direito
plt.grid()
graph_path = "./grafico_flechas.png"
plt.savefig(graph_path)
plt.close()
st.pyplot(fig)

# Exibir as menores distâncias entre cabo e solo
st.subheader("Menores Distâncias Cabo e Solo")

for key, val in distancias_minimas_75.items():
    st.write(f"**{key}**: Distância progressiva = `{val[0]:.2f} m`, Distância cabo-solo = `{val[1]:.2f} m`")

# Exibir tabela de resultados
st.subheader("Tabela de Locação (m)")
st.dataframe(
    resultados_df.style
    .format({col: "{:.2f}" for col in resultados_df.select_dtypes(include="number").columns})  # Apenas números
    .set_properties(**{"text-align": "center"})  # Centralizar texto
)

# Exibir tabela de tensão horizontal
st.subheader("Tração Horizontal (kgf)")
st.dataframe(
    tensao_df.style
    .format({col: "{:.2f}" for col in tensao_df.select_dtypes(include="number").columns})  # Apenas números
    .set_properties(**{"text-align": "center"})  # Centralizar texto
)

# Exibir tabela de flechas
st.subheader("Flechas (m)")
st.dataframe(
    flechas_df.style
    .format({col: "{:.2f}" for col in flechas_df.select_dtypes(include="number").columns})  # Apenas números
    .set_properties(**{"text-align": "center"})  # Centralizar texto
)

   
    # Botão para gerar relatório PDF
if st.button("📄 Gerar e Baixar Relatório PDF"):
    pdf_path = gerar_relatorio(x76, hu76, resultados_df, tensao_df, flechas_df)
    st.success("Relatório gerado com sucesso!")
    
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Baixar Relatório",
            data=pdf_file,
            file_name=f"analise_T76_{x76}_{hu76}.pdf",
            mime="application/pdf"
        )
