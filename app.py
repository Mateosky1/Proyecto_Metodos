import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math
from scipy.stats import norm, t as t_dist
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONSTANTES DEL MODELO
# ==========================================
N_CROMOS        = 980
CROMOS_POR_FUNDA = 7
COSTO_FUNDA     = 1.20
FUNDAS_INICIALES = 140
MAX_RONDAS      = 500

# -----------------------------------------
# Paleta de colores personalizada
# -----------------------------------------
COLOR_NINGUNA    = "#E24B4A"   # rojo
COLOR_BILATERAL  = "#EF9F27"  # ambar
COLOR_MULTILAT   = "#1D9E75"  # verde teal

# ==========================================
# MODULO 0 - MOTOR DE INTERCAMBIOS
# ==========================================
def ejecutar_intercambios(albumes, repetidos, estrategia):
    """
    Ejecuta una ronda de intercambios entre participantes.
    Retorna (albumes, repetidos, intercambios_realizados).
    """
    num_participantes = albumes.shape[0]
    intercambios_realizados = 0

    if estrategia == "Ninguna":
        return albumes, repetidos, 0

    G = nx.DiGraph()

    # Construir grafo de oferta: arista (a->b) si 'a' tiene un repetido que le falta a 'b'
    for a in range(num_participantes):
        for b in range(num_participantes):
            if a == b:
                continue
            for cromo_rep, cantidad in repetidos[a].items():
                if cantidad > 0 and not albumes[b, cromo_rep]:
                    if G.has_edge(a, b):
                        G[a][b]['cromos'].append(cromo_rep)
                    else:
                        G.add_edge(a, b, cromos=[cromo_rep])

    # -- Estrategia 1: Solo intercambios bilaterales --
    if estrategia == "Solo Bilateral":
        aristas = list(G.edges())
        for u, v in aristas:
            if not (G.has_edge(u, v) and G.has_edge(v, u)):
                continue
            cromos_uv = G[u][v]['cromos']
            cromos_vu = G[v][u]['cromos']
            if not cromos_uv or not cromos_vu:
                continue

            cromo_u_v = cromos_uv[0]
            cromo_v_u = cromos_vu[0]

            # Validar que el intercambio sigue siendo valido
            if repetidos[u].get(cromo_u_v, 0) > 0 and not albumes[v, cromo_u_v] \
               and repetidos[v].get(cromo_v_u, 0) > 0 and not albumes[u, cromo_v_u]:

                repetidos[u][cromo_u_v] -= 1
                albumes[v, cromo_u_v] = True
                repetidos[v][cromo_v_u] -= 1
                albumes[u, cromo_v_u] = True
                intercambios_realizados += 2

                # Limpiar grafo
                cromos_uv.remove(cromo_u_v)
                if not cromos_uv:
                    G.remove_edge(u, v)
                cromos_vu.remove(cromo_v_u)
                if not cromos_vu:
                    G.remove_edge(v, u)

    # -- Estrategia 2: Triangular y multilateral (ciclos) --
    elif estrategia == "Triangular y Multilateral":
        while True:
            try:
                ciclo = nx.find_cycle(G, orientation='original')
                for origen, destino, _ in ciclo:
                    if not G.has_edge(origen, destino):
                        continue
                    cromos_disponibles = G[origen][destino]['cromos']
                    if not cromos_disponibles:
                        continue
                    cromo = cromos_disponibles[0]

                    # Validacion de estado actual
                    if repetidos[origen].get(cromo, 0) > 0 and not albumes[destino, cromo]:
                        repetidos[origen][cromo] -= 1
                        albumes[destino, cromo] = True
                        intercambios_realizados += 1

                    cromos_disponibles.remove(cromo)
                    if not cromos_disponibles:
                        G.remove_edge(origen, destino)

                    # Limpiar otras ofertas del mismo cromo hacia 'destino'
                    aristas_a_borrar = []
                    for u, v, data in list(G.in_edges(destino, data=True)):
                        if cromo in data['cromos']:
                            data['cromos'].remove(cromo)
                            if not data['cromos']:
                                aristas_a_borrar.append((u, v))
                    G.remove_edges_from(aristas_a_borrar)

            except nx.NetworkXNoCycle:
                break

    return albumes, repetidos, intercambios_realizados


# ==========================================
# MODULO 1 - PROBABILIDAD ANALITICA
# ==========================================
def prob_analitica_completar(faltantes, total_repetidos_grupo, n_cromos=N_CROMOS):
    """
    Retorna (p_funda_util, indice_liquidez).

    p_funda_util:
        Probabilidad de que al menos 1 de los k=7 cromos de una funda sea
        un cromo que le falta al participante, bajo el supuesto de distribucion
        uniforme e independiente.
            P = 1 - (1 - f/n)^k
        Es una probabilidad en sentido estricto.

    indice_liquidez:
        Indicador heuristico de disponibilidad de intercambio.
        Mide si el grupo dispone de suficientes repetidos para cubrir los
        faltantes del participante: min(1, R_total / f).
        NOTA: NO es una probabilidad formal. Asume que todos los repetidos
        del grupo son distintos y utiles para este participante, lo cual es
        una cota superior optimista. Se presenta como indicador de liquidez,
        no como probabilidad de completacion.
    """
    if faltantes <= 0:
        return 0.0, 0.0

    # Probabilidad analitica: funda con al menos 1 cromo nuevo
    p_funda_util = 1.0 - (1.0 - faltantes / n_cromos) ** CROMOS_POR_FUNDA

    # Indice de liquidez del grupo (heuristica, no probabilidad formal)
    if total_repetidos_grupo > 0:
        indice_liquidez = min(1.0, total_repetidos_grupo / faltantes)
    else:
        indice_liquidez = 0.0

    return p_funda_util, indice_liquidez


def prob_analitica_completar_exacto(faltantes, n_cromos=N_CROMOS, n_cromos_funda=CROMOS_POR_FUNDA):
    """
    P(completar album exactamente en la proxima compra de k_fundas fundas).
    Aproximacion con coleccionista de cupones:
        E[fundas adicionales] = n * H_n - (n-f) * H_(n-f)
    donde H_k es la k-esima suma armonica y f = cromos ya obtenidos.
    """
    if faltantes <= 0:
        return 1.0, 0
    cromos_obtenidos = n_cromos - faltantes
    # E[cromos para completar | tienes c] = sum_{i=c}^{n-1} n/(n-i)
    # Aproximamos con la integral
    esperanza_extra = sum(n_cromos / (n_cromos - i) for i in range(cromos_obtenidos, n_cromos))
    fundas_esperadas = math.ceil(esperanza_extra / n_cromos_funda)
    return fundas_esperadas


def prob_experimental_completar(historial_completados, num_participantes):
    """
    CDF empirica de completacion: fraccion acumulada de participantes
    que han completado su album hasta la ronda r.

    Parametros
    ----------
    historial_completados : list[int]
        Cuantos participantes completaron en cada ronda (no acumulado).
    num_participantes : int
        Total de participantes de la simulacion.

    Retorna
    -------
    list[float] con la probabilidad experimental acumulada por ronda.
    """
    if num_participantes <= 0:
        return []
    probs = []
    acumulado = 0
    for completados in historial_completados:
        acumulado += completados
        probs.append(acumulado / num_participantes)
    return probs


# ==========================================
# MODULO 2 - SIMULACION COMPLETA
# ==========================================
def simular_ciclo_completo(num_participantes, estrategia):
    """
    Ejecuta un ciclo de vida completo de la simulacion.
    Retorna un dict con todas las metricas.
    """
    # Inicializacion
    albumes  = np.zeros((num_participantes, N_CROMOS), dtype=bool)
    repetidos = [{} for _ in range(num_participantes)]

    resultados = {
        "fundas_compradas"           : np.zeros(num_participantes, dtype=int),
        "cromos_comprados_totales"   : np.zeros(num_participantes, dtype=int),
        "rondas_necesarias"          : 0,
        "intercambios_totales"       : 0,
        "historial_faltantes"        : [],          # shape: (rondas, participantes)
        "historial_prob_analitica"   : [],          # P(funda util) promedio por ronda
        "historial_indice_liquidez"  : [],          # Indice heuristico de liquidez de intercambio
        "historial_completados_ronda": [],          # cuantos completan en cada ronda
        "ronda_completacion"         : np.full(num_participantes, -1, dtype=int),  # ronda en que cada uno termina
    }

    # -- Compra base inicial --
    for i in range(num_participantes):
        resultados["fundas_compradas"][i]         += FUNDAS_INICIALES
        resultados["cromos_comprados_totales"][i] += FUNDAS_INICIALES * CROMOS_POR_FUNDA
        cromos_comprados = np.random.randint(0, N_CROMOS, FUNDAS_INICIALES * CROMOS_POR_FUNDA)
        for cromo in cromos_comprados:
            if not albumes[i, cromo]:
                albumes[i, cromo] = True
            else:
                repetidos[i][cromo] = repetidos[i].get(cromo, 0) + 1

    ronda_actual = 1

    while ronda_actual <= MAX_RONDAS:
        # A. Intercambios
        albumes, repetidos, intercambios_ronda = ejecutar_intercambios(
            albumes, repetidos, estrategia
        )
        resultados["intercambios_totales"] += intercambios_ronda

        # B. Estado actual
        faltantes_actuales = N_CROMOS - np.sum(albumes, axis=1)

        # B1. Registrar quien completa en esta ronda
        completados_ronda = 0
        for i in range(num_participantes):
            if faltantes_actuales[i] == 0 and resultados["ronda_completacion"][i] == -1:
                resultados["ronda_completacion"][i] = ronda_actual
                completados_ronda += 1
        resultados["historial_completados_ronda"].append(completados_ronda)

        # B2. Historial de faltantes (por participante)
        resultados["historial_faltantes"].append(faltantes_actuales.copy())

        # B3. Probabilidades analiticas por ronda
        total_repetidos_grupo = sum(
            sum(v for v in rep.values()) for rep in repetidos
        )
        probs_funda   = []
        probs_liquid  = []
        for i in range(num_participantes):
            if faltantes_actuales[i] > 0:
                pf, pl = prob_analitica_completar(
                    faltantes_actuales[i], total_repetidos_grupo
                )
                probs_funda.append(pf)
                probs_liquid.append(pl)
        resultados["historial_prob_analitica"].append(
            float(np.mean(probs_funda)) if probs_funda else 0.0
        )
        resultados["historial_indice_liquidez"].append(
            float(np.mean(probs_liquid)) if probs_liquid else 0.0
        )

        # C. Condicion de victoria
        if np.sum(faltantes_actuales) == 0:
            resultados["rondas_necesarias"] = ronda_actual
            break

        # D. Compras para la siguiente ronda
        for i in range(num_participantes):
            if faltantes_actuales[i] > 0:
                # Politica: minimo necesario, pero al menos 1
                fundas_a_comprar = max(1, math.ceil(faltantes_actuales[i] / CROMOS_POR_FUNDA))
                resultados["fundas_compradas"][i]         += fundas_a_comprar
                resultados["cromos_comprados_totales"][i] += fundas_a_comprar * CROMOS_POR_FUNDA

                nuevos_cromos = np.random.randint(0, N_CROMOS, fundas_a_comprar * CROMOS_POR_FUNDA)
                for cromo in nuevos_cromos:
                    if not albumes[i, cromo]:
                        albumes[i, cromo] = True
                    else:
                        repetidos[i][cromo] = repetidos[i].get(cromo, 0) + 1

        ronda_actual += 1

    if resultados["rondas_necesarias"] == 0:
        resultados["rondas_necesarias"] = ronda_actual - 1

    return resultados


# ==========================================
# MODULO 3 - MONTE CARLO
# ==========================================
@st.cache_data(show_spinner=False)
def simulacion_monte_carlo(num_participantes, estrategia, iteraciones=10):
    total_fundas       = []
    total_rondas       = []
    total_intercambios = []

    for _ in range(iteraciones):
        res = simular_ciclo_completo(num_participantes, estrategia)
        total_fundas.append(np.mean(res["fundas_compradas"]))
        total_rondas.append(res["rondas_necesarias"])
        total_intercambios.append(res["intercambios_totales"])

    n  = len(total_fundas)
    mu = np.mean(total_fundas)
    s  = np.std(total_fundas, ddof=1)

    # Intervalo de confianza al 95% con t-Student (gl = n-1).
    # Se prefiere t-Student sobre Z=1.96 porque el numero de iteraciones
    # Monte Carlo suele ser pequeno (< 50), donde la aproximacion normal
    # subestima la variabilidad real de las colas.
    if n > 1:
        t_crit = t_dist.ppf(0.975, df=n - 1)
        margen = t_crit * s / math.sqrt(n)
    else:
        margen = 0.0

    return {
        "fundas_promedio"      : mu,
        "fundas_std"           : s,
        "fundas_ic95_inf"      : mu - margen,
        "fundas_ic95_sup"      : mu + margen,
        "rondas_promedio"      : np.mean(total_rondas),
        "intercambios_promedio": np.mean(total_intercambios),
        "costo_promedio"       : mu * COSTO_FUNDA,
    }


# ==========================================
# UTILIDADES DE GRAFICACION
# ==========================================
def estilo_figura():
    """Aplica estilo oscuro consistente a todas las figuras."""
    plt.rcParams.update({
        "figure.facecolor"  : "#0F1117",
        "axes.facecolor"    : "#1A1D27",
        "axes.edgecolor"    : "#2E3245",
        "axes.labelcolor"   : "#C8CADE",
        "xtick.color"       : "#8A8FA8",
        "ytick.color"       : "#8A8FA8",
        "text.color"        : "#E0E2F0",
        "grid.color"        : "#2E3245",
        "grid.linestyle"    : "--",
        "grid.alpha"        : 0.6,
        "font.family"       : "monospace",
    })


def figura_faltantes_individual(historial_faltantes, num_participantes, estrategia):
    """
    Grafica de evolucion de faltantes por ronda.
    Muestra cada participante individualmente + promedio.
    """
    estilo_figura()
    fig, ax = plt.subplots(figsize=(11, 4.5))

    rondas = range(1, len(historial_faltantes) + 1)
    data   = np.array(historial_faltantes)   # shape: (rondas, participantes)

    # Lineas individuales (semitransparentes)
    colores_part = plt.cm.cool(np.linspace(0.15, 0.85, num_participantes))
    for i in range(num_participantes):
        ax.plot(rondas, data[:, i], color=colores_part[i],
                alpha=0.35, linewidth=0.9, zorder=1)

    # Promedio grupal
    promedio = data.mean(axis=1)
    ax.plot(rondas, promedio, color="#F0F050", linewidth=2.5,
            label="Promedio grupo", zorder=3)

    # Anotacion del minimo
    min_idx = np.argmin(promedio)
    ax.annotate(f"  min={promedio[min_idx]:.1f}",
                xy=(min_idx + 1, promedio[min_idx]),
                color="#F0F050", fontsize=9)

    ax.set_title(f"Evolucion de faltantes - {estrategia}", fontsize=13, pad=10)
    ax.set_xlabel("Ronda")
    ax.set_ylabel("Cromos faltantes")
    ax.legend(loc="upper right", fontsize=9,
              facecolor="#1A1D27", edgecolor="#2E3245")
    ax.grid(True)
    fig.tight_layout()
    return fig


def figura_probabilidades(historial_prob_analitica, historial_indice_liquidez,
                           historial_completados_ronda, num_participantes):
    """
    Panel de 2 subgraficas:
      1. P(funda util) analitica  vs  Indice de liquidez (heuristica)
      2. Probabilidad experimental acumulada de completar album
    """
    estilo_figura()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    rondas = range(1, len(historial_prob_analitica) + 1)

    # -- Sub-grafica 1: Analitica --
    ax1.plot(rondas, [p * 100 for p in historial_prob_analitica],
             color=COLOR_BILATERAL, linewidth=2, label="P(funda util) %")
    ax1.plot(rondas, [p * 100 for p in historial_indice_liquidez],
             color=COLOR_MULTILAT,  linewidth=2,
             linestyle="--", label="Indice de liquidez % (heuristica)")
    ax1.set_title("Indicadores analiticos por ronda", fontsize=12)
    ax1.set_xlabel("Ronda")
    ax1.set_ylabel("Valor (%)")
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=9, facecolor="#1A1D27", edgecolor="#2E3245")
    ax1.grid(True)

    # -- Sub-grafica 2: Experimental acumulada --
    # Construir historial: cuantos participantes han completado hasta cada ronda
    completados_acum = np.cumsum(historial_completados_ronda)
    prob_exp_acum    = completados_acum / num_participantes * 100

    ax2.fill_between(rondas, prob_exp_acum, alpha=0.25, color=COLOR_MULTILAT)
    ax2.plot(rondas, prob_exp_acum, color=COLOR_MULTILAT, linewidth=2,
             label="P experimental acumulada %")
    ax2.axhline(100, color="#666", linestyle="--", linewidth=0.8)
    ax2.set_title("P experimental de completar album", fontsize=12)
    ax2.set_xlabel("Ronda")
    ax2.set_ylabel("% participantes que completaron")
    ax2.set_ylim(-5, 110)
    ax2.legend(fontsize=9, facecolor="#1A1D27", edgecolor="#2E3245")
    ax2.grid(True)

    fig.tight_layout()
    return fig


def figura_heatmap_progreso(historial_faltantes, num_participantes):
    """
    Heatmap: eje X = rondas, eje Y = participante.
    Color = cromos faltantes (0 = verde = completado).
    """
    estilo_figura()
    data = np.array(historial_faltantes).T   # shape: (participantes, rondas)

    fig, ax = plt.subplots(figsize=(12, max(3, num_participantes * 0.45 + 1.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=N_CROMOS, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Cromos faltantes", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#8A8FA8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8A8FA8")

    ax.set_xlabel("Ronda")
    ax.set_ylabel("Participante #")
    ax.set_title("Progreso individual - Heatmap (verde = album completo)", fontsize=12)
    ax.set_yticks(range(num_participantes))
    ax.set_yticklabels([f"P{i+1}" for i in range(num_participantes)], fontsize=8)

    # Marcar la ronda en que cada participante completa
    for i in range(num_participantes):
        completado = np.argmax(data[i] == 0)
        if data[i, completado] == 0:
            ax.plot(completado, i, marker="*", color="gold", markersize=8, zorder=5)

    fig.tight_layout()
    return fig


def figura_ronda_completacion(ronda_completacion, estrategia):
    """Histograma de la ronda en que cada participante completa su album."""
    estilo_figura()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    validos = ronda_completacion[ronda_completacion > 0]
    if len(validos) == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        return fig

    bins = np.arange(validos.min(), validos.max() + 2) - 0.5
    ax.hist(validos, bins=bins, color=COLOR_MULTILAT, edgecolor="#0F1117", alpha=0.85)
    ax.axvline(validos.mean(), color=COLOR_BILATERAL, linewidth=2,
               linestyle="--", label=f"Media = {validos.mean():.1f}")
    ax.set_title(f"Distribucion de ronda de completacion - {estrategia}", fontsize=12)
    ax.set_xlabel("Ronda en que completa el album")
    ax.set_ylabel("Participantes")
    ax.legend(fontsize=9, facecolor="#1A1D27", edgecolor="#2E3245")
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig


def figura_comparativa_estrategias(datos_grafica, max_p):
    """Curva de fundas promedio vs participantes para las 3 estrategias."""
    estilo_figura()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(1, max_p + 1))

    colores    = [COLOR_NINGUNA, COLOR_BILATERAL, COLOR_MULTILAT]
    estrategias = ["Ninguna", "Solo Bilateral", "Triangular y Multilateral"]

    for est, color in zip(estrategias, colores):
        ax.plot(x, datos_grafica[est], color=color, linewidth=2.5,
                marker="o", markersize=3.5, label=est)

    ax.fill_between(x, datos_grafica["Ninguna"],
                    datos_grafica["Triangular y Multilateral"],
                    alpha=0.12, color=COLOR_MULTILAT, label="Zona de ahorro")

    ax.set_title("Fundas promedio necesarias vs numero de participantes", fontsize=13)
    ax.set_xlabel("Numero de participantes")
    ax.set_ylabel("Fundas promedio / persona")
    ax.legend(fontsize=10, facecolor="#1A1D27", edgecolor="#2E3245")
    ax.grid(True)
    fig.tight_layout()
    return fig


def figura_costo_comparativo(datos_grafica, max_p):
    """Grafica de barras agrupadas del costo promedio en USD por estrategia."""
    estilo_figura()
    fig, ax = plt.subplots(figsize=(11, 4.5))

    seleccion = [1, 5, 10, 20, 30, 50]
    seleccion = [s for s in seleccion if s <= max_p]
    x     = np.arange(len(seleccion))
    ancho = 0.25

    estrategias = ["Ninguna", "Solo Bilateral", "Triangular y Multilateral"]
    colores     = [COLOR_NINGUNA, COLOR_BILATERAL, COLOR_MULTILAT]

    for idx, (est, color) in enumerate(zip(estrategias, colores)):
        vals = [datos_grafica[est][s - 1] * COSTO_FUNDA for s in seleccion]
        bars = ax.bar(x + idx * ancho, vals, ancho, label=est,
                      color=color, alpha=0.85, edgecolor="#0F1117")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"${v:.0f}", ha="center", va="bottom", fontsize=7.5, color=color)

    ax.set_xticks(x + ancho)
    ax.set_xticklabels([f"n={s}" for s in seleccion])
    ax.set_title("Costo total por persona (USD) segun estrategia y grupo", fontsize=12)
    ax.set_xlabel("Numero de participantes")
    ax.set_ylabel("USD por persona")
    ax.legend(fontsize=9, facecolor="#1A1D27", edgecolor="#2E3245")
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig


# ==========================================
# INTERFAZ STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Simulador Album Mundial 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS personalizado --
st.markdown("""
<style>
    /* Fondo general */
    .main { background-color: #0F1117; }
    [data-testid="stAppViewContainer"] { background-color: #0F1117; }
    [data-testid="stSidebar"] { background-color: #13161F; }

    /* Tarjetas de metricas */
    [data-testid="metric-container"] {
        background: #1A1D27;
        border: 1px solid #2E3245;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"]  { color: #7DF9C0; font-size: 2rem !important; }
    [data-testid="stMetricLabel"]  { color: #8A8FA8 !important; }
    [data-testid="stMetricDelta"]  { color: #EF9F27 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #13161F; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]      { color: #8A8FA8 !important; }
    .stTabs [aria-selected="true"]    { color: #7DF9C0 !important; border-bottom: 2px solid #7DF9C0; }

    /* Titulos */
    h1, h2, h3 { color: #E0E2F0 !important; }

    /* Info / success boxes */
    .stAlert { border-radius: 8px; }

    /* Boton principal */
    .stButton > button {
        background: linear-gradient(135deg, #1D9E75 0%, #0F6E56 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover { filter: brightness(1.15); }
</style>
""", unsafe_allow_html=True)

# -- Header --
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1A1D27 0%, #0F1117 100%);
    border: 1px solid #2E3245;
    border-left: 4px solid #1D9E75;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
">
    <h1 style="margin:0; font-size:2rem; color:#E0E2F0; font-family:monospace;">
        ⚽ Simulador Estocastico - Album Mundial 2026
    </h1>
    <p style="margin:8px 0 0; color:#8A8FA8; font-size:0.9rem;">
        Modelo Coleccionista de Cupones · Teoria de Grafos · Monte Carlo · 980 cromos · 48 selecciones
    </p>
</div>
""", unsafe_allow_html=True)

# -- Sidebar --
st.sidebar.markdown("## ⚙️ Parametros")
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Constantes del album**
- 🃏 Cromos totales: `{N_CROMOS}`
- 📦 Cromos por funda: `{CROMOS_POR_FUNDA}`
- 💵 Costo por funda: `${COSTO_FUNDA}`
- 🎯 Fundas minimas teoricas: `{FUNDAS_INICIALES}`
- 💰 Inversion minima teorica: `${FUNDAS_INICIALES * COSTO_FUNDA:.2f}`
""")
st.sidebar.markdown("---")

poblacion          = st.sidebar.slider("👥 Participantes (simulacion unica)", 1, 50, 10, key="slider_pob")
estrategia_elegida = st.sidebar.selectbox(
    "🔀 Estrategia de intercambio",
    ["Ninguna", "Solo Bilateral", "Triangular y Multilateral"],
    key="sel_est"
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analisis comparativo")
max_participantes_analisis = st.sidebar.slider("Max. participantes (analisis)", 2, 50, 20)
iteraciones_mc             = st.sidebar.slider("Iteraciones Monte Carlo", 5, 50, 10,
                                                help="Mas iteraciones = mas precision, mas tiempo")

# ==========================================
# PESTANAS PRINCIPALES
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "🔬 Simulacion individual (ronda a ronda)",
    "📊 Analisis comparativo (Monte Carlo)",
    "📐 Teoria y formulas",
])

# ══════════════════════════════════════════
# PESTANA 1 - SIMULACION INDIVIDUAL
# ══════════════════════════════════════════
with tab1:
    st.subheader(f"Simulacion con {poblacion} participante(s) - estrategia: {estrategia_elegida}")

    if st.button("▶ Ejecutar simulacion unica", key="btn_sim"):
        with st.spinner("Simulando ciclo de vida completo..."):
            res = simular_ciclo_completo(poblacion, estrategia_elegida)

        # -- Metricas principales --
        fundas_prom  = float(np.mean(res["fundas_compradas"]))
        costo_prom   = fundas_prom * COSTO_FUNDA
        ahorro_vs_sin = (fundas_prom - (res["fundas_compradas"].max())) * COSTO_FUNDA

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Fundas promedio/persona",   f"{fundas_prom:.1f}")
        col2.metric("Costo promedio/persona",    f"${costo_prom:.2f}")
        col3.metric("Rondas necesarias",          res["rondas_necesarias"])
        col4.metric("Intercambios realizados",    res["intercambios_totales"])
        col5.metric("Cromos intercambiados/ronda",
                    f"{res['intercambios_totales'] / max(1, res['rondas_necesarias']):.1f}")

        st.markdown("---")

        # -- Probabilidades analiticas de la ronda 1 --
        if len(res["historial_faltantes"]) > 0:
            falt_promedio_r1 = float(np.mean(res["historial_faltantes"][0]))
            
            # Extraemos los datos calculados reales (sin diccionarios vacios)
            pf = res["historial_prob_analitica"][0]
            pi = res["historial_indice_liquidez"][0]
            
            fund_esp = prob_analitica_completar_exacto(int(falt_promedio_r1))

            c1, c2, c3 = st.columns(3)
            c1.info(f"**P(funda util) en Ronda 1**\n\n`{pf*100:.2f}%`\n\nProb. de que una funda comprada aporte al menos 1 cromo nuevo")
            c2.info(f"**Indice de liquidez en Ronda 1**\n\n`{pi*100:.2f}%`\n\nIndicador heuristico: ¿tiene el grupo repetidos suficientes para cubrir los faltantes? (cota superior optimista)")
            c3.info(f"**Fundas esperadas (coleccionista)**\n\n`{fund_esp}`\n\nEstimado teorico para completar desde faltantes iniciales")

        st.markdown("---")

        # -- Grafica 1: Evolucion de faltantes por participante --
        st.subheader("1. Progreso individual por ronda")
        if len(res["historial_faltantes"]) > 0:
            fig1 = figura_faltantes_individual(
                res["historial_faltantes"], poblacion, estrategia_elegida
            )
            st.pyplot(fig1)
            plt.close(fig1)

        # -- Grafica 2: Probabilidades por ronda --
        st.subheader("2. Probabilidades analitica y experimental por ronda")
        if len(res["historial_prob_analitica"]) > 0:
            fig2 = figura_probabilidades(
                res["historial_prob_analitica"],
                res["historial_indice_liquidez"],
                res["historial_completados_ronda"],
                poblacion,
            )
            st.pyplot(fig2)
            plt.close(fig2)

        # -- Grafica 3: Heatmap de progreso individual --
        st.subheader("3. Heatmap de progreso - cada participante, cada ronda")
        if len(res["historial_faltantes"]) > 0 and poblacion > 1:
            fig3 = figura_heatmap_progreso(res["historial_faltantes"], poblacion)
            st.pyplot(fig3)
            plt.close(fig3)
        elif poblacion == 1:
            st.info("El heatmap aplica para 2 o mas participantes.")

        # -- Grafica 4: Histograma de ronda de completacion --
        st.subheader("4. Distribucion de ronda en que cada participante completa")
        if poblacion > 1:
            fig4 = figura_ronda_completacion(res["ronda_completacion"], estrategia_elegida)
            st.pyplot(fig4)
            plt.close(fig4)

        # -- Tabla individual de resultados --
        st.subheader("5. Tabla de resultados por participante")
        df_individual = pd.DataFrame({
            "Participante"   : [f"P{i+1}" for i in range(poblacion)],
            "Fundas compradas": res["fundas_compradas"],
            "Costo (USD)"    : (res["fundas_compradas"] * COSTO_FUNDA).round(2),
            "Cromos comprados": res["cromos_comprados_totales"],
            "Ronda completacion": [
                r if r > 0 else "No completo"
                for r in res["ronda_completacion"]
            ],
        })
        st.dataframe(df_individual, use_container_width=True)

        # -- Probabilidad experimental acumulada por ronda (tabla) --
        st.subheader("6. Probabilidad experimental de completar album (por ronda)")
        probs_acum = prob_experimental_completar(res["historial_completados_ronda"], poblacion)
        df_prob = pd.DataFrame({
            "Ronda"                              : list(range(1, len(probs_acum) + 1)),
            "Completaron esta ronda"             : res["historial_completados_ronda"],
            "P experimental acumulada (%)"       : [f"{p*100:.1f}" for p in probs_acum],
            "P analitica funda util (%)"         : [
                f"{p*100:.2f}" for p in res["historial_prob_analitica"]
            ],
            "Indice de liquidez (heuristica, %)": [
                f"{p*100:.2f}" for p in res["historial_indice_liquidez"]
            ],
        })
        st.dataframe(df_prob, use_container_width=True)


# ══════════════════════════════════════════
# PESTANA 2 - ANALSIS COMPARATIVO
# ══════════════════════════════════════════
with tab2:
    st.header("Analisis poblacional comparativo (Monte Carlo)")
    st.markdown(
        "Compara el impacto de las 3 estrategias de intercambio para grupos de 1 a N participantes. "
        "Cada punto es el promedio de multiples simulaciones."
    )

    if st.button("🚀 Iniciar analisis comparativo", key="btn_mc"):
        progreso = st.progress(0)
        estado   = st.empty()

        datos_grafica = {
            "Participantes"              : list(range(1, max_participantes_analisis + 1)),
            "Ninguna"                    : [],
            "Solo Bilateral"             : [],
            "Triangular y Multilateral"  : [],
        }

        total_pasos = max_participantes_analisis * 3
        paso_actual = 0

        for estrategia in ["Ninguna", "Solo Bilateral", "Triangular y Multilateral"]:
            for p in range(1, max_participantes_analisis + 1):
                estado.text(f"Calculando: {estrategia} - {p}/{max_participantes_analisis} participantes...")
                res_mc = simulacion_monte_carlo(p, estrategia, iteraciones=iteraciones_mc)
                datos_grafica[estrategia].append(res_mc["fundas_promedio"])
                paso_actual += 1
                progreso.progress(paso_actual / total_pasos)

        estado.empty()
        progreso.empty()

        # -- Grafica comparativa de fundas --
        st.subheader("1. Fundas promedio por persona vs numero de participantes")
        fig_comp = figura_comparativa_estrategias(datos_grafica, max_participantes_analisis)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        # -- Grafica de costo en USD --
        st.subheader("2. Costo total por persona (USD) - puntos de comparacion")
        fig_costo = figura_costo_comparativo(datos_grafica, max_participantes_analisis)
        st.pyplot(fig_costo)
        plt.close(fig_costo)

        # -- Conclusion --
        ahorro_fundas   = datos_grafica["Ninguna"][-1] - datos_grafica["Triangular y Multilateral"][-1]
        ahorro_usd      = ahorro_fundas * COSTO_FUNDA
        pct_ahorro      = (ahorro_fundas / datos_grafica["Ninguna"][-1]) * 100

        ahorro_bilateral = datos_grafica["Ninguna"][-1] - datos_grafica["Solo Bilateral"][-1]
        ahorro_usd_bi    = ahorro_bilateral * COSTO_FUNDA

        st.markdown("---")
        st.success(f"""
### 🏆 Conclusion cientifica del analisis

Para un grupo de **{max_participantes_analisis} personas**:

| Estrategia              | Fundas promedio | Costo promedio | Ahorro vs sin intercambio |
|-------------------------|-----------------|---------------|--------------------------|
| Sin intercambio         | {datos_grafica['Ninguna'][-1]:.1f}          | ${datos_grafica['Ninguna'][-1]*COSTO_FUNDA:.2f}         | —                        |
| Solo bilateral          | {datos_grafica['Solo Bilateral'][-1]:.1f}          | ${datos_grafica['Solo Bilateral'][-1]*COSTO_FUNDA:.2f}         | **${ahorro_usd_bi:.2f}** ({ahorro_bilateral/datos_grafica['Ninguna'][-1]*100:.1f}%)  |
| Multilateral (ciclos)   | {datos_grafica['Triangular y Multilateral'][-1]:.1f}          | ${datos_grafica['Triangular y Multilateral'][-1]*COSTO_FUNDA:.2f}         | **${ahorro_usd:.2f}** ({pct_ahorro:.1f}%)   |

El algoritmo multilateral explota oportunidades de intercambio en ciclos de longitud >= 2 que el bilateral no detecta. No se garantiza optimalidad global (es una estrategia voraz sobre el primer ciclo encontrado), pero produce consistentemente mayor ahorro que el intercambio bilateral en las simulaciones.
""")

        # -- Tabla de datos completa --
        st.subheader("3. Tabla de datos completa")
        df_full = pd.DataFrame(datos_grafica).set_index("Participantes")
        df_full.columns = ["Sin intercambio (fundas)", "Bilateral (fundas)", "Multilateral (fundas)"]
        
        # Aqui le agregamos el Costo USD
        df_full["Sin intercambio ($)"] = (df_full["Sin intercambio (fundas)"] * COSTO_FUNDA).round(2)
        df_full["Bilateral ($)"]       = (df_full["Bilateral (fundas)"]        * COSTO_FUNDA).round(2)
        df_full["Multilateral ($)"]    = (df_full["Multilateral (fundas)"]     * COSTO_FUNDA).round(2)
        df_full["Ahorro multilateral ($)"] = (df_full["Sin intercambio ($)"] - df_full["Multilateral ($)"]).round(2)
        st.dataframe(df_full.round(2), use_container_width=True)


# ══════════════════════════════════════════
# PESTANA 3 - TEORIA Y FORMULAS
# ══════════════════════════════════════════
with tab3:
    st.header("📐 Marco teorico, supuestos y limitaciones del modelo")

    st.markdown(r"""
### 1. Supuestos del modelo

> Declarar los supuestos es parte esencial de cualquier modelo de simulacion.
> Los resultados son validos **dentro del marco de estos supuestos**, no fuera de el.

| # | Supuesto | Justificacion en el modelo |
|---|----------|---------------------------|
| S1 | Los 980 cromos tienen la **misma probabilidad** de aparecer en cualquier funda | Distribucion uniforme discreta sobre {0, ..., 979} |
| S2 | Los sobres son **independientes** entre si y entre participantes | Muestras i.i.d. con reposicion |
| S3 | Los intercambios ocurren **siempre que existe la oportunidad matematica** | No se modela preferencia personal ni rechazo |
| S4 | No existen **cromos especiales, foil ni distribucion desigual** | Un solo tipo de cromo en el espacio muestral |
| S5 | La **politica de compra** es `faltantes/7` fundas por ronda | Estrategia de decision definida por el modelo; no pretende reproducir el comportamiento de un coleccionista real |
| S6 | La poblacion es **cerrada**: solo se intercambia entre los participantes de la simulacion | No hay mercado externo ni cromos adicionales |

---

### 2. Modelo del Coleccionista de Cupones (Coupon Collector's Problem)

El numero esperado de cromos que hay que comprar para completar un album de $n$ cromos
cuando cada compra es uniformemente aleatoria es:

$$
E[T] = n \cdot H_n = n \sum_{k=1}^{n} \frac{1}{k} \approx n \ln(n) + \gamma n
$$

donde $H_n$ es el $n$-esimo numero armonico y $\gamma \approx 0.5772$ es la constante de Euler-Mascheroni.

Para el album del Mundial 2026 con $n = 980$ cromos:

$$
E[T_{980}] \approx 980 \ln(980) + 0.5772 \cdot 980 \approx 7{,}082 \text{ cromos} \approx 1{,}012 \text{ fundas}
$$

Este es el **caso de referencia** (un solo coleccionista, sin intercambios, bajo el supuesto S1).

---

### 3. Probabilidad analitica - funda util por ronda

En la ronda $r$, si a un participante le faltan $f$ cromos:

$$
P(\text{funda util}) = 1 - \left(1 - \frac{f}{n}\right)^k
$$

donde $k = 7$ es el numero de cromos por funda. Esta es la probabilidad de que al menos
uno de los $k$ cromos sea uno de los $f$ que aun le faltan, **bajo los supuestos S1 y S2**.

---

### 4. Indice de liquidez del grupo (indicador heuristico)

Si el grupo tiene $R_{total}$ cromos repetidos disponibles y a un participante le faltan $f$:

$$
\text{Indice de liquidez} = \min\left(1,\ \frac{R_{total}}{f}\right)
$$

> ⚠️ **Esto NO es una probabilidad formal.** Es una cota superior optimista que asume
> que todos los repetidos del grupo son distintos y utiles para este participante.
> En la practica, si 500 repetidos son del mismo cromo y solo te faltan 5,
> el indice daría 100 % aunque en realidad no completes el album.
> Se presenta como **indicador de disponibilidad relativa**, no como probabilidad de completacion.

---

### 5. Probabilidad experimental de completar por ronda

Para $N$ participantes y en la ronda $r$:

$$
\hat{P}_{exp}(r) = \frac{\sum_{i=1}^{r} C_i}{N}
$$

donde $C_i$ es el numero de participantes que completaron en la ronda $i$.
Esta es la **funcion de distribucion empirica (CDF)** del tiempo de completacion.
Converge a la CDF real a medida que $N \to \infty$.

---

### 6. Algoritmo de intercambio multilateral (Teoria de Grafos)

Se construye un **grafo dirigido** $G = (V, E)$ donde:
- $V$ = participantes
- $(a \to b) \in E$ si $a$ tiene un cromo repetido que le falta a $b$

Los intercambios se realizan encontrando **ciclos** en $G$:

$$
\text{ciclo: } p_1 \to p_2 \to \cdots \to p_k \to p_1
$$

Cada nodo del ciclo entrega un cromo al siguiente. Esto permite intercambios
triangulares ($k=3$) y de mayor longitud que el bilateral ($k=2$) no detectaria.

> ⚠️ **El algoritmo NO garantiza optimalidad global.** Usa `nx.find_cycle()`,
> que devuelve el **primer ciclo encontrado** (estrategia voraz / *greedy*).
> Un algoritmo de flujo maximo o *matching* maximo podria resolver mas intercambios
> por ronda, pero su complejidad computacional es mayor.
> La afirmacion correcta es: **"el algoritmo explota oportunidades multilaterales
> de intercambio de manera eficiente"**, no que las maximice.

La complejidad de deteccion de ciclos con DFS es $O(|V| + |E|)$ por ciclo encontrado.

---

### 7. Politica de compra por ronda

Despues de cada ronda de intercambios, quien le falten $f > 0$ cromos compra:

$$
N_{adicional} = \left\lceil \frac{f}{k} \right\rceil \text{ fundas}
$$

> ⚠️ **Limitacion:** Esta politica es una decesion de diseno del modelo (supuesto S5).
> Asume que el participante compra exactamente lo mínimo necesario en el escenario ideal
> (sin repetidos). En la practica, con $f$ pequeno y $n=980$, la probabilidad de que
> esa unica funda traiga el cromo exacto es baja, lo que hace que las rondas finales
> sean muchas con pocas compras. Esto es un artefacto de la politica, no del problema real.

---

### 8. Intervalo de confianza Monte Carlo

Para las estimaciones del promedio de fundas sobre $m$ iteraciones:

$$
\bar{X} \pm t_{\alpha/2,\, m-1} \cdot \frac{s}{\sqrt{m}}
$$

Se utiliza la distribucion **$t$-Student con $m-1$ grados de libertad** en lugar de $Z = 1.96$,
porque con pocas iteraciones Monte Carlo ($m < 50$) la aproximacion normal subestima
la variabilidad real de las colas. Usar $t$-Student es formalmente mas correcto
para muestras pequenas sin conocimiento de $\sigma$ poblacional.

---

### 9. Limitaciones generales del modelo

1. **No se modela escasez real**: Panini podria distribuir algunos cromos con menor frecuencia por razones comerciales (supuesto S1 podria no cumplirse en la practica).
2. **No se modela comportamiento humano**: coleccionistas reales pueden no intercambiar, pedir algo a cambio, o comprar en lote.
3. **El algoritmo greedy no es optimo**: un solver de flujo maximo podria liberar mas intercambios por ronda.
4. **La politica de compra es artificial**: ver supuesto S5.
5. **Poblacion cerrada**: no considera el mercado secundario real (redes sociales, ferias de intercambio).
""")

    # -- Tabla de referencia de valores esperados --
    st.markdown("---")
    st.subheader("Tabla de referencia - Fundas esperadas segun coleccionista de cupones")

    rows = []
    for f in [980, 500, 200, 100, 50, 20, 10, 5, 1]:
        fe = prob_analitica_completar_exacto(f)
        rows.append({
            "Cromos faltantes": f,
            "Cromos obtenidos": N_CROMOS - f,
            "Fundas esperadas adicionales (teorico)": fe,
            "Costo adicional estimado (USD)": f"${fe * COSTO_FUNDA:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)