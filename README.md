# Simulador Album Mundial 2026

Proyecto de simulación estocástica para completar el álbum del Mundial 2026.
La aplicación usa Streamlit para analizar el impacto de distintos esquemas de intercambio entre coleccionistas.

## Descripción

El código modela un conjunto de participantes que compran cromos de forma aleatoria y luego realizan intercambios según tres estrategias:

- `Ninguna`: no hay intercambios.
- `Solo Bilateral`: solo se aceptan intercambios directos entre dos participantes.
- `Triangular y Multilateral`: se buscan ciclos de intercambio de longitud ≥ 2 en un grafo dirigido.

El objetivo del modelo es comparar la cantidad de fundas necesarias y el ahorro potencial en función de la estrategia de intercambio.

## Archivos principales

- `app.py`: aplicación principal de Streamlit.
- `requirements.txt`: dependencias del proyecto.
- `analisis_mundial.ipynb`: cuaderno Jupyter adicional (opcional para análisis).

## Dependencias

El proyecto usa:

- `numpy`
- `pandas`
- `matplotlib`
- `networkx`
- `streamlit`

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecución

Para iniciar la aplicación Streamlit:

```bash
streamlit run app.py
```

Luego abre el enlace local que muestra Streamlit en tu navegador.

## Funcionalidad principal

### Módulos de `app.py`

1. **Motor de intercambios**
   - Construye un grafo de oferta entre participantes.
   - Ejecuta intercambios bilaterales o ciclos multilaterales.

2. **Probabilidad analítica**
   - Calcula la probabilidad de que una funda contenga al menos un cromo faltante.
   - Estima un índice heurístico de liquidez de intercambios.
   - Calcula un estimado de fundas esperadas para completar un álbum con la aproximación del coleccionista de cupones.

3. **Simulación completa**
   - Simula rondas de compra e intercambio hasta que todos los participantes completan el álbum o se agota el máximo de rondas.
   - Registra métricas por participante, rondas y estado del grupo.

4. **Monte Carlo**
   - Ejecuta múltiples simulaciones para estimar promedios, desviaciones estándar e intervalos de confianza de fundas y costos.

5. **Visualización**
   - Genera gráficos de evolución de faltantes, probabilidades analíticas, heatmaps de progreso y comparativas de estrategias.

## Supuestos del modelo

- 980 cromos distintos y distribución uniforme de cromos en cada funda.
- Cada funda contiene 7 cromos.
- Las compras son independientes.
- Los intercambios se realizan siempre cuando hay oportunidad.
- Los participantes compran la cantidad mínima de fundas necesaria cada ronda.
- Población cerrada: solo se intercambia dentro del grupo simulado.

## Notas importantes

- El algoritmo multilateral usa `networkx.find_cycle()` y no garantiza optimalidad global.
- El índice de liquidez es heurístico y no representa una probabilidad formal.
- La política de compra de `ceil(faltantes / 7)` puede ser optimista en las rondas finales.

## Estructura de la interfaz

La aplicación Streamlit tiene tres pestañas principales:

- `Simulación individual (ronda a ronda)`.
- `Análisis comparativo (Monte Carlo)`.
- `Teoría y fórmulas`.

## Recomendación

Si deseas mejorar el modelo, considera:

- Implementar un algoritmo de matching o flujo máximo para maximizar intercambios.
- Introducir distribuciones no uniformes de cromos.
- Modelar preferencias y rechazos en intercambios.
- Ajustar la política de compra final para evitar rondas con pocas fundas.
