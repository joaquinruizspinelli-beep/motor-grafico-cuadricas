import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Configuración general de la página
st.set_page_config(
    page_title="Simulador de Cuádricas - Álgebra y Geometría Analítica",
    layout="wide"
)

st.title("Motor Gráfico de Superficies Cuádricas en Coordenadas Homogéneas")
st.markdown("Visualización dinámica de transformaciones afines ($S, R, T$) y proyecciones de $\mathbb{R}^3 \to \mathbb{R}^2$.")

# --- BARRA LATERAL: CONTROLES INTERACTIVOS ---
st.sidebar.header("1. Selección de Superficie")
tipo_cuadrica = st.sidebar.selectbox(
    "Superficie Cuádrica:",
    [
        "Hiperboloide de 1 Hoja",
        "Elipsoide",
        "Paraboloide Elíptico",
        "Paraboloide Hiperbólico (Silla de montar)",
        "Hiperboloide de 2 Hojas"
    ]
)

st.sidebar.header("2. Transformaciones Afines")
# Rotaciones
rx_deg = st.sidebar.slider("Rotación eje X (θx)", -180, 180, 50)
ry_deg = st.sidebar.slider("Rotación eje Y (θy)", -180, 180, 25)
rz_deg = st.sidebar.slider("Rotación eje Z (θz)", -180, 180, 0)

# Escalado
sx = st.sidebar.slider("Escalado X (sx)", 0.3, 2.5, 1.0, 0.1)
sy = st.sidebar.slider("Escalado Y (sy)", 0.3, 2.5, 1.0, 0.1)
sz = st.sidebar.slider("Escalado Z (sz)", 0.3, 2.5, 1.0, 0.1)

# Traslación
tx = st.sidebar.slider("Traslación X (tx)", -5.0, 5.0, 0.0, 0.5)
ty = st.sidebar.slider("Traslación Y (ty)", -5.0, 5.0, 0.0, 0.5)
tz = st.sidebar.slider("Traslación Z / Profundidad (tz)", 4.0, 18.0, 8.0, 0.5)

st.sidebar.header("3. Parámetros de Cámara")
d = st.sidebar.slider("Distancia Focal (d)", 1.0, 10.0, 5.0, 0.5)
modo_visualizacion = st.sidebar.radio("Estilo de renderizado:", ["Malla Alámbrica (Wireframe)", "Nube de Puntos"])

# --- GENERACIÓN DE PUNTOS PARAMÉTRICOS ---
def generar_malla(tipo, n_u=28, n_v=36):
    u = np.linspace(-1.2, 1.2, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v)

    if tipo == "Hiperboloide de 1 Hoja":
        X = np.cosh(U) * np.cos(V)
        Y = np.cosh(U) * np.sin(V)
        Z = 1.2 * np.sinh(U)
    elif tipo == "Elipsoide":
        phi = np.linspace(0, np.pi, n_u)
        PHI, THETA = np.meshgrid(phi, v)
        X = 1.5 * np.sin(PHI) * np.cos(THETA)
        Y = 1.0 * np.sin(PHI) * np.sin(THETA)
        Z = 1.2 * np.cos(PHI)
    elif tipo == "Paraboloide Elíptico":
        r = np.linspace(0, 1.6, n_u)
        R, THETA = np.meshgrid(r, v)
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)
        Z = (R**2) - 1.0
    elif tipo == "Paraboloide Hiperbólico (Silla de montar)":
        x_lin = np.linspace(-1.4, 1.4, n_u)
        y_lin = np.linspace(-1.4, 1.4, n_v)
        X, Y = np.meshgrid(x_lin, y_lin)
        Z = 0.6 * (X**2 - Y**2)
    elif tipo == "Hiperboloide de 2 Hojas":
        u_pos = np.linspace(0.2, 1.3, n_u // 2)
        U2, V2 = np.meshgrid(u_pos, v)
        X_top = np.sinh(U2) * np.cos(V2)
        Y_top = np.sinh(U2) * np.sin(V2)
        Z_top = 1.2 * np.cosh(U2)
        # Combinamos ambas hojas separadas por NaN para que no se unan líneas
        nan_row = np.full((n_v, 1), np.nan)
        X = np.hstack([X_top, nan_row, X_top])
        Y = np.hstack([Y_top, nan_row, Y_top])
        Z = np.hstack([Z_top, nan_row, -Z_top])

    return X, Y, Z

X, Y, Z = generar_malla(tipo_cuadrica)
shape_orig = X.shape

# Vectorización en coordenadas homogéneas (4 x N)
ones = np.ones_like(X)
puntos_hom = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), ones.ravel()])

# --- CONSTRUCCIÓN DE MATRICES HOMOGÉNEAS (4x4) ---
rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)

S = np.array([
    [sx, 0,  0,  0],
    [0,  sy, 0,  0],
    [0,  0,  sz, 0],
    [0,  0,  0,  1]
])

Rx = np.array([
    [1, 0,           0,          0],
    [0, np.cos(rx), -np.sin(rx), 0],
    [0, np.sin(rx),  np.cos(rx), 0],
    [0, 0,           0,          1]
])

Ry = np.array([
    [ np.cos(ry), 0, np.sin(ry), 0],
    [ 0,          1, 0,          0],
    [-np.sin(ry), 0, np.cos(ry), 0],
    [ 0,          0, 0,          1]
])

Rz = np.array([
    [np.cos(rz), -np.sin(rz), 0, 0],
    [np.sin(rz),  np.cos(rz), 0, 0],
    [0,           0,          1, 0],
    [0,           0,          0, 1]
])

T = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]
])

# Matriz compuesta M = T * Rz * Ry * Rx * S
M = T @ Rz @ Ry @ Rx @ S
puntos_trans = M @ puntos_hom

Xt = puntos_trans[0].reshape(shape_orig)
Yt = puntos_trans[1].reshape(shape_orig)
Zt = puntos_trans[2].reshape(shape_orig)

# --- PROYECCIONES ---
# 1. Ortogonal
P_ort = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
])
proy_ort = P_ort @ puntos_trans
X_ort = proy_ort[0].reshape(shape_orig)
Y_ort = proy_ort[1].reshape(shape_orig)

# 2. Perspectiva
P_persp = np.array([
    [d, 0, 0, 0],
    [0, d, 0, 0],
    [0, 0, d, 0],
    [0, 0, 1, 0]
])
proy_persp_hom = P_persp @ puntos_trans
w = proy_persp_hom[3]
X_persp = (proy_persp_hom[0] / w).reshape(shape_orig)
Y_persp = (proy_persp_hom[1] / w).reshape(shape_orig)

# --- FUNCIÓN AUXILIAR PARA DIBUJAR MALLAS EN PLOTLY ---
def crear_trazas_wireframe(X_mat, Y_mat, Z_mat=None, color="#1f77b4", es_3d=True):
    trazas = []
    # Líneas en dirección u
    for i in range(X_mat.shape[0]):
        if es_3d:
            trazas.append(go.Scatter3d(
                x=X_mat[i, :], y=Y_mat[i, :], z=Z_mat[i, :],
                mode='lines', line=dict(color=color, width=2), showlegend=False
            ))
        else:
            trazas.append(go.Scatter(
                x=X_mat[i, :], y=Y_mat[i, :],
                mode='lines', line=dict(color=color, width=1.2), showlegend=False
            ))
    # Líneas en dirección v
    for j in range(X_mat.shape[1]):
        if es_3d:
            trazas.append(go.Scatter3d(
                x=X_mat[:, j], y=Y_mat[:, j], z=Z_mat[:, j],
                mode='lines', line=dict(color=color, width=2), showlegend=False
            ))
        else:
            trazas.append(go.Scatter(
                x=X_mat[:, j], y=Y_mat[:, j],
                mode='lines', line=dict(color=color, width=1.2), showlegend=False
            ))
    return trazas

# --- VISUALIZACIÓN EN COLUMNAS ---
col_mat, col_3d = st.columns([1, 2])

with col_mat:
    st.subheader("Matriz Compuesta $M$")
    st.markdown("$M = T \cdot R_z \cdot R_y \cdot R_x \cdot S$")
    st.dataframe(np.round(M, 3), use_container_width=True)
    
    st.subheader("Matriz Perspectiva $P$")
    st.dataframe(np.round(P_persp, 2), use_container_width=True)

with col_3d:
    st.subheader("1. Espacio Transformado ($\mathbb{R}^3$)")
    fig_3d = go.Figure()
    if modo_visualizacion == "Malla Alámbrica (Wireframe)":
        for traza in crear_trazas_wireframe(Xt, Yt, Zt, color="#2563eb", es_3d=True):
            fig_3d.add_trace(traza)
    else:
        fig_3d.add_trace(go.Scatter3d(
            x=Xt.ravel(), y=Yt.ravel(), z=Zt.ravel(),
            mode='markers', marker=dict(size=3, color=Zt.ravel(), colorscale='Viridis')
        ))
    fig_3d.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=10), height=400
    )
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

col_ort, col_persp = st.columns(2)

with col_ort:
    st.subheader("2. Proyección Ortogonal ($\mathbb{R}^2$)")
    fig_ort = go.Figure()
    if modo_visualizacion == "Malla Alámbrica (Wireframe)":
        for traza in crear_trazas_wireframe(X_ort, Y_ort, color="#16a34a", es_3d=False):
            fig_ort.add_trace(traza)
    else:
        fig_ort.add_trace(go.Scatter(
            x=X_ort.ravel(), y=Y_ort.ravel(),
            mode='markers', marker=dict(size=4, color='#16a34a')
        ))
    fig_ort.update_layout(
        xaxis_title='xp', yaxis_title='yp',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, b=20, t=20), height=380
    )
    st.plotly_chart(fig_ort, use_container_width=True)

with col_persp:
   st.subheader(f"3. Proyección Perspectiva ($\mathbb{{R}}^2$, d={d})")
    fig_persp = go.Figure()
    if modo_visualizacion == "Malla Alámbrica (Wireframe)":
        for traza in crear_trazas_wireframe(X_persp, Y_persp, color="#dc2626", es_3d=False):
            fig_persp.add_trace(traza)
    else:
        fig_persp.add_trace(go.Scatter(
            x=X_persp.ravel(), y=Y_persp.ravel(),
            mode='markers', marker=dict(size=4, color='#dc2626')
        ))
    fig_persp.update_layout(
        xaxis_title='xp = dx/z', yaxis_title='yp = dy/z',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, b=20, t=20), height=380
    )
    st.plotly_chart(fig_persp, use_container_width=True)
