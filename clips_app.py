import streamlit as st
import requests
import os
import json
import pandas as pd
import time
import io
import cloudinary
import cloudinary.uploader
import cloudinary.utils

# ── CONFIGURACIÓN DE SECRETOS ──
try:
    CLIENT_ID     = st.secrets["CLIENT_ID"]
    CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
except KeyError:
    st.error("Faltan las credenciales en los Secretos (CLIENT_ID o CLIENT_SECRET)")
    st.stop()

REDIRECT_URI  = "https://httpbin.org/get"
ML_SCOPES     = "offline_access read_listings write_listings"

# Archivos de estado (únicos para no pisar las otras apps)
TOKEN_FILE    = "ml_token_clips.json"
ITEMS_FILE    = "items_clips.json"
STEP_FILE     = "step_clips.json"
RESULTS_FILE  = "results_clips.json"
CLOUD_FILE    = "cloud_config.json"
UPLOAD_FILE   = "upload_clips.json"

st.set_page_config(page_title="Generador de Clips ML", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.main-title{font-size:28px;font-weight:600;margin-bottom:4px}
.sub-title{font-size:15px;color:#888;margin-bottom:24px}
.step-box{border-radius:12px;padding:14px 18px;margin-bottom:10px;border-left:4px solid #444;background:#1e1e1e}
.step-done{border-left-color:#00A650!important;background:#0a1f0f!important}
.step-active{border-left-color:#3483FA!important;background:#0a1020!important}
.video-card{border-radius:10px;padding:12px 16px;margin-bottom:8px;background:#1a1a2e;border:1px solid #2d2d4e}
.video-ok{border-color:#00A650!important}
.video-err{border-color:#e74c3c!important}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──
def save_json(file, data):
    with open(file, "w") as f: json.dump(data, f)

def load_json(file, default):
    if os.path.exists(file):
        with open(file) as f: return json.load(f)
    return default

def get_token():
    if st.session_state.get("token"): return st.session_state.token
    saved = load_json(TOKEN_FILE, {})
    if "access_token" in saved:
        if (time.time() - saved.get("saved_at", 0)) < (saved.get("expires_in", 21600) - 3600):
            st.session_state.token = saved["access_token"]
            return saved["access_token"]
        if "refresh_token" in saved:
            r = requests.post("https://api.mercadolibre.com/oauth/token", data={
                "grant_type":"refresh_token","client_id":CLIENT_ID,
                "client_secret":CLIENT_SECRET,"refresh_token":saved["refresh_token"]})
            if r.status_code == 200:
                d = r.json(); d["saved_at"] = time.time()
                save_json(TOKEN_FILE, d)
                st.session_state.token = d["access_token"]
                return d["access_token"]
    return None

def obtener_token_code(code):
    r = requests.post("https://api.mercadolibre.com/oauth/token", data={
        "grant_type":"authorization_code","client_id":CLIENT_ID,
        "client_secret":CLIENT_SECRET,"code":code,"redirect_uri":REDIRECT_URI})
    if r.status_code == 200:
        d = r.json(); d["saved_at"] = time.time()
        save_json(TOKEN_FILE, d)
        return d["access_token"]
    return None

def configurar_cloudinary(cloud_name, api_key, api_secret):
    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)

def obtener_url_imagen_ml(item_id, token):
    """Obtiene la URL de la imagen de portada de una publicación ML."""
    r = requests.get(
        f"https://api.mercadolibre.com/items/{item_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10
    )
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code} al consultar {item_id}")
    pics = r.json().get("pictures", [])
    if not pics:
        raise Exception(f"Sin imágenes en {item_id}")
    # Usar la URL de mayor resolución disponible
    url = pics[0].get("secure_url") or pics[0].get("url")
    if not url:
        raise Exception(f"URL de imagen vacía en {item_id}")
    return url

def generar_clip_cloudinary(item_id, img_url, audio_public_id, fondo):
    """
    Genera el video MP4 localmente con Python (sin depender de transformaciones
    de Cloudinary) y sube el archivo final.

    Flujo:
    1. Descarga la imagen de ML.
    2. Crea frames con efecto Ken Burns (zoom-in suave) usando Pillow + numpy.
    3. Codifica el MP4 con imageio + ffmpeg.
    4. Sube el MP4 a Cloudinary y devuelve la URL directa.
    """
    import io, tempfile, os
    import numpy as np
    from PIL import Image, ImageOps, ImageFilter
    import imageio

    public_id = f"ml_clips/{item_id}"

    # ── Descargar imagen ─────────────────────────────────────────────────────
    img_r = requests.get(img_url, timeout=30)
    if img_r.status_code != 200:
        raise Exception(f"Error al descargar imagen: HTTP {img_r.status_code}")

    W, H   = 1080, 1920
    FPS    = 15          # 15fps = buen balance calidad/velocidad
    DURACION = 12
    TOTAL_FRAMES = FPS * DURACION

    # ── Preparar frame base 1080×1920 ────────────────────────────────────────
    img = Image.open(io.BytesIO(img_r.content)).convert("RGB")

    if fondo == "blurred":
        # Fondo: imagen escalada y difuminada
        bg = img.copy().resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=30))
    else:
        color_map = {"white": (255, 255, 255), "black": (0, 0, 0)}
        bg = Image.new("RGB", (W, H), color_map.get(fondo, (255, 255, 255)))

    # Redimensionar imagen para que entre en 1080×1920 con padding
    img_fit = ImageOps.contain(img, (W, H), method=Image.LANCZOS)
    paste_x = (W - img_fit.width)  // 2
    paste_y = (H - img_fit.height) // 2
    bg.paste(img_fit, (paste_x, paste_y))
    frame_base = np.array(bg)

    # ── Generar frames con Ken Burns (zoom-in suave de 1.0 → 1.2) ────────────
    frames = []
    for i in range(TOTAL_FRAMES):
        t     = i / max(TOTAL_FRAMES - 1, 1)
        scale = 1.0 + 0.2 * t           # zoom de 1.0 a 1.2
        new_w = int(W / scale)
        new_h = int(H / scale)
        x1    = (W - new_w) // 2
        y1    = (H - new_h) // 2
        crop  = frame_base[y1:y1 + new_h, x1:x1 + new_w]
        frame = np.array(Image.fromarray(crop).resize((W, H), Image.LANCZOS))
        frames.append(frame)

    # ── Codificar MP4 ────────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()

    writer = imageio.get_writer(
        tmp.name,
        fps=FPS,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-crf", "28", "-preset", "ultrafast"]
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    # ── Subir MP4 a Cloudinary ───────────────────────────────────────────────
    result = cloudinary.uploader.upload(
        tmp.name,
        public_id=public_id,
        resource_type="video",
        overwrite=True
    )
    os.unlink(tmp.name)

    url = result.get("secure_url")
    if not url:
        raise Exception("Cloudinary no devolvió URL del video subido")
    return url

# ── ESTADO DESDE DISCO ──
step       = load_json(STEP_FILE, 1)
items      = load_json(ITEMS_FILE, [])
results    = load_json(RESULTS_FILE, {"ok": {}, "errores": {}})
cloud_cfg  = load_json(CLOUD_FILE, {})
upload_res = load_json(UPLOAD_FILE, {"ok": {}, "errores": {}})

# ── HEADER ──
st.markdown('<div class="main-title">🎬 Generador de Clips ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Convertí las imágenes de tus publicaciones en videos 9:16 de 12 segundos con movimiento y audio</div>', unsafe_allow_html=True)

col_r1, col_r2 = st.columns([6,1])
with col_r2:
    if st.button('↺ Reiniciar'):
        for f in [ITEMS_FILE, STEP_FILE, RESULTS_FILE, CLOUD_FILE, UPLOAD_FILE]:
            if os.path.exists(f): os.remove(f)
        for key in ["token", "generando", "clips_terminado", "subiendo", "subida_terminada"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── LOGIN ──
token = get_token()
if not token:
    st.divider()
    st.subheader("Conectar con MercadoLibre")
    st.warning("⚠️ No hay sesión activa o el token venció. Autenticate.")
    auth_url = f"https://auth.mercadolibre.com.ar/authorization?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope={ML_SCOPES.replace(' ', '%20')}"
    st.markdown(f"**Paso 1:** [Hacé click acá para autorizar]({auth_url})")
    st.markdown("**Paso 2:** Pegá el código `TG-` que aparece en la URL:")
    code = st.text_input("Código TG-", placeholder="TG-XXXXXXXXX")
    if st.button("Conectar", type="primary") and code:
        t = obtener_token_code(code.strip().strip('"'))
        if t:
            st.session_state.token = t
            st.success("¡Conectado!")
            st.rerun()
        else:
            st.error("Código inválido o vencido.")
    st.stop()
else:
    with st.sidebar:
        st.caption("🔒 Sesión activa")
        if st.button("Desconectar de ML"):
            st.session_state.pop("token", None)
            if os.path.exists(TOKEN_FILE): os.remove(TOKEN_FILE)
            st.rerun()

# ── INDICADOR DE PASOS ──
st.divider()
for n, title, desc in [
    (1, "Importar publicaciones",   "Subí el Excel con los IDs de las publicaciones a procesar"),
    (2, "Configurar Cloudinary",    "Ingresá tus credenciales y elegí las opciones del clip"),
    (3, "Generar clips",            "Se procesan las imágenes y se generan los links de video"),
    (4, "Resultados",               "Revisá los links generados y avanzá a la subida"),
    (5, "Subir videos a ML",        "Se asocia cada clip a su publicación en MercadoLibre"),
]:
    cls  = "step-done"   if step > n else ("step-active" if step == n else "")
    icon = "✅"          if step > n else ("🔵"          if step == n else "⭕")
    badge = "Listo"      if step > n else ("En curso"    if step == n else "Pendiente")
    st.markdown(
        f'<div class="step-box {cls}">{icon} <strong>Paso {n}: {title}</strong> — '
        f'<small>{badge}</small><br><small style="color:#888">{desc}</small></div>',
        unsafe_allow_html=True
    )
st.divider()

# ══════════════════════════════════════════════════════════════
# PASO 1 — Importar publicaciones
# ══════════════════════════════════════════════════════════════
if step == 1:
    st.subheader("Paso 1 — Importar publicaciones")
    archivo = st.file_uploader(
        "Subí tu planilla de MercadoLibre (.xlsx)",
        type=["xlsx"],
        help="Usá la misma planilla que exporta ML desde 'Mis publicaciones'"
    )
    if archivo:
        df = pd.read_excel(archivo, sheet_name="Publicaciones", header=None)
        nuevos_items = df.iloc[4:, 1].dropna().astype(str).tolist()
        nuevos_items = [i for i in nuevos_items if i.startswith("MLA")]
        st.success(f"✅ {len(nuevos_items)} publicaciones encontradas")
        if nuevos_items:
            st.dataframe(
                pd.DataFrame(nuevos_items, columns=["ID Publicación"]),
                use_container_width=True,
                height=200
            )
            if st.button("Continuar al Paso 2 →", type="primary"):
                save_json(ITEMS_FILE, nuevos_items)
                save_json(STEP_FILE, 2)
                st.rerun()

# ══════════════════════════════════════════════════════════════
# PASO 2 — Configurar Cloudinary
# ══════════════════════════════════════════════════════════════
elif step == 2:
    st.subheader("Paso 2 — Configurar Cloudinary")

    if not items:
        st.warning("No hay publicaciones cargadas. Volvé al Paso 1.")
        if st.button("← Volver al Paso 1"):
            save_json(STEP_FILE, 1); st.rerun()
        st.stop()

    st.info(f"📋 Se van a generar clips para **{len(items)}** publicaciones.")

    with st.expander("ℹ️ ¿Dónde encuentro mis credenciales de Cloudinary?", expanded=False):
        st.markdown("""
1. Entrá a [cloudinary.com/console](https://cloudinary.com/console)  
2. En el dashboard vas a ver **Cloud Name**, **API Key** y **API Secret**  
3. Copiá esos tres valores abajo ↓
        """)

    col1, col2 = st.columns(2)
    with col1:
        cloud_name = st.text_input(
            "Cloud Name",
            value=cloud_cfg.get("cloud_name", ""),
            placeholder="mi-cloud"
        )
        api_key = st.text_input(
            "API Key",
            value=cloud_cfg.get("api_key", ""),
            placeholder="123456789012345"
        )
    with col2:
        api_secret = st.text_input(
            "API Secret",
            value=cloud_cfg.get("api_secret", ""),
            placeholder="AbCdEfGhIjKlMnOpQrStUv",
            type="password"
        )
        audio_public_id = st.text_input(
            "Public ID del audio (opcional)",
            value=cloud_cfg.get("audio_public_id", ""),
            placeholder="musica_fondo_stock",
            help="ID del archivo de audio que ya subiste a Cloudinary. Dejalo vacío si no querés música."
        )

    st.divider()
    st.markdown("**Opciones del clip:**")
    col3, col4 = st.columns(2)
    with col3:
        fondo_opcion = st.selectbox(
            "Fondo del encuadre 9:16",
            options=["white", "black", "blurred"],
            index=0,
            format_func=lambda x: {"white":"⬜ Blanco","black":"⬛ Negro","blurred":"🌫️ Difuminado"}[x],
            help="Color de relleno cuando la imagen no ocupa todo el frame 9:16"
        )
    with col4:
        st.metric("Formato del video", "9:16 (1080×1920)")
        st.metric("Duración", "12 segundos")

    if st.button("Guardar y Continuar al Paso 3 →", type="primary"):
        if not cloud_name or not api_key or not api_secret:
            st.error("⚠️ Cloud Name, API Key y API Secret son obligatorios.")
        else:
            cfg = {
                "cloud_name": cloud_name,
                "api_key": api_key,
                "api_secret": api_secret,
                "audio_public_id": audio_public_id,
                "fondo": fondo_opcion
            }
            save_json(CLOUD_FILE, cfg)
            save_json(STEP_FILE, 3)
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PASO 3 — Generar clips
# ══════════════════════════════════════════════════════════════
elif step == 3:
    st.subheader("Paso 3 — Generar clips")

    if not items:
        st.warning("No hay publicaciones cargadas. Volvé al Paso 1.")
        if st.button("← Volver al Paso 1"):
            save_json(STEP_FILE, 1); st.rerun()
        st.stop()

    if not cloud_cfg:
        st.warning("No hay configuración de Cloudinary. Volvé al Paso 2.")
        if st.button("← Volver al Paso 2"):
            save_json(STEP_FILE, 2); st.rerun()
        st.stop()

    # Configurar Cloudinary con las credenciales guardadas
    configurar_cloudinary(
        cloud_cfg["cloud_name"],
        cloud_cfg["api_key"],
        cloud_cfg["api_secret"]
    )

    audio_public_id = cloud_cfg.get("audio_public_id", "")
    fondo           = cloud_cfg.get("fondo", "white")

    # Calcular pendientes (resume si hubo corte)
    ya_ok     = set(results.get("ok", {}).keys())
    ya_err    = dict(results.get("errores", {}))
    pendientes = [i for i in items if i not in ya_ok and i not in ya_err]

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total", len(items))
    col_info2.metric("✅ Procesados", len(ya_ok))
    col_info3.metric("⏳ Pendientes", len(pendientes))

    if pendientes and not st.session_state.get("generando"):
        st.info(f"Quedan **{len(pendientes)}** clips por generar.")
        if st.button("▶ Iniciar generación", type="primary"):
            st.session_state["generando"] = True
            st.rerun()

    if not pendientes and not st.session_state.get("clips_terminado"):
        st.session_state["clips_terminado"] = True

    # ── Procesamiento ──
    if st.session_state.get("generando") and pendientes:
        bar     = st.progress(0, text="Iniciando...")
        estado  = st.empty()
        ok_run  = 0
        err_run = 0
        total_pendientes = len(pendientes)

        for idx, item_id in enumerate(pendientes):
            try:
                # 1. Obtener URL de imagen desde ML
                img_url = obtener_url_imagen_ml(item_id, token)

                # 2. Subir a Cloudinary y obtener URL del video
                video_url = generar_clip_cloudinary(
                    item_id, img_url, audio_public_id, fondo
                )

                ya_ok.add(item_id)
                results.setdefault("ok", {})[item_id] = video_url
                ok_run += 1

            except Exception as e:
                ya_err[item_id] = str(e)
                results.setdefault("errores", {})[item_id] = str(e)
                err_run += 1

            # Guardar checkpoint tras cada ítem
            save_json(RESULTS_FILE, results)

            progreso = (idx + 1) / total_pendientes
            bar.progress(progreso, text=f"{idx+1}/{total_pendientes} — {item_id}")
            estado.markdown(f"✅ **{ok_run}** OK · ❌ **{err_run}** errores")

            time.sleep(0.15)  # Pausa mínima para no saturar las APIs

        bar.progress(1.0, text="✅ Completado")
        st.session_state.pop("generando", None)
        st.session_state["clips_terminado"] = True
        save_json(STEP_FILE, 4)
        st.rerun()

    # Botón para ir a resultados cuando ya terminó
    if st.session_state.get("clips_terminado") or not pendientes:
        st.success(f"✅ Clips generados: {len(ya_ok)} OK, {len(ya_err)} errores.")
        if st.button("Ver resultados →", type="primary"):
            save_json(STEP_FILE, 4)
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PASO 4 — Resultados
# ══════════════════════════════════════════════════════════════
elif step == 4:
    st.subheader("Paso 4 — Resultados")

    ok_dict  = results.get("ok", {})
    err_dict = results.get("errores", {})

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total procesadas", len(items))
    col_m2.metric("✅ Clips generados", len(ok_dict))
    col_m3.metric("❌ Errores", len(err_dict))

    st.divider()

    # ── CSV para descarga ──
    # ── Vista de clips exitosos ──
    if ok_dict:
        filas = [{"ID Publicacion": k, "Video URL": v} for k, v in ok_dict.items()]
        if err_dict:
            filas += [{"ID Publicacion": k, "Video URL": f"ERROR: {v}"} for k, v in err_dict.items()]
        df_export = pd.DataFrame(filas)
        csv_buf = io.StringIO()
        df_export.to_csv(csv_buf, index=False)
        st.download_button(
            label="💾 Descargar CSV con todos los links",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="clips_ml.csv",
            mime="text/csv",
            type="primary"
        )
        st.divider()
        st.markdown(f"### ✅ {len(ok_dict)} clips generados correctamente")
        st.caption("Cada URL es un video listo para usar. Cloudinary lo renderiza en el momento de la primera visita.")
        for item_id, video_url in ok_dict.items():
            st.markdown(
                f'<div class="video-card video-ok">'
                f'📦 <strong>{item_id}</strong><br>'
                f'<small><a href="{video_url}" target="_blank">{video_url[:80]}...</a></small>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Vista de errores (siempre visible si existen) ──
    if err_dict:
        st.divider()
        st.markdown(f"### ❌ {len(err_dict)} errores")
        for item_id, motivo in err_dict.items():
            st.markdown(
                f'<div class="video-card video-err">'
                f'📦 <strong>{item_id}</strong><br>'
                f'<small style="color:#ff6b6b">{motivo}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
        if st.button("↺ Reintentar errores"):
            new_results = {"ok": ok_dict, "errores": {}}
            save_json(RESULTS_FILE, new_results)
            save_json(ITEMS_FILE, list(err_dict.keys()))
            save_json(STEP_FILE, 3)
            st.session_state.pop("clips_terminado", None)
            st.rerun()

    if not ok_dict and not err_dict:
        st.warning("No hay resultados. Volvé al Paso 3 para generar los clips.")
        if st.button("← Volver al Paso 3", key="volver_p3_sin_resultados"):
            save_json(STEP_FILE, 3); st.rerun()

    st.divider()
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("← Volver al Paso 3", key="volver_p3_footer"):
            save_json(STEP_FILE, 3)
            st.session_state.pop("clips_terminado", None)
            st.rerun()
    with col_b2:
        if st.button("⚙️ Cambiar credenciales Cloudinary"):
            for f in [RESULTS_FILE, CLOUD_FILE]:
                if os.path.exists(f): os.remove(f)
            save_json(STEP_FILE, 2)
            st.session_state.pop("clips_terminado", None)
            st.rerun()
    with col_b3:
        if ok_dict and st.button("Subir videos a ML →", type="primary", key="ir_paso5"):
            save_json(STEP_FILE, 5)
            st.rerun()

    st.divider()
    if st.button("↺ Nueva tanda de publicaciones"):
        for f in [ITEMS_FILE, STEP_FILE, RESULTS_FILE, CLOUD_FILE, UPLOAD_FILE]:
            if os.path.exists(f): os.remove(f)
        st.session_state.pop("clips_terminado", None)
        st.rerun()

# ══════════════════════════════════════════════════════════════
# PASO 5 — Descargar videos para subir a MercadoLibre
# ══════════════════════════════════════════════════════════════
elif step == 5:
    st.subheader("Paso 5 — Descargar videos")

    ok_dict = results.get("ok", {})
    if not ok_dict:
        st.warning("No hay clips generados. Volvé al Paso 3.")
        if st.button("← Volver al Paso 3"):
            save_json(STEP_FILE, 3); st.rerun()
        st.stop()

    st.info(
        "📥 Descargá cada video y subilo manualmente a tu publicación en MercadoLibre.\n\n"
        "**Cómo subir:** Mis publicaciones → Editar → sección Video → Agregar video → Guardar"
    )
    st.divider()

    st.markdown(f"### 🎬 {len(ok_dict)} videos listos para descargar")

    for item_id, video_url in ok_dict.items():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(
                f'<div class="video-card video-ok">'
                f'📦 <strong>{item_id}</strong><br>'
                f'<small><a href="{video_url}" target="_blank">Ver video en Cloudinary ↗</a></small>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_b:
            # Descargar el MP4 y ofrecer botón de descarga directa
            try:
                r = requests.get(video_url, timeout=60)
                if r.status_code == 200:
                    st.download_button(
                        label="⬇ Descargar MP4",
                        data=r.content,
                        file_name=f"{item_id}.mp4",
                        mime="video/mp4",
                        key=f"dl_{item_id}"
                    )
                else:
                    st.caption("Error al obtener video")
            except Exception as e:
                st.caption(f"Error: {e}")

    st.divider()
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.button("← Volver al Paso 4"):
            save_json(STEP_FILE, 4)
            st.rerun()
    with col_f2:
        if st.button("↺ Nueva tanda de publicaciones", key="nueva_tanda_p5"):
            for f in [ITEMS_FILE, STEP_FILE, RESULTS_FILE, CLOUD_FILE, UPLOAD_FILE]:
                if os.path.exists(f): os.remove(f)
            for key in ["clips_terminado", "subida_terminada", "subiendo"]:
                st.session_state.pop(key, None)
            st.rerun()
