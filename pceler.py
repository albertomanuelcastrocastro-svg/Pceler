"""
PCELER — Acelerómetro PALMERO (v2.4)
========================================
v2.4: añade LABORATORIO DE ELONGACIÓN — testea umbrales de elongación
      del MACD 15m para entradas en bordes (no en el centro).
v2.3: añade MONITOR automático — graba cada señal en vivo a GitHub
      (pceler_signals_log.json) para auditoría de escala_amplia vs escala_xl.
v2.2: añade lógica SIMPLIFICADA en paralelo a la de percentiles.
Lógica simplificada:
  - Giro alcista: pendiente del MACD 15M pasa de negativa a positiva
    (sign change up)
  - Giro bajista: pendiente del MACD 15M pasa de positiva a negativa
    (sign change down)
  - Filtro 4H igual: solo LONGs cuando 4H sube, SHORTs cuando baja
  - Anti-spam: gap mínimo entre señales
Esta lógica es PORTABLE a Pine Script sin riesgo de repintado.
Si los resultados son similares a la lógica de percentiles, será la
elegida para el indicador de TradingView.
"""
import os
import time
import requests
import numpy as np
from datetime import datetime, timezone
import threading
import json
import base64
from flask import Flask, jsonify
app = Flask(__name__)
@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response
SYMBOLS = ["XRPUSDT", "SOLUSDT"]
BINANCE_BASE = "https://data-api.binance.vision/api/v3/klines"
TIMEFRAMES = {
    "5m": {"interval": "5m", "limit": 500},
    "15m": {"interval": "15m", "limit": 500},
    "1h": {"interval": "1h", "limit": 500},
    "4h": {"interval": "4h", "limit": 500},
}
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
UMBRALES_GIRO = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CONFIGS_LAB = {
    "actual": {"sl_pct": -0.005, "tp1_pct": 0.005, "tp1_peso": 0.40,
        "tp2_pct": 0.008, "tp2_peso": 0.30, "stop_tras_tp1_pct": 0.0, "stop_tras_tp2_pct": 0.0},
    "margen_amplio": {"sl_pct": -0.005, "tp1_pct": 0.005, "tp1_peso": 0.40,
        "tp2_pct": 0.008, "tp2_peso": 0.30, "stop_tras_tp1_pct": -0.003, "stop_tras_tp2_pct": -0.003},
    "escala_amplia": {"sl_pct": -0.02, "tp1_pct": 0.01, "tp1_peso": 0.40,
        "tp2_pct": 0.016, "tp2_peso": 0.30, "stop_tras_tp1_pct": -0.01, "stop_tras_tp2_pct": -0.01},
    "escala_xl": {"sl_pct": -0.03, "tp1_pct": 0.015, "tp1_peso": 0.40,
        "tp2_pct": 0.025, "tp2_peso": 0.30, "stop_tras_tp1_pct": -0.015, "stop_tras_tp2_pct": -0.015},
    "escala_xxl": {"sl_pct": -0.04, "tp1_pct": 0.02, "tp1_peso": 0.40,
        "tp2_pct": 0.035, "tp2_peso": 0.30, "stop_tras_tp1_pct": -0.02, "stop_tras_tp2_pct": -0.02},
    "sin_breakeven": {"sl_pct": -0.005, "tp1_pct": 0.005, "tp1_peso": 0.40,
        "tp2_pct": 0.008, "tp2_peso": 0.30, "stop_tras_tp1_pct": -0.005, "stop_tras_tp2_pct": -0.005},
}

# ─── MONITOR: config para grabación automática de señales ───
GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GH_REPO = os.environ.get("PCELER_LOG_REPO", "albertomanuelcastrocastro-svg/palmero-bot-pytho")
GH_LOG_FILE = "pceler_signals_log.json"
MONITOR_INTERVAL = 960  # 16 minutos (> 1 vela de 15m)
MONITOR_TF = "15m"
MONITOR_UMBRAL = 0.25  # legacy, se mantiene por compatibilidad
MONITOR_ELONG_PCT = 0.25  # % del precio
MONITOR_DIST_PCT = 0.15   # % del precio
_logged_timestamps = set()  # timestamps ya registrados (en memoria)
_monitor_initialized = False

_cache = {}
_cache_ttl = 30
def fetch_klines(symbol, interval, limit=500):
    key = (symbol, interval, limit)
    now = time.time()
    if key in _cache and now - _cache[key]["ts"] < _cache_ttl:
        return _cache[key]["data"]
    url = f"{BINANCE_BASE}?symbol={symbol}&interval={interval}&limit={limit}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    _cache[key] = {"ts": now, "data": raw}
    return raw

def fetch_klines_extended(symbol, interval, total=3000):
    """Fetch más de 1000 velas haciendo múltiples llamadas a Binance."""
    all_klines = []
    end_time = None
    remaining = total
    while remaining > 0:
        batch = min(remaining, 1000)
        url = f"{BINANCE_BASE}?symbol={symbol}&interval={interval}&limit={batch}"
        if end_time:
            url += f"&endTime={end_time}"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break
        all_klines = raw + all_klines  # prepend (oldest first)
        end_time = int(raw[0][0]) - 1  # antes de la primera vela de este batch
        remaining -= len(raw)
        if len(raw) < batch:
            break
        time.sleep(0.2)  # no saturar Binance
    # Eliminar duplicados por timestamp
    seen = set()
    unique = []
    for k in all_klines:
        ts = int(k[0])
        if ts not in seen:
            seen.add(ts)
            unique.append(k)
    unique.sort(key=lambda x: int(x[0]))
    return unique
def ema(values, period):
    alpha = 2 / (period + 1)
    result = np.zeros_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result
def calcular_macd(closes):
    ema_fast = ema(closes, MACD_FAST)
    ema_slow = ema(closes, MACD_SLOW)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, MACD_SIGNAL)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
def calcular_pendientes(macd_line):
    return np.diff(macd_line)
def calcular_aceleracion(pendientes):
    return np.diff(pendientes)
def estadisticas_pendientes(pendientes):
    if len(pendientes) < 10:
        return None
    return {
        "n_velas": int(len(pendientes)),
        "media": round(float(np.mean(pendientes)), 8),
        "std": round(float(np.std(pendientes)), 8),
        "min": round(float(np.min(pendientes)), 8),
        "max": round(float(np.max(pendientes)), 8),
        "percentil_5": round(float(np.percentile(pendientes, 5)), 8),
        "percentil_10": round(float(np.percentile(pendientes, 10)), 8),
        "percentil_25": round(float(np.percentile(pendientes, 25)), 8),
        "percentil_50": round(float(np.percentile(pendientes, 50)), 8),
        "percentil_75": round(float(np.percentile(pendientes, 75)), 8),
        "percentil_90": round(float(np.percentile(pendientes, 90)), 8),
        "percentil_95": round(float(np.percentile(pendientes, 95)), 8),
    }
def detectar_apogeo_y_giro(pendientes, percentiles):
    if percentiles is None or len(pendientes) < 3:
        return None
    p90 = percentiles["percentil_90"]
    p10 = percentiles["percentil_10"]
    actual = float(pendientes[-1])
    anterior = float(pendientes[-2])
    en_apogeo_alcista = actual >= p90
    en_apogeo_bajista = actual <= p10
    retroceso_desde_max = None
    retroceso_desde_min = None
    ventana = min(20, len(pendientes))
    max_reciente = float(np.max(pendientes[-ventana:]))
    min_reciente = float(np.min(pendientes[-ventana:]))
    if max_reciente > 0 and actual < max_reciente:
        retroceso_desde_max = round((max_reciente - actual) / abs(max_reciente), 4)
    if min_reciente < 0 and actual > min_reciente:
        retroceso_desde_min = round((actual - min_reciente) / abs(min_reciente), 4)
    percentil_actual = float(np.searchsorted(np.sort(pendientes), actual) / len(pendientes) * 100)
    return {
        "pendiente_actual": round(actual, 8),
        "pendiente_anterior": round(anterior, 8),
        "pendiente_subiendo": bool(actual > anterior),
        "pendiente_bajando": bool(actual < anterior),
        "en_apogeo_alcista": bool(en_apogeo_alcista),
        "en_apogeo_bajista": bool(en_apogeo_bajista),
        "max_reciente_20v": round(max_reciente, 8),
        "min_reciente_20v": round(min_reciente, 8),
        "retroceso_desde_max_pct": retroceso_desde_max,
        "retroceso_desde_min_pct": retroceso_desde_min,
        "percentil_actual": round(percentil_actual, 1),
    }
def obtener_direccion_4h(symbol, timestamps_senales):
    raw_4h = fetch_klines(symbol, "4h", 500)
    closes_4h = np.array([float(k[4]) for k in raw_4h])
    open_times_4h = [int(k[0]) for k in raw_4h]
    close_times_4h = [int(k[6]) for k in raw_4h]
    macd_4h, _, _ = calcular_macd(closes_4h)
    pendientes_4h = calcular_pendientes(macd_4h)
    direccion_por_ts = {}
    for ts_str in timestamps_senales:
        if ts_str is None:
            continue
        ts_dt = datetime.fromisoformat(ts_str)
        ts_ms = int(ts_dt.timestamp() * 1000)
        dir_4h = "lateral"
        for j in range(len(open_times_4h)):
            if open_times_4h[j] <= ts_ms <= close_times_4h[j]:
                if j >= MACD_SLOW + MACD_SIGNAL + 1:
                    idx_pendiente = j - 1
                    if idx_pendiente < len(pendientes_4h):
                        p = float(pendientes_4h[idx_pendiente])
                        if p > 0:
                            dir_4h = "alcista"
                        elif p < 0:
                            dir_4h = "bajista"
                break
        direccion_por_ts[ts_str] = dir_4h
    return direccion_por_ts
def simular_senales_percentiles(pendientes, closes, timestamps, percentiles, umbral):
    if percentiles is None or len(pendientes) < 30:
        return []
    senales = []
    p90 = percentiles["percentil_90"]
    p10 = percentiles["percentil_10"]
    en_apogeo_alcista = False
    max_pendiente_alcista = 0.0
    en_apogeo_bajista = False
    min_pendiente_bajista = 0.0
    ultima_senal_idx = -30
    for i in range(1, len(pendientes)):
        p = float(pendientes[i])
        if p >= p90:
            en_apogeo_alcista = True
            max_pendiente_alcista = max(max_pendiente_alcista, p)
        if p <= p10:
            en_apogeo_bajista = True
            min_pendiente_bajista = min(min_pendiente_bajista, p)
        if en_apogeo_alcista and max_pendiente_alcista > 0:
            retroceso = (max_pendiente_alcista - p) / abs(max_pendiente_alcista)
            if retroceso >= umbral and (i - ultima_senal_idx) >= 6:
                precio_idx = i + 1
                if precio_idx < len(closes):
                    senales.append({
                        "tipo": "SHORT", "motivo": "giro_bajista_desde_apogeo",
                        "vela_idx": int(precio_idx),
                        "precio": round(float(closes[precio_idx]), 6),
                        "timestamp": timestamps[precio_idx] if precio_idx < len(timestamps) else None,
                        "umbral_usado": float(umbral),
                    })
                    ultima_senal_idx = i
                en_apogeo_alcista = False
                max_pendiente_alcista = 0.0
        if en_apogeo_bajista and min_pendiente_bajista < 0:
            retroceso = (p - min_pendiente_bajista) / abs(min_pendiente_bajista)
            if retroceso >= umbral and (i - ultima_senal_idx) >= 6:
                precio_idx = i + 1
                if precio_idx < len(closes):
                    senales.append({
                        "tipo": "LONG", "motivo": "giro_alcista_desde_apogeo",
                        "vela_idx": int(precio_idx),
                        "precio": round(float(closes[precio_idx]), 6),
                        "timestamp": timestamps[precio_idx] if precio_idx < len(timestamps) else None,
                        "umbral_usado": float(umbral),
                    })
                    ultima_senal_idx = i
                en_apogeo_bajista = False
                min_pendiente_bajista = 0.0
    return senales
def simular_senales_simple(pendientes, closes, timestamps, gap_min=6, min_racha=3):
    if len(pendientes) < min_racha + 2:
        return []
    senales = []
    ultima_senal_idx = -30
    for i in range(min_racha + 1, len(pendientes)):
        p_actual = float(pendientes[i])
        p_prev = float(pendientes[i - 1])
        racha_neg = all(float(pendientes[i - k - 1]) < 0 for k in range(min_racha))
        racha_pos = all(float(pendientes[i - k - 1]) > 0 for k in range(min_racha))
        if (i - ultima_senal_idx) < gap_min:
            continue
        if racha_neg and p_prev < 0 and p_actual > 0:
            precio_idx = i + 1
            if precio_idx < len(closes):
                senales.append({
                    "tipo": "LONG", "motivo": "cambio_signo_a_positivo",
                    "vela_idx": int(precio_idx),
                    "precio": round(float(closes[precio_idx]), 6),
                    "timestamp": timestamps[precio_idx] if precio_idx < len(timestamps) else None,
                    "min_racha": int(min_racha),
                })
                ultima_senal_idx = i
        elif racha_pos and p_prev > 0 and p_actual < 0:
            precio_idx = i + 1
            if precio_idx < len(closes):
                senales.append({
                    "tipo": "SHORT", "motivo": "cambio_signo_a_negativo",
                    "vela_idx": int(precio_idx),
                    "precio": round(float(closes[precio_idx]), 6),
                    "timestamp": timestamps[precio_idx] if precio_idx < len(timestamps) else None,
                    "min_racha": int(min_racha),
                })
                ultima_senal_idx = i
    return senales
def filtrar_por_4h(senales, direccion_4h):
    filtradas = []
    for s in senales:
        ts = s.get("timestamp")
        if ts is None:
            continue
        dir_4h = direccion_4h.get(ts, "lateral")
        s["direccion_4h"] = dir_4h
        if s["tipo"] == "LONG" and dir_4h == "alcista":
            filtradas.append(s)
        elif s["tipo"] == "SHORT" and dir_4h == "bajista":
            filtradas.append(s)
    return filtradas
def evaluar_senales(senales, closes, n_velas_futuras=20):
    resultados = []
    for s in senales:
        idx = s["vela_idx"]
        if idx + n_velas_futuras >= len(closes):
            continue
        entrada = s["precio"]
        if entrada == 0:
            continue
        dir_mult = -1.0 if s["tipo"] == "SHORT" else 1.0
        precios_futuros = closes[idx + 1: idx + 1 + n_velas_futuras]
        if len(precios_futuros) == 0:
            continue
        mejor_precio = float(np.min(precios_futuros)) if s["tipo"] == "SHORT" else float(np.max(precios_futuros))
        mejor_pct = dir_mult * (mejor_precio - entrada) / entrada * 100
        cierre_20v = float(closes[idx + n_velas_futuras])
        resultado_20v = dir_mult * (cierre_20v - entrada) / entrada * 100
        resultado = {
            "tipo": s["tipo"], "precio": s["precio"], "timestamp": s["timestamp"],
            "mejor_pct_20v": round(float(mejor_pct), 3),
            "resultado_20v_pct": round(float(resultado_20v), 3),
            "ganadora_20v": bool(resultado_20v > 0),
        }
        if "direccion_4h" in s:
            resultado["direccion_4h"] = s["direccion_4h"]
        if "min_racha" in s:
            resultado["min_racha"] = s["min_racha"]
        if "umbral_usado" in s:
            resultado["umbral_usado"] = s["umbral_usado"]
        resultados.append(resultado)
    return resultados
def resumir_evaluacion(evaluadas):
    if not evaluadas:
        return {"n_senales": 0, "winrate_pct": None, "resultado_medio_pct": None, "mejor_medio_pct": None}
    ganadoras = sum(1 for s in evaluadas if s["ganadora_20v"])
    return {
        "n_senales": int(len(evaluadas)),
        "winrate_pct": round(float(ganadoras / len(evaluadas) * 100), 1),
        "resultado_medio_pct": round(float(sum(s["resultado_20v_pct"] for s in evaluadas) / len(evaluadas)), 3),
        "mejor_medio_pct": round(float(sum(s["mejor_pct_20v"] for s in evaluadas) / len(evaluadas)), 3),
    }
def simular_trade_sltp(senal, closes_validas, cfg):
    idx = senal["vela_idx"]
    if idx >= len(closes_validas):
        return None
    entrada = senal["precio"]
    if entrada == 0:
        return None
    es_long = senal["tipo"] == "LONG"
    dir_mult = 1.0 if es_long else -1.0
    sl = cfg["sl_pct"]; tp1 = cfg["tp1_pct"]; tp1_peso = cfg["tp1_peso"]
    tp2 = cfg["tp2_pct"]; tp2_peso = cfg["tp2_peso"]
    be1 = cfg["stop_tras_tp1_pct"]; be2 = cfg["stop_tras_tp2_pct"]
    stop_actual = sl; fase = 1; realizado = 0.0; estado = None
    for k_idx in range(idx + 1, len(closes_validas)):
        precio_vela = float(closes_validas[k_idx])
        avance = dir_mult * (precio_vela - entrada) / entrada
        if fase == 1:
            if avance <= stop_actual:
                realizado = stop_actual; estado = "cerrada_sl"; break
            if avance >= tp1:
                realizado += tp1_peso * tp1; fase = 2; stop_actual = be1
        elif fase == 2:
            if avance <= stop_actual:
                peso_resto = 1 - tp1_peso
                realizado += peso_resto * stop_actual; estado = "cerrada_be1"; break
            if avance >= tp2:
                realizado += tp2_peso * tp2; fase = 3; stop_actual = be2
        elif fase == 3:
            if avance <= stop_actual:
                peso_resto = 1 - tp1_peso - tp2_peso
                realizado += peso_resto * stop_actual; estado = "cerrada_be2"; break
        if k_idx - idx >= 100:
            break
    if estado is None:
        ultimo = float(closes_validas[min(idx + 100, len(closes_validas) - 1)])
        avance_final = dir_mult * (ultimo - entrada) / entrada
        if fase == 1:
            resultado = avance_final
        elif fase == 2:
            resultado = realizado + (1 - tp1_peso) * avance_final
        else:
            resultado = realizado + (1 - tp1_peso - tp2_peso) * avance_final
        estado = f"abierta_f{fase}"
    else:
        resultado = realizado
    return {"estado": estado, "resultado_pct": round(resultado * 100, 3)}
def analizar_tf(symbol, tf_label, interval, limit):
    raw = fetch_klines(symbol, interval, limit)
    if len(raw) < MACD_SLOW + MACD_SIGNAL + 10:
        return {"error": "datos_insuficientes"}
    closes = np.array([float(k[4]) for k in raw])
    macd_line, signal_line, histogram = calcular_macd(closes)
    pendientes = calcular_pendientes(macd_line)
    skip = MACD_SLOW + MACD_SIGNAL
    pendientes_validas = pendientes[skip:]
    percentiles = estadisticas_pendientes(pendientes_validas)
    estado = detectar_apogeo_y_giro(pendientes_validas, percentiles)
    return {
        "tf": tf_label,
        "macd_actual": round(float(macd_line[-1]), 8),
        "macd_anterior": round(float(macd_line[-2]), 8),
        "signal_actual": round(float(signal_line[-1]), 8),
        "calibracion": percentiles, "estado": estado,
        "n_velas_analizadas": int(len(pendientes_validas)),
    }
def analizar_senales_tf(symbol, tf_label, interval, limit):
    raw = fetch_klines(symbol, interval, limit)
    if len(raw) < MACD_SLOW + MACD_SIGNAL + 30:
        return {"error": "datos_insuficientes"}
    closes = np.array([float(k[4]) for k in raw])
    timestamps = [datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).isoformat() for k in raw]
    macd_line, _, _ = calcular_macd(closes)
    pendientes = calcular_pendientes(macd_line)
    skip = MACD_SLOW + MACD_SIGNAL
    pendientes_validas = pendientes[skip:]
    closes_validas = closes[skip + 1:]
    timestamps_validos = timestamps[skip + 1:]
    percentiles = estadisticas_pendientes(pendientes_validas)
    es_tf_menor = tf_label in ("5m", "15m", "1h")
    direccion_4h = {}
    if es_tf_menor:
        try:
            all_timestamps = []
            for umbral in UMBRALES_GIRO:
                senales_temp = simular_senales_percentiles(pendientes_validas, closes_validas, timestamps_validos, percentiles, umbral)
                all_timestamps.extend([s["timestamp"] for s in senales_temp if s["timestamp"]])
            all_timestamps = list(set(all_timestamps))
            if all_timestamps:
                direccion_4h = obtener_direccion_4h(symbol, all_timestamps)
        except Exception as e:
            print(f"Error 4H: {e}")
    resultados_por_umbral = {}
    for umbral in UMBRALES_GIRO:
        senales = simular_senales_percentiles(pendientes_validas, closes_validas, timestamps_validos, percentiles, umbral)
        evaluadas_sin = evaluar_senales(senales, closes_validas)
        resumen_sin = resumir_evaluacion(evaluadas_sin)
        resumen_sin["senales"] = evaluadas_sin[-10:]
        resultado_umbral = {"umbral": float(umbral), "sin_filtro": resumen_sin}
        if es_tf_menor and direccion_4h:
            senales_filt = filtrar_por_4h(senales, direccion_4h)
            evaluadas_filt = evaluar_senales(senales_filt, closes_validas)
            resumen_filt = resumir_evaluacion(evaluadas_filt)
            resumen_filt["senales"] = evaluadas_filt[-10:]
            resultado_umbral["filtro_4h"] = resumen_filt
        resultados_por_umbral[str(umbral)] = resultado_umbral
    return {"tf": tf_label, "n_velas": int(len(closes)),
        "filtro_4h_aplicado": bool(es_tf_menor and direccion_4h),
        "umbrales": resultados_por_umbral}
def analizar_senales_simple_tf(symbol, tf_label, interval, limit, min_racha=3):
    raw = fetch_klines(symbol, interval, limit)
    if len(raw) < MACD_SLOW + MACD_SIGNAL + 30:
        return {"error": "datos_insuficientes"}
    closes = np.array([float(k[4]) for k in raw])
    timestamps = [datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).isoformat() for k in raw]
    macd_line, _, _ = calcular_macd(closes)
    pendientes = calcular_pendientes(macd_line)
    skip = MACD_SLOW + MACD_SIGNAL
    pendientes_validas = pendientes[skip:]
    closes_validas = closes[skip + 1:]
    timestamps_validos = timestamps[skip + 1:]
    senales = simular_senales_simple(pendientes_validas, closes_validas, timestamps_validos, gap_min=6, min_racha=min_racha)
    es_tf_menor = tf_label in ("5m", "15m", "1h")
    direccion_4h = {}
    if es_tf_menor:
        try:
            all_ts = [s["timestamp"] for s in senales if s["timestamp"]]
            if all_ts:
                direccion_4h = obtener_direccion_4h(symbol, list(set(all_ts)))
        except Exception as e:
            print(f"Error 4H: {e}")
    evaluadas_sin = evaluar_senales(senales, closes_validas)
    resumen_sin = resumir_evaluacion(evaluadas_sin)
    resumen_sin["senales"] = evaluadas_sin[-10:]
    resultado = {"tf": tf_label, "min_racha": min_racha, "n_velas": int(len(closes)), "sin_filtro": resumen_sin}
    if es_tf_menor and direccion_4h:
        senales_filtradas = filtrar_por_4h(senales, direccion_4h)
        evaluadas_filt = evaluar_senales(senales_filtradas, closes_validas)
        resumen_filt = resumir_evaluacion(evaluadas_filt)
        resumen_filt["senales"] = evaluadas_filt[-10:]
        resultado["filtro_4h"] = resumen_filt
        resultado["filtro_4h_aplicado"] = True
    return resultado
def calcular_laboratorio(symbol, tf_label, interval, limit, umbral_fijo=0.25):
    raw = fetch_klines(symbol, interval, limit)
    if len(raw) < MACD_SLOW + MACD_SIGNAL + 30:
        return {"error": "datos_insuficientes"}
    closes = np.array([float(k[4]) for k in raw])
    timestamps = [datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).isoformat() for k in raw]
    macd_line, _, _ = calcular_macd(closes)
    pendientes = calcular_pendientes(macd_line)
    skip = MACD_SLOW + MACD_SIGNAL
    pendientes_validas = pendientes[skip:]
    closes_validas = closes[skip + 1:]
    timestamps_validos = timestamps[skip + 1:]
    percentiles = estadisticas_pendientes(pendientes_validas)
    senales = simular_senales_percentiles(pendientes_validas, closes_validas, timestamps_validos, percentiles, umbral_fijo)
    es_tf_menor = tf_label in ("5m", "15m", "1h")
    if es_tf_menor:
        try:
            all_ts = [s["timestamp"] for s in senales if s["timestamp"]]
            if all_ts:
                dir_4h = obtener_direccion_4h(symbol, list(set(all_ts)))
                senales = filtrar_por_4h(senales, dir_4h)
        except Exception as e:
            print(f"Error 4H: {e}")
    resultados_por_config = []
    for nombre, cfg in CONFIGS_LAB.items():
        valores = []; ganadoras = 0; perdedoras_sl = 0; ganancias = []; perdidas = []
        for s in senales:
            r = simular_trade_sltp(s, closes_validas, cfg)
            if r is None:
                continue
            pct = r["resultado_pct"]
            valores.append(pct)
            if pct > 0:
                ganadoras += 1; ganancias.append(pct)
            else:
                perdidas.append(pct)
            if r["estado"] == "cerrada_sl":
                perdedoras_sl += 1
        if valores:
            ganancia_media = round(float(np.mean(ganancias)), 3) if ganancias else 0
            perdida_media = round(float(np.mean(perdidas)), 3) if perdidas else 0
            ratio = round(abs(ganancia_media / perdida_media), 2) if perdida_media != 0 else None
            resultados_por_config.append({
                "config": nombre, "sl_pct": cfg["sl_pct"] * 100,
                "tp1_pct": cfg["tp1_pct"] * 100, "tp2_pct": cfg["tp2_pct"] * 100,
                "n_senales": int(len(valores)),
                "winrate_pct": round(float(ganadoras / len(valores) * 100), 1),
                "resultado_medio_pct": round(float(sum(valores) / len(valores)), 3),
                "ganancia_media_pct": ganancia_media, "perdida_media_pct": perdida_media,
                "ratio_beneficio_perdida": ratio, "n_stops": int(perdedoras_sl),
            })
        else:
            resultados_por_config.append({"config": nombre, "n_senales": 0})
    resultados_por_config.sort(key=lambda x: x.get("resultado_medio_pct") or -999, reverse=True)
    return {
        "simbolo": symbol, "tf": tf_label, "logica": "percentiles",
        "umbral_usado": float(umbral_fijo),
        "n_senales_tras_filtro_4h": int(len(senales)),
        "configs": resultados_por_config,
    }
def calcular_laboratorio_simple(symbol, tf_label, interval, limit, min_racha=3):
    raw = fetch_klines(symbol, interval, limit)
    if len(raw) < MACD_SLOW + MACD_SIGNAL + 30:
        return {"error": "datos_insuficientes"}
    closes = np.array([float(k[4]) for k in raw])
    timestamps = [datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).isoformat() for k in raw]
    macd_line, _, _ = calcular_macd(closes)
    pendientes = calcular_pendientes(macd_line)
    skip = MACD_SLOW + MACD_SIGNAL
    pendientes_validas = pendientes[skip:]
    closes_validas = closes[skip + 1:]
    timestamps_validos = timestamps[skip + 1:]
    senales = simular_senales_simple(pendientes_validas, closes_validas, timestamps_validos, gap_min=6, min_racha=min_racha)
    es_tf_menor = tf_label in ("5m", "15m", "1h")
    if es_tf_menor:
        try:
            all_ts = [s["timestamp"] for s in senales if s["timestamp"]]
            if all_ts:
                dir_4h = obtener_direccion_4h(symbol, list(set(all_ts)))
                senales = filtrar_por_4h(senales, dir_4h)
        except Exception as e:
            print(f"Error 4H: {e}")
    resultados_por_config = []
    for nombre, cfg in CONFIGS_LAB.items():
        valores = []; ganadoras = 0; perdedoras_sl = 0; ganancias = []; perdidas = []
        for s in senales:
            r = simular_trade_sltp(s, closes_validas, cfg)
            if r is None:
                continue
            pct = r["resultado_pct"]
            valores.append(pct)
            if pct > 0:
                ganadoras += 1; ganancias.append(pct)
            else:
                perdidas.append(pct)
            if r["estado"] == "cerrada_sl":
                perdedoras_sl += 1
        if valores:
            ganancia_media = round(float(np.mean(ganancias)), 3) if ganancias else 0
            perdida_media = round(float(np.mean(perdidas)), 3) if perdidas else 0
            ratio = round(abs(ganancia_media / perdida_media), 2) if perdida_media != 0 else None
            resultados_por_config.append({
                "config": nombre, "sl_pct": cfg["sl_pct"] * 100,
                "tp1_pct": cfg["tp1_pct"] * 100, "tp2_pct": cfg["tp2_pct"] * 100,
                "n_senales": int(len(valores)),
                "winrate_pct": round(float(ganadoras / len(valores) * 100), 1),
                "resultado_medio_pct": round(float(sum(valores) / len(valores)), 3),
                "ganancia_media_pct": ganancia_media, "perdida_media_pct": perdida_media,
                "ratio_beneficio_perdida": ratio, "n_stops": int(perdedoras_sl),
            })
        else:
            resultados_por_config.append({"config": nombre, "n_senales": 0})
    resultados_por_config.sort(key=lambda x: x.get("resultado_medio_pct") or -999, reverse=True)
    return {
        "simbolo": symbol, "tf": tf_label, "logica": "simple_cambio_signo",
        "min_racha": min_racha, "n_senales_tras_filtro_4h": int(len(senales)),
        "configs": resultados_por_config,
    }
@app.route("/")
def home():
    return jsonify({
        "servicio": "PCELER — Acelerómetro PALMERO",
        "version": "2.4",
        "novedad": "Laboratorio de elongación: entradas en bordes, no en el centro",
        "endpoints_percentiles": [
            "/estado/<symbol>", "/estado/<symbol>/<tf>",
            "/calibracion/<symbol>",
            "/senales/<symbol>/<tf>",
            "/laboratorio/<symbol>/<tf>/<umbral>",
        ],
        "endpoints_simple": [
            "/senales_simple/<symbol>/<tf>",
            "/laboratorio_simple/<symbol>/<tf>",
        ],
        "comparativa": "/comparativa/<symbol>/<tf>",
        "monitor": ["/monitor/status", "/monitor/log"],
        "laboratorio_elongacion": "/laboratorio_elongacion/<symbol>",
    })
@app.route("/estado/<symbol>")
def estado_symbol(symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    resultado = {"simbolo": symbol, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "timeframes": {}}
    for label, cfg in TIMEFRAMES.items():
        try:
            resultado["timeframes"][label] = analizar_tf(symbol, label, cfg["interval"], cfg["limit"])
        except Exception as e:
            resultado["timeframes"][label] = {"error": str(e)}
    return jsonify(resultado)
@app.route("/estado/<symbol>/<tf>")
def estado_symbol_tf(symbol, tf):
    symbol = symbol.upper(); tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        data = analizar_tf(symbol, tf, cfg["interval"], cfg["limit"])
        return jsonify({"simbolo": symbol, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/calibracion/<symbol>")
def calibracion_symbol(symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    resultado = {"simbolo": symbol, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "timeframes": {}}
    for label, cfg in TIMEFRAMES.items():
        try:
            raw = fetch_klines(symbol, cfg["interval"], cfg["limit"])
            closes = np.array([float(k[4]) for k in raw])
            macd_line, _, _ = calcular_macd(closes)
            pendientes = calcular_pendientes(macd_line)
            skip = MACD_SLOW + MACD_SIGNAL
            pendientes_validas = pendientes[skip:]
            aceleracion = calcular_aceleracion(pendientes_validas)
            resultado["timeframes"][label] = {
                "tf": label,
                "pendientes": estadisticas_pendientes(pendientes_validas),
                "aceleracion": estadisticas_pendientes(aceleracion),
                "n_velas": int(len(pendientes_validas)),
            }
        except Exception as e:
            resultado["timeframes"][label] = {"error": str(e)}
    return jsonify(resultado)
@app.route("/senales/<symbol>/<tf>")
def senales_symbol_tf(symbol, tf):
    symbol = symbol.upper(); tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        data = analizar_senales_tf(symbol, tf, cfg["interval"], cfg["limit"])
        return jsonify({"simbolo": symbol, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/senales_simple/<symbol>/<tf>")
@app.route("/senales_simple/<symbol>/<tf>/<min_racha>")
def senales_simple_endpoint(symbol, tf, min_racha=3):
    symbol = symbol.upper(); tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    try:
        min_racha = int(min_racha)
    except ValueError:
        return jsonify({"error": "min_racha debe ser entero"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        data = analizar_senales_simple_tf(symbol, tf, cfg["interval"], cfg["limit"], min_racha=min_racha)
        return jsonify({"simbolo": symbol, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/laboratorio/<symbol>/<tf>")
def laboratorio_tf_default(symbol, tf):
    symbol = symbol.upper(); tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        data = calcular_laboratorio(symbol, tf, cfg["interval"], cfg["limit"])
        return jsonify({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/laboratorio/<symbol>/<tf>/<umbral>")
def laboratorio_tf_umbral(symbol, tf, umbral):
    symbol = symbol.upper(); tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    try:
        umbral_f = float(umbral)
    except ValueError:
        return jsonify({"error": "umbral debe ser un número"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        data = calcular_laboratorio(symbol, tf, cfg["interval"], cfg["limit"], umbral_fijo=umbral_f)
        return jsonify({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/laboratorio_simple/<symbol>/<tf>")
@app.route("/laboratorio_simple/<symbol>/<tf>/<min_racha>")
def laboratorio_simple_endpoint(symbol, tf, min_racha=3):
    symbol = symbol.upper(); tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    try:
        min_racha = int(min_racha)
    except ValueError:
        return jsonify({"error": "min_racha debe ser entero"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        data = calcular_laboratorio_simple(symbol, tf, cfg["interval"], cfg["limit"], min_racha=min_racha)
        return jsonify({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/comparativa/<symbol>/<tf>")
def comparativa(symbol, tf):
    symbol = symbol.upper(); tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        lab_perc = calcular_laboratorio(symbol, tf, cfg["interval"], cfg["limit"], umbral_fijo=0.25)
        lab_simple = calcular_laboratorio_simple(symbol, tf, cfg["interval"], cfg["limit"], min_racha=3)
        def get_xl(lab):
            for c in lab.get("configs", []):
                if c.get("config") == "escala_xl":
                    return c
            return None
        return jsonify({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "simbolo": symbol, "tf": tf,
            "logica_percentiles": {
                "umbral_usado": 0.25,
                "n_senales": lab_perc.get("n_senales_tras_filtro_4h"),
                "escala_xl": get_xl(lab_perc),
                "todas_configs": lab_perc.get("configs", []),
            },
            "logica_simple": {
                "min_racha": 3,
                "n_senales": lab_simple.get("n_senales_tras_filtro_4h"),
                "escala_xl": get_xl(lab_simple),
                "todas_configs": lab_simple.get("configs", []),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/t/<bust>")
def todo_nocache(bust):
    resultado = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    for symbol in SYMBOLS:
        resultado[symbol] = {}
        for label, cfg in TIMEFRAMES.items():
            try:
                resultado[symbol][label] = analizar_tf(symbol, label, cfg["interval"], cfg["limit"])
            except Exception as e:
                resultado[symbol][label] = {"error": str(e)}
    return jsonify(resultado)

# ─── MONITOR: funciones de grabación automática ───

def gh_read_log():
    """Lee pceler_signals_log.json desde GitHub. Retorna (data_list, sha)."""
    if not GH_TOKEN:
        return [], None
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/contents/{GH_LOG_FILE}",
            headers={"Authorization": f"Bearer {GH_TOKEN}",
                     "User-Agent": "pceler-monitor",
                     "Accept": "application/vnd.github+json"},
            timeout=15
        )
        if resp.status_code == 404:
            return [], None
        resp.raise_for_status()
        j = resp.json()
        decoded = base64.b64decode(j["content"]).decode("utf-8")
        data = json.loads(decoded)
        return data, j["sha"]
    except Exception as e:
        print(f"[MONITOR] Error leyendo GitHub: {e}")
        return [], None


def gh_write_log(data, sha):
    """Escribe pceler_signals_log.json a GitHub. Re-lee SHA justo antes."""
    if not GH_TOKEN:
        return
    try:
        # Re-leer SHA inmediatamente antes de escribir (evitar 409)
        resp_sha = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/contents/{GH_LOG_FILE}",
            headers={"Authorization": f"Bearer {GH_TOKEN}",
                     "User-Agent": "pceler-monitor",
                     "Accept": "application/vnd.github+json"},
            timeout=15
        )
        if resp_sha.status_code == 200:
            sha = resp_sha.json()["sha"]

        content_b64 = base64.b64encode(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")).decode("utf-8")
        body = {
            "message": f"PCELER signal log {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            "content": content_b64,
        }
        if sha:
            body["sha"] = sha

        resp = requests.put(
            f"https://api.github.com/repos/{GH_REPO}/contents/{GH_LOG_FILE}",
            headers={"Authorization": f"Bearer {GH_TOKEN}",
                     "User-Agent": "pceler-monitor",
                     "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json"},
            json=body,
            timeout=15
        )
        if resp.ok:
            print(f"[MONITOR] Log guardado en GitHub ({len(data)} señales)")
        else:
            print(f"[MONITOR] Error escribiendo GitHub: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"[MONITOR] Error escribiendo GitHub: {e}")


def detectar_senales_monitor(symbol):
    """Detecta señales con lógica relativa (% del precio) + filtro 4H histograma.
    Coincide con Pine PCELER v2 (filtro activado)."""
    try:
        raw = fetch_klines(symbol, "15m", 500)
        if len(raw) < MACD_SLOW + MACD_SIGNAL + 30:
            return []
        closes = np.array([float(k[4]) for k in raw])
        timestamps_ms = [int(k[0]) for k in raw]
        macd_line, signal_line, _ = calcular_macd(closes)
        pendientes = calcular_pendientes(macd_line)
        histograma = macd_line - signal_line
        dir_4h_map = get_4h_hist_direction(symbol, timestamps_ms)
        senales = []
        skip = MACD_SLOW + MACD_SIGNAL
        ultima_senal_idx = -7
        for i in range(skip + 1, len(pendientes)):
            if (i - ultima_senal_idx) < 6:
                continue
            precio_idx = i + 1
            if precio_idx >= len(closes):
                continue
            ts = timestamps_ms[precio_idx]
            dir_4h = dir_4h_map.get(ts, "lateral")
            if dir_4h == "lateral":
                continue
            precio = float(closes[precio_idx])
            if precio <= 0:
                continue
            pend_actual = float(pendientes[i])
            pend_anterior = float(pendientes[i - 1])
            macd_val = float(macd_line[i])
            hist_val = float(histograma[i])
            macd_pct = abs(macd_val) / precio * 100
            dist_pct = abs(hist_val) / precio * 100
            if dist_pct < MONITOR_DIST_PCT:
                continue
            ts_iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            if dir_4h == "alcista" and macd_val < 0 and macd_pct >= MONITOR_ELONG_PCT:
                if pend_actual > 0 and pend_anterior <= 0:
                    senales.append({
                        "tipo": "LONG", "precio": round(precio, 6),
                        "timestamp": ts_iso,
                        "macd_15m": round(macd_val, 8),
                        "macd_pct": round(macd_pct, 4),
                        "dist_pct": round(dist_pct, 4),
                        "direccion_4h": dir_4h,
                    })
                    ultima_senal_idx = i
            elif dir_4h == "bajista" and macd_val > 0 and macd_pct >= MONITOR_ELONG_PCT:
                if pend_actual < 0 and pend_anterior >= 0:
                    senales.append({
                        "tipo": "SHORT", "precio": round(precio, 6),
                        "timestamp": ts_iso,
                        "macd_15m": round(macd_val, 8),
                        "macd_pct": round(macd_pct, 4),
                        "dist_pct": round(dist_pct, 4),
                        "direccion_4h": dir_4h,
                    })
                    ultima_senal_idx = i
        return senales
    except Exception as e:
        print(f"[MONITOR] Error detectando señales {symbol}: {e}")
        return []


def monitor_cycle():
    """Un ciclo de monitorización: detectar señales nuevas y grabarlas."""
    global _logged_timestamps, _monitor_initialized

    # Primera vez: cargar log existente de GitHub
    if not _monitor_initialized:
        existing_log, _ = gh_read_log()
        for entry in existing_log:
            ts = entry.get("timestamp")
            if ts:
                _logged_timestamps.add(ts)
        print(f"[MONITOR] Inicializado con {len(_logged_timestamps)} señales previas")
        _monitor_initialized = True

    nuevas = []
    for symbol in SYMBOLS:
        senales = detectar_senales_monitor(symbol)
        for s in senales:
            ts = s.get("timestamp")
            if ts and ts not in _logged_timestamps:
                entry = {
                    "timestamp": ts,
                    "symbol": symbol,
                    "tf": MONITOR_TF,
                    "tipo": s["tipo"],
                    "precio": s["precio"],
                    "macd_pct": s.get("macd_pct", 0),
                    "dist_pct": s.get("dist_pct", 0),
                    "elong_umbral": MONITOR_ELONG_PCT,
                    "dist_umbral": MONITOR_DIST_PCT,
                    "direccion_4h": s.get("direccion_4h", "desconocida"),
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                }
                nuevas.append(entry)
                _logged_timestamps.add(ts)

    if nuevas:
        # Leer log actual, añadir nuevas, escribir
        log_actual, sha = gh_read_log()
        log_actual.extend(nuevas)
        gh_write_log(log_actual, sha)
        print(f"[MONITOR] {len(nuevas)} señales nuevas grabadas")
    else:
        print(f"[MONITOR] Sin señales nuevas ({datetime.now(timezone.utc).strftime('%H:%M')} UTC)")


def monitor_loop():
    """Bucle de fondo que corre indefinidamente."""
    # Esperar 30s al arrancar para que Flask esté listo
    time.sleep(30)
    print(f"[MONITOR] Arrancando monitor de señales (intervalo: {MONITOR_INTERVAL}s)")
    while True:
        try:
            monitor_cycle()
        except Exception as e:
            print(f"[MONITOR] Error en ciclo: {e}")
        time.sleep(MONITOR_INTERVAL)


@app.route("/monitor/status")
def monitor_status():
    """Endpoint para verificar el estado del monitor."""
    return jsonify({
        "monitor_activo": bool(GH_TOKEN),
        "señales_registradas": len(_logged_timestamps),
        "intervalo_segundos": MONITOR_INTERVAL,
        "tf_monitoreado": MONITOR_TF,
        "logica": "relativa_al_precio",
        "elong_pct": MONITOR_ELONG_PCT,
        "dist_pct": MONITOR_DIST_PCT,
        "filtro_4h": "histograma",
        "repo": GH_REPO,
        "archivo": GH_LOG_FILE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/monitor/log")
def monitor_log():
    """Endpoint para ver el log de señales grabadas."""
    log_data, _ = gh_read_log()
    return jsonify({
        "n_señales": len(log_data),
        "señales": log_data,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })

# ─── FIN MONITOR ───

# ─── LABORATORIO ELONGACIÓN ───
# Testea diferentes umbrales de elongación del MACD 15m
# Lógica: entrada cuando MACD 15m está elongado EN CONTRA de 4H
#         y la pendiente gira A FAVOR de 4H

UMBRALES_ELONG = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]

CONFIGS_ELONG = {
    "escala_amplia": {"sl_pct": -0.02, "tp1_pct": 0.01, "tp1_peso": 0.40,
        "tp2_pct": 0.016, "tp2_peso": 0.30, "stop_tras_tp1_pct": -0.01},
    "escala_xl": {"sl_pct": -0.03, "tp1_pct": 0.015, "tp1_peso": 0.40,
        "tp2_pct": 0.025, "tp2_peso": 0.30, "stop_tras_tp1_pct": -0.015},
}


def get_4h_direction_for_15m(symbol, timestamps_15m_ms):
    """Para cada timestamp de 15m, determina dirección de la 4H (pendiente MACD)."""
    raw_4h = fetch_klines(symbol, "4h", 500)
    closes_4h = np.array([float(k[4]) for k in raw_4h])
    ts_open_4h = [int(k[0]) for k in raw_4h]
    ts_close_4h = [int(k[6]) for k in raw_4h]

    macd_4h, _, _ = calcular_macd(closes_4h)
    pend_4h = np.diff(macd_4h)

    dir_map = {}
    for ts_15m in timestamps_15m_ms:
        best_idx = len(ts_open_4h) - 1
        for j in range(len(ts_open_4h)):
            if ts_15m >= ts_open_4h[j] and ts_15m <= ts_close_4h[j]:
                best_idx = j
                break
            elif ts_15m < ts_open_4h[j]:
                best_idx = max(0, j - 1)
                break

        pend_idx = min(best_idx, len(pend_4h) - 1)
        if pend_idx > 0:
            pend_idx -= 1  # vela 4H anterior confirmada

        if pend_4h[pend_idx] > 0:
            dir_map[ts_15m] = "alcista"
        elif pend_4h[pend_idx] < 0:
            dir_map[ts_15m] = "bajista"
        else:
            dir_map[ts_15m] = "lateral"

    return dir_map


def detectar_senales_elongacion(closes, macd_line, pendientes, timestamps_ms, dir_4h_map, umbral_elong, gap_min=6):
    """
    Señales con lógica de elongación:
    - LONG: 4H alcista, MACD 15m < -umbral (elongado en contra), pendiente gira a positivo
    - SHORT: 4H bajista, MACD 15m > +umbral (elongado en contra), pendiente gira a negativo
    """
    senales = []
    skip = MACD_SLOW + MACD_SIGNAL
    ultima_senal_idx = -gap_min - 1

    for i in range(skip + 1, len(pendientes)):
        if (i - ultima_senal_idx) < gap_min:
            continue

        precio_idx = i + 1
        if precio_idx >= len(closes):
            continue

        ts = timestamps_ms[precio_idx]
        dir_4h = dir_4h_map.get(ts, "lateral")
        if dir_4h == "lateral":
            continue

        pend_actual = pendientes[i]
        pend_anterior = pendientes[i - 1]
        macd_val = macd_line[i]

        # LONG: 4H alcista, MACD 15m elongado negativo, pendiente gira positiva
        if dir_4h == "alcista" and macd_val < -umbral_elong:
            if pend_actual > 0 and pend_anterior <= 0:
                senales.append({
                    "tipo": "LONG",
                    "precio": round(float(closes[precio_idx]), 6),
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    "macd_15m": round(float(macd_val), 8),
                    "pendiente": round(float(pend_actual), 8),
                    "direccion_4h": dir_4h,
                    "vela_idx": int(precio_idx),
                })
                ultima_senal_idx = i

        # SHORT: 4H bajista, MACD 15m elongado positivo, pendiente gira negativa
        elif dir_4h == "bajista" and macd_val > umbral_elong:
            if pend_actual < 0 and pend_anterior >= 0:
                senales.append({
                    "tipo": "SHORT",
                    "precio": round(float(closes[precio_idx]), 6),
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    "macd_15m": round(float(macd_val), 8),
                    "pendiente": round(float(pend_actual), 8),
                    "direccion_4h": dir_4h,
                    "vela_idx": int(precio_idx),
                })
                ultima_senal_idx = i

    return senales


def simular_con_config(senales, closes, cfg_name, cfg):
    """Simula resultados de señales con una config de SL/TP usando velas de 15m."""
    resultados = []
    for s in senales:
        precio_entrada = s["precio"]
        tipo = s["tipo"]
        idx = s["vela_idx"]

        sl = cfg["sl_pct"]
        tp1 = cfg["tp1_pct"]
        tp2 = cfg["tp2_pct"]
        tp1_peso = cfg["tp1_peso"]
        tp2_peso = cfg["tp2_peso"]
        stop_tras_tp1 = cfg.get("stop_tras_tp1_pct", 0.0)

        max_velas = min(20, len(closes) - idx - 1)
        mejor_pct = 0.0
        tp1_tocado = False
        tp2_tocado = False
        resultado_parcial = 0.0
        peso_restante = 1.0
        cerrado = False

        for v in range(1, max_velas + 1):
            precio_actual = closes[idx + v]
            if tipo == "LONG":
                cambio = (precio_actual - precio_entrada) / precio_entrada
            else:
                cambio = (precio_entrada - precio_actual) / precio_entrada

            mejor_pct = max(mejor_pct, cambio)

            if cambio <= sl:
                resultado_parcial += peso_restante * sl
                cerrado = True
                break

            if not tp1_tocado and cambio >= tp1:
                tp1_tocado = True
                resultado_parcial += tp1_peso * tp1
                peso_restante -= tp1_peso
                sl = stop_tras_tp1

            if not tp2_tocado and cambio >= tp2:
                tp2_tocado = True
                resultado_parcial += tp2_peso * tp2
                peso_restante -= tp2_peso

            if v == max_velas:
                resultado_parcial += peso_restante * cambio
                cerrado = True

        resultados.append({
            "timestamp": s["timestamp"],
            "tipo": s["tipo"],
            "precio": s["precio"],
            "macd_15m": s.get("macd_15m", 0),
            "resultado_pct": round(resultado_parcial * 100, 3),
            "mejor_pct": round(mejor_pct * 100, 3),
            "ganadora": bool(resultado_parcial > 0),
        })

    return resultados


@app.route("/laboratorio_elongacion/<symbol>")
def laboratorio_elongacion(symbol):
    """Testea diferentes umbrales de elongación del MACD 15m."""
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400

    try:
        raw_15m = fetch_klines_extended(symbol, "15m", 3000)
        closes = np.array([float(k[4]) for k in raw_15m])
        timestamps_ms = [int(k[0]) for k in raw_15m]

        macd_line, _, _ = calcular_macd(closes)
        pendientes = calcular_pendientes(macd_line)

        dir_4h_map = get_4h_direction_for_15m(symbol, timestamps_ms)

        resultados = {}
        mejor_umbral = None
        mejor_resultado = -999

        for umbral in UMBRALES_ELONG:
            senales = detectar_senales_elongacion(
                closes, macd_line, pendientes, timestamps_ms, dir_4h_map, umbral
            )

            configs_result = {}
            for cfg_name, cfg in CONFIGS_ELONG.items():
                results = simular_con_config(senales, closes, cfg_name, cfg)
                n = len(results)
                if n == 0:
                    configs_result[cfg_name] = {"n_senales": 0, "winrate": 0, "resultado_medio": 0}
                    continue

                ganadoras = sum(1 for r in results if r["ganadora"])
                winrate = round(ganadoras / n * 100, 1)
                resultado_medio = round(sum(r["resultado_pct"] for r in results) / n, 3)
                total_pct = round(sum(r["resultado_pct"] for r in results), 3)

                configs_result[cfg_name] = {
                    "n_senales": n,
                    "winrate": winrate,
                    "resultado_medio": resultado_medio,
                    "total_pct": total_pct,
                    "senales": results,
                }

                if cfg_name == "escala_xl" and n >= 3 and resultado_medio > mejor_resultado:
                    mejor_resultado = resultado_medio
                    mejor_umbral = umbral

            resultados[str(umbral)] = {
                "umbral": umbral,
                "n_senales": len(senales),
                "configs": configs_result,
            }

        # Tabla resumen
        resumen = []
        for umbral in UMBRALES_ELONG:
            data = resultados[str(umbral)]
            xl = data["configs"].get("escala_xl", {})
            amp = data["configs"].get("escala_amplia", {})
            resumen.append({
                "umbral": umbral,
                "n_senales": data["n_senales"],
                "xl_winrate": xl.get("winrate", 0),
                "xl_resultado_medio": xl.get("resultado_medio", 0),
                "xl_total": xl.get("total_pct", 0),
                "amplia_winrate": amp.get("winrate", 0),
                "amplia_resultado_medio": amp.get("resultado_medio", 0),
                "amplia_total": amp.get("total_pct", 0),
            })

        return jsonify({
            "simbolo": symbol,
            "logica": "elongacion",
            "descripcion": "Entrada cuando MACD 15m elongado EN CONTRA de 4H y pendiente gira A FAVOR",
            "mejor_umbral_xl": mejor_umbral,
            "resumen": resumen,
            "detalle": resultados,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── FIN LABORATORIO ELONGACIÓN ───

# ─── LABORATORIO ELONGACIÓN 4H (fractal) ───
UMBRALES_ELONG_4H = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015]
UMBRAL_15M_FIJO = 0.0025


def calcular_zonas_4h(macd_4h, umbral_4h):
    """Zonas LONG/SHORT basadas en cruces del MACD 4H por cero, solo si venía de ±umbral."""
    zonas = []
    modo_actual = None
    max_positivo = 0.0
    min_negativo = 0.0
    for i in range(1, len(macd_4h)):
        val = float(macd_4h[i])
        val_prev = float(macd_4h[i - 1])
        if val > 0:
            max_positivo = max(max_positivo, val)
        elif val < 0:
            min_negativo = min(min_negativo, val)
        if val_prev <= 0 and val > 0:
            if abs(min_negativo) >= umbral_4h:
                if modo_actual != "LONG":
                    modo_actual = "LONG"
                    zonas.append({"inicio": i, "modo": "LONG"})
                min_negativo = 0.0
                max_positivo = val
        elif val_prev >= 0 and val < 0:
            if max_positivo >= umbral_4h:
                if modo_actual != "SHORT":
                    modo_actual = "SHORT"
                    zonas.append({"inicio": i, "modo": "SHORT"})
                max_positivo = 0.0
                min_negativo = val
    for j in range(len(zonas)):
        if j + 1 < len(zonas):
            zonas[j]["fin"] = zonas[j + 1]["inicio"]
        else:
            zonas[j]["fin"] = len(macd_4h)
    return zonas


def modo_4h_en_timestamp(zonas_4h, ts_open_4h, ts_close_4h, ts_15m):
    """Dado un timestamp de 15m, retorna el modo 4H (LONG/SHORT/None)."""
    idx_4h = len(ts_open_4h) - 1
    for j in range(len(ts_open_4h)):
        if ts_15m >= ts_open_4h[j] and ts_15m <= ts_close_4h[j]:
            idx_4h = j
            break
        elif ts_15m < ts_open_4h[j]:
            idx_4h = max(0, j - 1)
            break
    for z in zonas_4h:
        if z["inicio"] <= idx_4h < z["fin"]:
            return z["modo"]
    return None


@app.route("/laboratorio_elongacion_4h/<symbol>")
def laboratorio_elongacion_4h(symbol):
    """Testea umbrales de elongación del MACD 4H (lógica fractal)."""
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    try:
        raw_15m = fetch_klines_extended(symbol, "15m", 3000)
        closes_15m = np.array([float(k[4]) for k in raw_15m])
        timestamps_15m_ms = [int(k[0]) for k in raw_15m]
        macd_15m, _, _ = calcular_macd(closes_15m)
        pendientes_15m = calcular_pendientes(macd_15m)
        raw_4h = fetch_klines(symbol, "4h", 500)
        closes_4h = np.array([float(k[4]) for k in raw_4h])
        ts_open_4h = [int(k[0]) for k in raw_4h]
        ts_close_4h = [int(k[6]) for k in raw_4h]
        macd_4h, _, _ = calcular_macd(closes_4h)
        resultados = {}
        mejor_umbral = None
        mejor_resultado = -999
        for umbral_4h in UMBRALES_ELONG_4H:
            zonas = calcular_zonas_4h(macd_4h, umbral_4h)
            senales = []
            skip = MACD_SLOW + MACD_SIGNAL
            ultima_senal_idx = -7
            for i in range(skip + 1, len(pendientes_15m)):
                if (i - ultima_senal_idx) < 6:
                    continue
                precio_idx = i + 1
                if precio_idx >= len(closes_15m):
                    continue
                ts = timestamps_15m_ms[precio_idx]
                modo = modo_4h_en_timestamp(zonas, ts_open_4h, ts_close_4h, ts)
                if modo is None:
                    continue
                pend_actual = float(pendientes_15m[i])
                pend_anterior = float(pendientes_15m[i - 1])
                macd_val = float(macd_15m[i])
                if modo == "LONG" and macd_val < -UMBRAL_15M_FIJO:
                    if pend_actual > 0 and pend_anterior <= 0:
                        senales.append({
                            "tipo": "LONG",
                            "precio": round(float(closes_15m[precio_idx]), 6),
                            "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                            "macd_15m": round(macd_val, 8),
                            "modo_4h": modo,
                            "vela_idx": int(precio_idx),
                        })
                        ultima_senal_idx = i
                elif modo == "SHORT" and macd_val > UMBRAL_15M_FIJO:
                    if pend_actual < 0 and pend_anterior >= 0:
                        senales.append({
                            "tipo": "SHORT",
                            "precio": round(float(closes_15m[precio_idx]), 6),
                            "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                            "macd_15m": round(macd_val, 8),
                            "modo_4h": modo,
                            "vela_idx": int(precio_idx),
                        })
                        ultima_senal_idx = i
            configs_result = {}
            for cfg_name, cfg in CONFIGS_ELONG.items():
                results = simular_con_config(senales, closes_15m, cfg_name, cfg)
                n = len(results)
                if n == 0:
                    configs_result[cfg_name] = {"n_senales": 0, "winrate": 0, "resultado_medio": 0}
                    continue
                ganadoras = sum(1 for r in results if r["ganadora"])
                winrate = round(ganadoras / n * 100, 1)
                resultado_medio = round(sum(r["resultado_pct"] for r in results) / n, 3)
                total_pct = round(sum(r["resultado_pct"] for r in results), 3)
                configs_result[cfg_name] = {
                    "n_senales": n, "winrate": winrate,
                    "resultado_medio": resultado_medio, "total_pct": total_pct,
                    "senales": results,
                }
                if cfg_name == "escala_xl" and n >= 3 and resultado_medio > mejor_resultado:
                    mejor_resultado = resultado_medio
                    mejor_umbral = umbral_4h
            n_zonas_long = sum(1 for z in zonas if z["modo"] == "LONG")
            n_zonas_short = sum(1 for z in zonas if z["modo"] == "SHORT")
            resultados[str(umbral_4h)] = {
                "umbral_4h": umbral_4h, "umbral_15m_fijo": UMBRAL_15M_FIJO,
                "n_zonas_long": n_zonas_long, "n_zonas_short": n_zonas_short,
                "n_senales": len(senales), "configs": configs_result,
            }
        resumen = []
        for umbral_4h in UMBRALES_ELONG_4H:
            data = resultados[str(umbral_4h)]
            xl = data["configs"].get("escala_xl", {})
            amp = data["configs"].get("escala_amplia", {})
            resumen.append({
                "umbral_4h": umbral_4h, "n_senales": data["n_senales"],
                "zonas_L": data["n_zonas_long"], "zonas_S": data["n_zonas_short"],
                "xl_winrate": xl.get("winrate", 0),
                "xl_resultado_medio": xl.get("resultado_medio", 0),
                "xl_total": xl.get("total_pct", 0),
                "amplia_winrate": amp.get("winrate", 0),
                "amplia_resultado_medio": amp.get("resultado_medio", 0),
                "amplia_total": amp.get("total_pct", 0),
            })
        return jsonify({
            "simbolo": symbol, "logica": "elongacion_fractal_4h",
            "descripcion": "Fractal: umbral 4H para zonas LONG/SHORT (cruce por cero solo si venia de +-umbral) + 15m con elongacion 0.0025",
            "mejor_umbral_4h_xl": mejor_umbral, "resumen": resumen,
            "detalle": resultados, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── FIN LABORATORIO ELONGACIÓN 4H ───

# ─── LABORATORIO COMBINADO: ELONGACIÓN + DISTANCIA MACD-SEÑAL ───
UMBRALES_DISTANCIA = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]
UMBRALES_ELONG_COMBO = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004]


def get_4h_hist_direction(symbol, timestamps_15m_ms):
    """Dirección 4H basada en histograma (MACD - señal). Verde=alcista, Rojo=bajista."""
    raw_4h = fetch_klines(symbol, "4h", 500)
    closes_4h = np.array([float(k[4]) for k in raw_4h])
    ts_open_4h = [int(k[0]) for k in raw_4h]
    ts_close_4h = [int(k[6]) for k in raw_4h]
    macd_4h, sig_4h, _ = calcular_macd(closes_4h)
    hist_4h = macd_4h - sig_4h
    dir_map = {}
    for ts_15m in timestamps_15m_ms:
        idx_4h = len(ts_open_4h) - 1
        for j in range(len(ts_open_4h)):
            if ts_15m >= ts_open_4h[j] and ts_15m <= ts_close_4h[j]:
                idx_4h = j
                break
            elif ts_15m < ts_open_4h[j]:
                idx_4h = max(0, j - 1)
                break
        if idx_4h > 0:
            idx_4h -= 1
        if hist_4h[idx_4h] > 0:
            dir_map[ts_15m] = "alcista"
        elif hist_4h[idx_4h] < 0:
            dir_map[ts_15m] = "bajista"
        else:
            dir_map[ts_15m] = "lateral"
    return dir_map


def detectar_senales_combo(closes, macd_line, signal_line, pendientes, timestamps_ms, dir_4h_map, umbral_elong, umbral_distancia, gap_min=6):
    """Señales con elongación + distancia mínima MACD-señal."""
    senales = []
    skip = MACD_SLOW + MACD_SIGNAL
    ultima_senal_idx = -gap_min - 1
    histograma = macd_line - signal_line
    for i in range(skip + 1, len(pendientes)):
        if (i - ultima_senal_idx) < gap_min:
            continue
        precio_idx = i + 1
        if precio_idx >= len(closes):
            continue
        ts = timestamps_ms[precio_idx]
        dir_4h = dir_4h_map.get(ts, "lateral")
        if dir_4h == "lateral":
            continue
        pend_actual = float(pendientes[i])
        pend_anterior = float(pendientes[i - 1])
        macd_val = float(macd_line[i])
        hist_val = abs(float(histograma[i]))
        if hist_val < umbral_distancia:
            continue
        if dir_4h == "alcista" and macd_val < -umbral_elong:
            if pend_actual > 0 and pend_anterior <= 0:
                senales.append({
                    "tipo": "LONG",
                    "precio": round(float(closes[precio_idx]), 6),
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    "macd_15m": round(macd_val, 8),
                    "histograma_15m": round(float(histograma[i]), 8),
                    "direccion_4h": dir_4h,
                    "vela_idx": int(precio_idx),
                })
                ultima_senal_idx = i
        elif dir_4h == "bajista" and macd_val > umbral_elong:
            if pend_actual < 0 and pend_anterior >= 0:
                senales.append({
                    "tipo": "SHORT",
                    "precio": round(float(closes[precio_idx]), 6),
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    "macd_15m": round(macd_val, 8),
                    "histograma_15m": round(float(histograma[i]), 8),
                    "direccion_4h": dir_4h,
                    "vela_idx": int(precio_idx),
                })
                ultima_senal_idx = i
    return senales


@app.route("/laboratorio_combo/<symbol>")
def laboratorio_combo(symbol):
    """Testea combinaciones de elongación + distancia MACD-señal."""
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    try:
        raw_15m = fetch_klines_extended(symbol, "15m", 3000)
        closes = np.array([float(k[4]) for k in raw_15m])
        timestamps_ms = [int(k[0]) for k in raw_15m]
        macd_line, signal_line, _ = calcular_macd(closes)
        pendientes = calcular_pendientes(macd_line)
        dir_4h_map = get_4h_hist_direction(symbol, timestamps_ms)
        resultados = []
        mejor_combo = None
        mejor_resultado = -999
        for umbral_e in UMBRALES_ELONG_COMBO:
            for umbral_d in UMBRALES_DISTANCIA:
                senales = detectar_senales_combo(
                    closes, macd_line, signal_line, pendientes, timestamps_ms,
                    dir_4h_map, umbral_e, umbral_d
                )
                for cfg_name, cfg in CONFIGS_ELONG.items():
                    if cfg_name != "escala_xl":
                        continue
                    results = simular_con_config(senales, closes, cfg_name, cfg)
                    n = len(results)
                    if n == 0:
                        resultados.append({
                            "umbral_elong": umbral_e, "umbral_distancia": umbral_d,
                            "n_senales": 0, "xl_winrate": 0, "xl_media": 0, "xl_total": 0,
                        })
                        continue
                    ganadoras = sum(1 for r in results if r["ganadora"])
                    winrate = round(ganadoras / n * 100, 1)
                    media = round(sum(r["resultado_pct"] for r in results) / n, 3)
                    total = round(sum(r["resultado_pct"] for r in results), 3)
                    resultados.append({
                        "umbral_elong": umbral_e, "umbral_distancia": umbral_d,
                        "n_senales": n, "xl_winrate": winrate, "xl_media": media, "xl_total": total,
                    })
                    if n >= 3 and media > mejor_resultado:
                        mejor_resultado = media
                        mejor_combo = {"elong": umbral_e, "dist": umbral_d, "n": n,
                                       "wr": winrate, "media": media, "total": total}
        # Detalle del mejor combo
        mejor_detalle = None
        if mejor_combo:
            senales = detectar_senales_combo(
                closes, macd_line, signal_line, pendientes, timestamps_ms,
                dir_4h_map, mejor_combo["elong"], mejor_combo["dist"]
            )
            results = simular_con_config(senales, closes, "escala_xl", CONFIGS_ELONG["escala_xl"])
            mejor_detalle = {"senales": results}
        return jsonify({
            "simbolo": symbol,
            "logica": "combo_elongacion_distancia",
            "descripcion": "Elongación MACD 15m + distancia mínima MACD-señal 15m + histograma 4H",
            "mejor_combo": mejor_combo,
            "tabla": resultados,
            "mejor_detalle": mejor_detalle,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── FIN LABORATORIO COMBINADO ───

# ─── LABORATORIO RELATIVO: umbrales como % del precio ───
UMBRALES_ELONG_REL = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]  # en %
UMBRALES_DIST_REL = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # en %


def detectar_senales_relativo(closes, macd_line, signal_line, pendientes, timestamps_ms, dir_4h_map, umbral_elong_pct, umbral_dist_pct, gap_min=6):
    """Señales con umbrales relativos al precio."""
    senales = []
    skip = MACD_SLOW + MACD_SIGNAL
    ultima_senal_idx = -gap_min - 1
    histograma = macd_line - signal_line
    for i in range(skip + 1, len(pendientes)):
        if (i - ultima_senal_idx) < gap_min:
            continue
        precio_idx = i + 1
        if precio_idx >= len(closes):
            continue
        ts = timestamps_ms[precio_idx]
        dir_4h = dir_4h_map.get(ts, "lateral")
        if dir_4h == "lateral":
            continue
        precio = float(closes[precio_idx])
        if precio <= 0:
            continue
        pend_actual = float(pendientes[i])
        pend_anterior = float(pendientes[i - 1])
        macd_val = float(macd_line[i])
        hist_val = float(histograma[i])
        macd_pct = abs(macd_val) / precio * 100
        dist_pct = abs(hist_val) / precio * 100
        if dist_pct < umbral_dist_pct:
            continue
        if dir_4h == "alcista" and macd_val < 0 and macd_pct >= umbral_elong_pct:
            if pend_actual > 0 and pend_anterior <= 0:
                senales.append({
                    "tipo": "LONG",
                    "precio": round(precio, 6),
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    "macd_15m": round(macd_val, 8),
                    "macd_pct": round(macd_pct, 4),
                    "dist_pct": round(dist_pct, 4),
                    "direccion_4h": dir_4h,
                    "vela_idx": int(precio_idx),
                })
                ultima_senal_idx = i
        elif dir_4h == "bajista" and macd_val > 0 and macd_pct >= umbral_elong_pct:
            if pend_actual < 0 and pend_anterior >= 0:
                senales.append({
                    "tipo": "SHORT",
                    "precio": round(precio, 6),
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    "macd_15m": round(macd_val, 8),
                    "macd_pct": round(macd_pct, 4),
                    "dist_pct": round(dist_pct, 4),
                    "direccion_4h": dir_4h,
                    "vela_idx": int(precio_idx),
                })
                ultima_senal_idx = i
    return senales


@app.route("/laboratorio_relativo/<symbol>")
def laboratorio_relativo(symbol):
    """Testea combinaciones de elongacion + distancia como % del precio."""
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    try:
        raw_15m = fetch_klines_extended(symbol, "15m", 3000)
        closes = np.array([float(k[4]) for k in raw_15m])
        timestamps_ms = [int(k[0]) for k in raw_15m]
        macd_line, signal_line, _ = calcular_macd(closes)
        pendientes = calcular_pendientes(macd_line)
        dir_4h_map = get_4h_hist_direction(symbol, timestamps_ms)
        resultados = []
        mejor_combo = None
        mejor_resultado = -999
        for umbral_e in UMBRALES_ELONG_REL:
            for umbral_d in UMBRALES_DIST_REL:
                senales = detectar_senales_relativo(
                    closes, macd_line, signal_line, pendientes, timestamps_ms,
                    dir_4h_map, umbral_e, umbral_d
                )
                results = simular_con_config(senales, closes, "escala_xl", CONFIGS_ELONG["escala_xl"])
                n = len(results)
                if n == 0:
                    resultados.append({
                        "elong_pct": umbral_e, "dist_pct": umbral_d,
                        "n_senales": 0, "xl_winrate": 0, "xl_media": 0, "xl_total": 0,
                    })
                    continue
                ganadoras = sum(1 for r in results if r["ganadora"])
                winrate = round(ganadoras / n * 100, 1)
                media = round(sum(r["resultado_pct"] for r in results) / n, 3)
                total = round(sum(r["resultado_pct"] for r in results), 3)
                resultados.append({
                    "elong_pct": umbral_e, "dist_pct": umbral_d,
                    "n_senales": n, "xl_winrate": winrate, "xl_media": media, "xl_total": total,
                })
                if n >= 5 and media > mejor_resultado:
                    mejor_resultado = media
                    mejor_combo = {"elong_pct": umbral_e, "dist_pct": umbral_d, "n": n,
                                   "wr": winrate, "media": media, "total": total}
        mejor_detalle = None
        if mejor_combo:
            senales = detectar_senales_relativo(
                closes, macd_line, signal_line, pendientes, timestamps_ms,
                dir_4h_map, mejor_combo["elong_pct"], mejor_combo["dist_pct"]
            )
            results = simular_con_config(senales, closes, "escala_xl", CONFIGS_ELONG["escala_xl"])
            mejor_detalle = {"senales": results}
        return jsonify({
            "simbolo": symbol,
            "logica": "relativo_al_precio",
            "descripcion": "Umbrales como % del precio: se adaptan automaticamente a cualquier nivel de precio y activo",
            "mejor_combo": mejor_combo,
            "tabla": resultados,
            "mejor_detalle": mejor_detalle,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── FIN LABORATORIO RELATIVO ───

# ─── LABORATORIO RELATIVO SIN FILTRO 4H ───
@app.route("/laboratorio_relativo_sin4h/<symbol>")
def laboratorio_relativo_sin4h(symbol):
    """Testea umbrales relativos SIN filtro 4H."""
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    try:
        raw_15m = fetch_klines_extended(symbol, "15m", 3000)
        closes = np.array([float(k[4]) for k in raw_15m])
        timestamps_ms = [int(k[0]) for k in raw_15m]
        macd_line, signal_line, _ = calcular_macd(closes)
        pendientes = calcular_pendientes(macd_line)
        histograma = macd_line - signal_line
        resultados = []
        mejor_combo = None
        mejor_resultado = -999
        for umbral_e in UMBRALES_ELONG_REL:
            for umbral_d in UMBRALES_DIST_REL:
                senales = []
                skip = MACD_SLOW + MACD_SIGNAL
                ultima_idx = -7
                for i in range(skip + 1, len(pendientes)):
                    if (i - ultima_idx) < 6:
                        continue
                    precio_idx = i + 1
                    if precio_idx >= len(closes):
                        continue
                    precio = float(closes[precio_idx])
                    if precio <= 0:
                        continue
                    pend_actual = float(pendientes[i])
                    pend_anterior = float(pendientes[i - 1])
                    macd_val = float(macd_line[i])
                    hist_val = float(histograma[i])
                    macd_pct = abs(macd_val) / precio * 100
                    dist_pct = abs(hist_val) / precio * 100
                    if dist_pct < umbral_d:
                        continue
                    ts = timestamps_ms[precio_idx]
                    ts_iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
                    if macd_val < 0 and macd_pct >= umbral_e:
                        if pend_actual > 0 and pend_anterior <= 0:
                            senales.append({"tipo": "LONG", "precio": round(precio, 6),
                                "timestamp": ts_iso, "macd_pct": round(macd_pct, 4), "macd_15m": round(macd_val, 8),
                                "vela_idx": int(precio_idx)})
                            ultima_idx = i
                    elif macd_val > 0 and macd_pct >= umbral_e:
                        if pend_actual < 0 and pend_anterior >= 0:
                            senales.append({"tipo": "SHORT", "precio": round(precio, 6),
                                "timestamp": ts_iso, "macd_pct": round(macd_pct, 4), "macd_15m": round(macd_val, 8),
                                "vela_idx": int(precio_idx)})
                            ultima_idx = i
                results = simular_con_config(senales, closes, "escala_xl", CONFIGS_ELONG["escala_xl"])
                n = len(results)
                if n == 0:
                    resultados.append({"elong_pct": umbral_e, "dist_pct": umbral_d,
                        "n_senales": 0, "xl_winrate": 0, "xl_media": 0, "xl_total": 0})
                    continue
                ganadoras = sum(1 for r in results if r["ganadora"])
                winrate = round(ganadoras / n * 100, 1)
                media = round(sum(r["resultado_pct"] for r in results) / n, 3)
                total = round(sum(r["resultado_pct"] for r in results), 3)
                resultados.append({"elong_pct": umbral_e, "dist_pct": umbral_d,
                    "n_senales": n, "xl_winrate": winrate, "xl_media": media, "xl_total": total})
                if n >= 5 and media > mejor_resultado:
                    mejor_resultado = media
                    mejor_combo = {"elong_pct": umbral_e, "dist_pct": umbral_d, "n": n,
                                   "wr": winrate, "media": media, "total": total}
        mejor_detalle = None
        if mejor_combo:
            senales_det = []
            skip = MACD_SLOW + MACD_SIGNAL
            ultima_idx = -7
            for i in range(skip + 1, len(pendientes)):
                if (i - ultima_idx) < 6:
                    continue
                precio_idx = i + 1
                if precio_idx >= len(closes):
                    continue
                precio = float(closes[precio_idx])
                if precio <= 0:
                    continue
                pend_actual = float(pendientes[i])
                pend_anterior = float(pendientes[i - 1])
                macd_val = float(macd_line[i])
                hist_val = float(histograma[i])
                macd_pct = abs(macd_val) / precio * 100
                dist_pct = abs(hist_val) / precio * 100
                if dist_pct < mejor_combo["dist_pct"]:
                    continue
                ts = timestamps_ms[precio_idx]
                ts_iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
                if macd_val < 0 and macd_pct >= mejor_combo["elong_pct"]:
                    if pend_actual > 0 and pend_anterior <= 0:
                        senales_det.append({"tipo": "LONG", "precio": round(precio, 6),
                            "timestamp": ts_iso, "macd_pct": round(macd_pct, 4),
                            "vela_idx": int(precio_idx)})
                        ultima_idx = i
                elif macd_val > 0 and macd_pct >= mejor_combo["elong_pct"]:
                    if pend_actual < 0 and pend_anterior >= 0:
                        senales_det.append({"tipo": "SHORT", "precio": round(precio, 6),
                            "timestamp": ts_iso, "macd_pct": round(macd_pct, 4),
                            "vela_idx": int(precio_idx)})
                        ultima_idx = i
            results = simular_con_config(senales_det, closes, "escala_xl", CONFIGS_ELONG["escala_xl"])
            mejor_detalle = {"senales": results}
        return jsonify({
            "simbolo": symbol, "logica": "relativo_sin_filtro_4h",
            "descripcion": "Umbrales relativos SIN filtro 4H",
            "mejor_combo": mejor_combo, "tabla": resultados,
            "mejor_detalle": mejor_detalle,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── FIN LABORATORIO RELATIVO SIN 4H ───

# ─── LABORATORIO CONTINUACIÓN: entrar A FAVOR del movimiento del MACD ───
# Lógica: MACD llega a un extremo (elongado) y empieza a caer/subir desde ahí.
# SHORT: MACD ESTUVO positivo y elongado, ahora pendiente gira negativa → acompaña la caída
# LONG: MACD ESTUVO negativo y elongado, ahora pendiente gira positiva → acompaña la subida
UMBRALES_CONT_ELONG = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
UMBRALES_CONT_DIST = [0.0, 0.05, 0.1, 0.15, 0.2]


def detectar_senales_continuacion(closes, macd_line, signal_line, pendientes, timestamps_ms, umbral_elong_pct, umbral_dist_pct, gap_min=6):
    """Señales de continuación: entra A FAVOR del movimiento del MACD desde extremos."""
    senales = []
    skip = MACD_SLOW + MACD_SIGNAL
    ultima_senal_idx = -gap_min - 1
    histograma = macd_line - signal_line
    max_macd_pct = 0.0
    min_macd_pct = 0.0
    for i in range(skip + 1, len(pendientes)):
        precio_idx = i + 1
        if precio_idx >= len(closes):
            continue
        precio = float(closes[precio_idx])
        if precio <= 0:
            continue
        macd_val = float(macd_line[i])
        macd_pct_actual = macd_val / precio * 100
        if macd_pct_actual > 0:
            max_macd_pct = max(max_macd_pct, macd_pct_actual)
        elif macd_pct_actual < 0:
            min_macd_pct = min(min_macd_pct, macd_pct_actual)
        if (i - ultima_senal_idx) < gap_min:
            continue
        pend_actual = float(pendientes[i])
        pend_anterior = float(pendientes[i - 1])
        hist_val = float(histograma[i])
        dist_pct = abs(hist_val) / precio * 100
        if dist_pct < umbral_dist_pct:
            continue
        ts = timestamps_ms[precio_idx]
        # SHORT: MACD estuvo positivo y elongado, pendiente gira negativa
        if pend_actual < 0 and pend_anterior >= 0 and max_macd_pct >= umbral_elong_pct:
            senales.append({
                "tipo": "SHORT",
                "precio": round(precio, 6),
                "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                "macd_15m": round(macd_val, 8),
                "macd_pct": round(macd_pct_actual, 4),
                "max_alcanzado_pct": round(max_macd_pct, 4),
                "dist_pct": round(dist_pct, 4),
                "vela_idx": int(precio_idx),
            })
            ultima_senal_idx = i
            max_macd_pct = 0.0
        # LONG: MACD estuvo negativo y elongado, pendiente gira positiva
        elif pend_actual > 0 and pend_anterior <= 0 and abs(min_macd_pct) >= umbral_elong_pct:
            senales.append({
                "tipo": "LONG",
                "precio": round(precio, 6),
                "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                "macd_15m": round(macd_val, 8),
                "macd_pct": round(macd_pct_actual, 4),
                "min_alcanzado_pct": round(min_macd_pct, 4),
                "dist_pct": round(dist_pct, 4),
                "vela_idx": int(precio_idx),
            })
            ultima_senal_idx = i
            min_macd_pct = 0.0
    return senales


@app.route("/laboratorio_continuacion/<symbol>")
def laboratorio_continuacion(symbol):
    """Testea lógica de continuación: entrar A FAVOR del movimiento MACD."""
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    try:
        raw_15m = fetch_klines_extended(symbol, "15m", 3000)
        closes = np.array([float(k[4]) for k in raw_15m])
        timestamps_ms = [int(k[0]) for k in raw_15m]
        macd_line, signal_line, _ = calcular_macd(closes)
        pendientes = calcular_pendientes(macd_line)
        resultados = []
        mejor_combo = None
        mejor_resultado = -999
        for umbral_e in UMBRALES_CONT_ELONG:
            for umbral_d in UMBRALES_CONT_DIST:
                senales = detectar_senales_continuacion(
                    closes, macd_line, signal_line, pendientes, timestamps_ms,
                    umbral_e, umbral_d
                )
                results = simular_con_config(senales, closes, "escala_xl", CONFIGS_ELONG["escala_xl"])
                n = len(results)
                if n == 0:
                    resultados.append({
                        "elong_pct": umbral_e, "dist_pct": umbral_d,
                        "n_senales": 0, "xl_winrate": 0, "xl_media": 0, "xl_total": 0,
                    })
                    continue
                ganadoras = sum(1 for r in results if r["ganadora"])
                winrate = round(ganadoras / n * 100, 1)
                media = round(sum(r["resultado_pct"] for r in results) / n, 3)
                total = round(sum(r["resultado_pct"] for r in results), 3)
                resultados.append({
                    "elong_pct": umbral_e, "dist_pct": umbral_d,
                    "n_senales": n, "xl_winrate": winrate, "xl_media": media, "xl_total": total,
                })
                if n >= 5 and media > mejor_resultado:
                    mejor_resultado = media
                    mejor_combo = {"elong_pct": umbral_e, "dist_pct": umbral_d, "n": n,
                                   "wr": winrate, "media": media, "total": total}
        mejor_detalle = None
        if mejor_combo:
            senales = detectar_senales_continuacion(
                closes, macd_line, signal_line, pendientes, timestamps_ms,
                mejor_combo["elong_pct"], mejor_combo["dist_pct"]
            )
            results = simular_con_config(senales, closes, "escala_xl", CONFIGS_ELONG["escala_xl"])
            mejor_detalle = {"senales": results}
        return jsonify({
            "simbolo": symbol,
            "logica": "continuacion",
            "descripcion": "Continuacion: MACD llega a extremo elongado y gira. SHORT cuando cae desde arriba, LONG cuando sube desde abajo. Sin filtro 4H.",
            "mejor_combo": mejor_combo,
            "tabla": resultados,
            "mejor_detalle": mejor_detalle,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── FIN LABORATORIO CONTINUACIÓN ───

# Arrancar monitor al importar el módulo (funciona con gunicorn Y con python directo)
if GH_TOKEN:
    _monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    _monitor_thread.start()
    print(f"[MONITOR] Hilo de monitor arrancado")
else:
    print("[MONITOR] GITHUB_TOKEN no configurado — monitor desactivado")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
