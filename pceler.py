"""
PCELER — Acelerómetro PALMERO (v2.0)
========================================
Servicio independiente que estudia la pendiente (ángulo de ataque) de la
línea MACD (azul, rápida) por timeframe para XRP y SOL.

Objetivo: anticipar giros de tendencia ANTES de que el precio los confirme,
cubriendo el vacío de las zonas grises de PALMERO 15 donde el sistema
principal dice "manos quietas".

Tres capas:
  1. CALIBRACIÓN — estadísticas históricas de pendientes: percentiles,
     extremos, distribución.
  2. ESTADO EN TIEMPO REAL — pendiente actual del MACD en cada TF,
     en qué percentil se encuentra, si está en zona de apogeo o giro.
  3. SEÑALES HISTÓRICAS — simulación retrospectiva con distintos umbrales.

v2.0: REGLA DE HIERRO integrada. Las señales en TFs menores (5m, 15m, 1h)
se filtran por la dirección del MACD 4H en el momento exacto de la señal.
Solo pasan LONGs cuando 4H sube, SHORTs cuando 4H baja.
Muestra resultados filtrados y sin filtrar para comparar.

No depende de ningún otro servicio PALMERO. Lee directamente de Binance.
"""

import os
import time
import requests
import numpy as np
from datetime import datetime, timezone
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

    pendiente_subiendo = actual > anterior
    pendiente_bajando = actual < anterior

    percentil_actual = float(
        np.searchsorted(np.sort(pendientes), actual) / len(pendientes) * 100
    )

    return {
        "pendiente_actual": round(actual, 8),
        "pendiente_anterior": round(anterior, 8),
        "pendiente_subiendo": bool(pendiente_subiendo),
        "pendiente_bajando": bool(pendiente_bajando),
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


def simular_senales(pendientes, closes, timestamps, percentiles, umbral):
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
                        "tipo": "SHORT",
                        "motivo": "giro_bajista_desde_apogeo",
                        "vela_idx": int(precio_idx),
                        "precio": round(float(closes[precio_idx]), 6),
                        "timestamp": timestamps[precio_idx] if precio_idx < len(timestamps) else None,
                        "pendiente_en_giro": round(p, 8),
                        "max_pendiente": round(max_pendiente_alcista, 8),
                        "retroceso_pct": round(float(retroceso), 4),
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
                        "tipo": "LONG",
                        "motivo": "giro_alcista_desde_apogeo",
                        "vela_idx": int(precio_idx),
                        "precio": round(float(closes[precio_idx]), 6),
                        "timestamp": timestamps[precio_idx] if precio_idx < len(timestamps) else None,
                        "pendiente_en_giro": round(p, 8),
                        "min_pendiente": round(min_pendiente_bajista, 8),
                        "retroceso_pct": round(float(retroceso), 4),
                        "umbral_usado": float(umbral),
                    })
                    ultima_senal_idx = i
                en_apogeo_bajista = False
                min_pendiente_bajista = 0.0

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
        if s["tipo"] == "SHORT":
            mejor_precio = float(np.min(precios_futuros))
        else:
            mejor_precio = float(np.max(precios_futuros))
        mejor_pct = dir_mult * (mejor_precio - entrada) / entrada * 100
        cierre_20v = float(closes[idx + n_velas_futuras])
        resultado_20v = dir_mult * (cierre_20v - entrada) / entrada * 100

        resultado = {
            "tipo": s["tipo"],
            "motivo": s["motivo"],
            "precio": s["precio"],
            "timestamp": s["timestamp"],
            "umbral_usado": s["umbral_usado"],
            "retroceso_pct": s["retroceso_pct"],
            "mejor_pct_20v": round(float(mejor_pct), 3),
            "resultado_20v_pct": round(float(resultado_20v), 3),
            "ganadora_20v": bool(resultado_20v > 0),
        }
        if "direccion_4h" in s:
            resultado["direccion_4h"] = s["direccion_4h"]
        resultados.append(resultado)
    return resultados


def resumir_evaluacion(evaluadas):
    if not evaluadas:
        return {
            "n_senales": 0,
            "winrate_pct": None,
            "resultado_medio_pct": None,
            "mejor_medio_pct": None,
        }
    ganadoras = sum(1 for s in evaluadas if s["ganadora_20v"])
    return {
        "n_senales": int(len(evaluadas)),
        "winrate_pct": round(float(ganadoras / len(evaluadas) * 100), 1),
        "resultado_medio_pct": round(
            float(sum(s["resultado_20v_pct"] for s in evaluadas) / len(evaluadas)), 3
        ),
        "mejor_medio_pct": round(
            float(sum(s["mejor_pct_20v"] for s in evaluadas) / len(evaluadas)), 3
        ),
    }


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
        "calibracion": percentiles,
        "estado": estado,
        "n_velas_analizadas": int(len(pendientes_validas)),
    }


def analizar_senales_tf(symbol, tf_label, interval, limit):
    raw = fetch_klines(symbol, interval, limit)
    if len(raw) < MACD_SLOW + MACD_SIGNAL + 30:
        return {"error": "datos_insuficientes"}

    closes = np.array([float(k[4]) for k in raw])
    timestamps = [
        datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).isoformat()
        for k in raw
    ]

    macd_line, signal_line, histogram = calcular_macd(closes)
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
                senales_temp = simular_senales(
                    pendientes_validas, closes_validas, timestamps_validos, percentiles, umbral
                )
                all_timestamps.extend([s["timestamp"] for s in senales_temp if s["timestamp"]])
            all_timestamps = list(set(all_timestamps))
            if all_timestamps:
                direccion_4h = obtener_direccion_4h(symbol, all_timestamps)
        except Exception as e:
            print(f"Error obteniendo dirección 4H: {e}")

    resultados_por_umbral = {}
    for umbral in UMBRALES_GIRO:
        senales = simular_senales(
            pendientes_validas, closes_validas, timestamps_validos, percentiles, umbral
        )
        evaluadas_sin_filtro = evaluar_senales(senales, closes_validas)
        resumen_sin_filtro = resumir_evaluacion(evaluadas_sin_filtro)
        resumen_sin_filtro["senales"] = evaluadas_sin_filtro[-10:]

        resultado_umbral = {
            "umbral": float(umbral),
            "sin_filtro": resumen_sin_filtro,
        }

        if es_tf_menor and direccion_4h:
            senales_filtradas = filtrar_por_4h(senales, direccion_4h)
            evaluadas_filtradas = evaluar_senales(senales_filtradas, closes_validas)
            resumen_filtrado = resumir_evaluacion(evaluadas_filtradas)
            resumen_filtrado["senales"] = evaluadas_filtradas[-10:]
            resultado_umbral["filtro_4h"] = resumen_filtrado

        resultados_por_umbral[str(umbral)] = resultado_umbral

    return {
        "tf": tf_label,
        "n_velas": int(len(closes)),
        "filtro_4h_aplicado": bool(es_tf_menor and direccion_4h),
        "umbrales": resultados_por_umbral,
    }


@app.route("/")
def home():
    return jsonify({
        "servicio": "PCELER — Acelerómetro PALMERO",
        "version": "2.0",
        "novedad": "Regla de hierro 4H integrada en señales de TFs menores",
        "descripcion": "Estudio de pendientes y aceleración de la línea MACD para anticipar giros de tendencia",
        "endpoints": [
            "/estado/<symbol> — estado actual: pendiente, percentil, apogeo, giro (todos los TFs)",
            "/estado/<symbol>/<tf> — estado de un TF concreto (4h, 1h, 15m, 5m)",
            "/calibracion/<symbol> — estadísticas históricas de pendientes por TF",
            "/senales/<symbol> — simulación de señales históricas con filtro 4H",
            "/senales/<symbol>/<tf> — señales de un TF concreto con filtro 4H",
            "/t/<bust> — versión sin caché de /estado de todos los símbolos",
        ],
    })


@app.route("/estado/<symbol>")
def estado_symbol(symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    resultado = {
        "simbolo": symbol,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }
    for label, cfg in TIMEFRAMES.items():
        try:
            resultado["timeframes"][label] = analizar_tf(symbol, label, cfg["interval"], cfg["limit"])
        except Exception as e:
            resultado["timeframes"][label] = {"error": str(e)}
    return jsonify(resultado)


@app.route("/estado/<symbol>/<tf>")
def estado_symbol_tf(symbol, tf):
    symbol = symbol.upper()
    tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        data = analizar_tf(symbol, tf, cfg["interval"], cfg["limit"])
        return jsonify({
            "simbolo": symbol,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/calibracion/<symbol>")
def calibracion_symbol(symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    resultado = {
        "simbolo": symbol,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }
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


@app.route("/senales/<symbol>")
def senales_symbol(symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    resultado = {
        "simbolo": symbol,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }
    for label, cfg in TIMEFRAMES.items():
        try:
            resultado["timeframes"][label] = analizar_senales_tf(
                symbol, label, cfg["interval"], cfg["limit"]
            )
        except Exception as e:
            resultado["timeframes"][label] = {"error": str(e)}
    return jsonify(resultado)


@app.route("/senales/<symbol>/<tf>")
def senales_symbol_tf(symbol, tf):
    symbol = symbol.upper()
    tf = tf.lower()
    if symbol not in SYMBOLS:
        return jsonify({"error": f"simbolo no soportado: {symbol}"}), 400
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"TF no soportado: {tf}"}), 400
    cfg = TIMEFRAMES[tf]
    try:
        data = analizar_senales_tf(symbol, tf, cfg["interval"], cfg["limit"])
        return jsonify({
            "simbolo": symbol,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **data,
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
