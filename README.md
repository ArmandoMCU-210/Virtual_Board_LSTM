# Virtual Keyboard with Eye‑Tracking, Hand Gestures **and** LSTM Autocompletion

> **Proyecto Inteligencia Artificial - Universidad Veracruzana** — Interfaces de accesibilidad en Python + OpenCV/MediaPipe + TensorFlow + Flask

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Funciones Destacadas](#funciones-destacadas)
3. [Requerimientos](#requerimientos)
4. [Arquitectura General](#arquitectura-general)
5. [Instalación](#instalación)
6. [Entrenamiento del Modelo LSTM](#entrenamiento-del-modelo-lstm)
7. [Uso](#uso)

   * [Teclado virtual de escritorio](#teclado-virtual-de-escritorio)
   * [Demo web con Flask](#demo-web-con-flask)
8. [Estructura del Repositorio](#estructura-del-repositorio)
9. [Personalización y Mejora](#personalización-y-mejora)
10. [Créditos](#créditos)
11. [Licencia](#licencia)

---

## Introducción

Este repositorio engloba **dos proyectos independientes pero complementarios** que comparten el mismo modelo de autocompletado basado en LSTM:

| Proyecto                         | Nombre sugerido               | Tecnologías clave                     | Descripción breve                                                                                                                                             |
| -------------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Proyecto I – “EyeKey”**        | Teclado Virtual de Escritorio | Python · OpenCV · MediaPipe · Pyttsx3 | Permite escribir en un teclado QWERTY en pantalla mediante seguimiento ocular (parpadeo) y gestos de mano. Incluye autocompletado de palabras en tiempo real. |
| **Proyecto II – “LSTM‑WebDemo”** | API Flask + Frontend JS       | Flask · HTML · JavaScript             | Expone una API REST `/predict` y una página web que demuestra la misma capacidad de completado, ideal para integración en otras apps.                         |

Cada proyecto puede usarse por separado o en conjunto (por ejemplo, el **Proyecto I** para accesibilidad en escritorio y el **Proyecto II** para aplicaciones web). Ambos se alimentan del mismo modelo (`autocomplete_es.h5`) y del tokenizador (`tokenizer.pkl`).

\----------|-------------|-------------|
\| **Teclado virtual de escritorio** | OpenCV · MediaPipe · Pyttsx3 | Controla un teclado QWERTY en pantalla mediante la mirada, parpadeo y gestos de mano. Incluye autocompletado LSTM en tiempo real. |
\| **Demo web** | Flask · HTML · JS | Campo de texto simple que muestra la misma capacidad de autocompletar palabras usando el modelo LSTM servido vía API. |

La meta principal es **facilitar la comunicación** a personas con movilidad reducida o necesidades especiales, manteniendo todo **100 % local** (sin enviar datos a la nube).

---

## Funciones Destacadas

| Archivo                                 | Propósito                                                                               |
| --------------------------------------- | --------------------------------------------------------------------------------------- |
| `vir_keyboard_eye_det.py`               | Programa principal de escritorio: integra cámara, UI OpenCV y lógica de selección.      |
| `modules/detecting_eye_blink_module.py` | Detección de parpadeo con MediaPipe FaceMesh.                                           |
| `modules/tracking_hand_module.py`       | Reconoce gestos de mano (espacio, borrar, tab, guardar).                                |
| `modules/autocompleter_module.py`       | Carga del modelo `autocomplete_es.h5` + tokenizador para predecir la siguiente palabra. |
| `data/spanish_corpus.txt`               | Pequeño corpus de ejemplo (puedes añadir tu propio dataset).                            |
| `train_model.py`                        | Script para entrenar/re‑entrenar el modelo LSTM desde cero.                             |
| **Web demo**                            |                                                                                         |
| `server.py`                             | API Flask que expone `/predict` y sirve la página de ejemplo.                           |
| `index.html`                            | Página con un `<input>` conectado vía *fetch* a la API.                                 |
| `static/script.js`                      | Lógica JS que detecta `Tab` para autocompletar.                                         |

---

## Requerimientos

### Proyecto I – EyeKey (Teclado Virtual de Escritorio)

**Hardware mínimo**
• Cámara web HD (≥ 720p @ 30 fps)
• CPU con soporte AVX; GPU NVIDIA (opcional) para acelerar TensorFlow
• Windows, Linux o macOS

**Dependencias Python** (`requirements_eye_key.txt`):

```text
opencv-python>=4.9
mediapipe>=0.10
numpy>=1.24
tensorflow>=2.15
h5py>=3.10
pyttsx3>=2.90
```

Instalar solo EyeKey:

```bash
pip install -r requirements_eye_key.txt
```

---

### Proyecto II – LSTM‑WebDemo (API Flask + Frontend JS)

**Dependencias Python** (`requirements_webdemo.txt`):

```text
flask>=3.0
tensorflow>=2.15
numpy>=1.24
h5py>=3.10
gunicorn>=22.0   # opcional para producción
```

Instalar solo LSTM‑WebDemo:

```bash
pip install -r requirements_webdemo.txt
```

> **Nota**: El modelo `autocomplete_es.h5` y `tokenizer.pkl` deben ubicarse en la carpeta `model/` accesible por ambos proyectos.

---

## Arquitectura General

```
┌────────────┐  frames  ┌────────────────────┐         ┌────────────┐
│  Webcam    │────────▶│  OpenCV + MediaPipe │────────▶│  Lógica    │
└────────────┘          │  (parpadeo & mano) │         │  Teclado   │
                        └─────────┬──────────┘         └─────┬──────┘
                                  │ texto                  │ sugerencia
                                  ▼                       ▼
                            ┌────────────┐        ┌────────────────────┐
                            │  LSTM      │        │  UI (OpenCV)       │
                            └────────────┘        └────────────────────┘
```

Para la **demo web** el modelo se reutiliza en el backend (**Flask**):

```
Usuario 🔄 JS fetch 🔄 Flask 🔄 TensorFlow LSTM
```

---

## Instalación

### 1 · Clonar repositorio

```bash
git clone https://github.com/ArmandoMCU-210/Virtual_Board_LSTM.git
cd Virtual_Board_LSTM
```

### 2 · Crear entorno virtual (opcional)

```bash
python -m venv EyeTracking
source EyeTracking/bin/activate  # Windows: EyeTracking\Scripts\activate
```

### 3 · Instalar dependencias

**Para Proyecto I – EyeKey**

```bash
pip install -r requirements_eye_key.txt
```

**Para Proyecto II – LSTM‑WebDemo**

```bash
pip install -r requirements_webdemo.txt
```

**Para ambos proyectos (todo‑en‑uno)**

```bash
pip install -r requirements.txt  # incluye todas las dependencias
```

---

## Entrenamiento del Modelo LSTM

Para mejorar el autocompletado con tu propio corpus:

```bash
python train_model.py   # Ajusta SEQ_LENGTH y EPOCHS si lo deseas
```

El script genera:

* `model/autocomplete_es.h5`
* `model/tokenizer.pkl`

Reemplázalos en producción cuando termines.

---

## Uso

### Teclado virtual de escritorio

```bash
python vir_keyboard_eye_det.py
```

Controles rápidos:

| Acción            | Gesto / Evento                                       |
| ----------------- | ---------------------------------------------------- |
| Seleccionar tecla | Mirar 1 s + parpadeo                                 |
| Autocompletar     | Parpadeo sobre **Tab** o gesto de mano índice ↘      |
| Espacio           | Parpadeo sobre **Space** o mano abierta              |
| Borrar            | Parpadeo sobre **Backspace** o mano con puño cerrado |
| Guardar texto     | Parpadeo sobre **Save**                              |
| Salir             | Pulsar **q** en teclado físico                       |

### Demo web con Flask

1. Lanza el backend:

   ```bash
   python server.py
   ```

   Por defecto abre [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. Escribe en el cuadro de texto; pulsa **Tab** para completar la palabra sugerida.

> **Flujo**: `script.js` envía la última palabra a `/predict`; Flask responde con JSON `{ suggestion: "..." }`.

---

## Estructura del Repositorio

La carpeta raíz contiene **dos proyectos independientes**.

```
├── TEST_04‑Autocompletador_LSTM/              # Proyecto II – LSTM‑WebDemo
│   ├── data/
│   │   └── spanish_corpus.txt
│   ├── model/
│   │   ├── autocomplete_es.h5
│   │   └── tokenizer.pkl
│   ├── static/
│   │   └── script.js
│   ├── templates/
│   │   └── index.html
│   ├── server.py                              # API Flask + página de demo
│   └── train_model.py                         # Entrenamiento del modelo
│
└── TEST_05‑Virtual_Board/                     # Proyecto I – EyeKey
    ├── data/
    ├── model/
    ├── autocompleter_module.py                # Carga del modelo y predicción
    ├── detecting_eye_blink_module.py          # Detección de parpadeo
    ├── tracking_hand_module.py                # Detección de gestos de mano
    ├── vir_keyboard_eye_det.py                # Programa principal del teclado
    ├── train_model.py                         # (opcional) re‑entrenar desde VS
    ├── click.mp3 | sclick.mp3                 # Sonidos opcionales
    └── output.txt                             # Texto guardado por el usuario
```

> Carpeta `__pycache__/` y archivos generados (
> `*.pyc`, entornos `venv/`) deben excluirse vía `.gitignore`.

