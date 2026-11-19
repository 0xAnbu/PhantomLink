# PHANTOMLINK — AI vs AI

> **Entrené dos inteligencias artificiales desde cero para que compitan entre ellas.
> Una inventa dominios. La otra intenta detectar si parecen de phishing.**

| 🔴 Generador | 🔵 Detector |
|---|---|
| Transformer *decoder-only* a nivel de carácter (PyTorch), escrito desde cero | `RandomForestClassifier` (scikit-learn) sobre 19 características léxicas |
| ~107k parámetros, 2 capas, 4 cabezas, 64 dims | 100 árboles, profundidad 12 |
| Perplejidad de validación 42,5 → **14,9** | **F1 0,785 · AUC 0,862** en test no visto |

Laboratorio interactivo con tres pantallas:

- **AI Arena**: el generador inventa N dominios, se validan sintácticamente, el detector los
  juzga y se muestra un resumen. Temperatura, top-K y umbral se ajustan en tiempo real, y las
  rondas se pueden comparar. Incluye un modo **«Tú contra la IA»** con dominios reales de test.
- **Model Inspector**: generación carácter a carácter con las probabilidades reales,
  la entropía, los mapas de atención de cada cabeza y las activaciones internas. Del lado del
  detector: la contribución exacta de cada característica (`base + Σ contribuciones = predict_proba`),
  los votos de los árboles y la importancia global (impureza y permutación).
- **Training Lab**: entrena los dos modelos desde el navegador, con telemetría en directo por
  SSE (loss de train y validación, perplejidad, norma del gradiente, learning rate, tokens/s,
  memoria, matriz de confusión, barrido de umbral) y presets **Fast / Balanced / Break it**.

Todas las cifras y gráficas salen de los modelos. Nada es simulado.

🎬 **Demo (42 s):** [`docs/demo.webm`](docs/demo.webm). Se graba automáticamente contra la app en
marcha con `python scripts/record_demo.py` (requiere `pip install playwright` y Chromium).

## Inicio rápido

Requisitos: Python ≥ 3.10, Node ≥ 18. Funciona en CPU; CUDA es opcional.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # o la build con CUDA
.venv/bin/pip install -r requirements.txt
./run.sh                    # descarga y prepara los datos si hace falta, compila la UI y arranca
# → http://127.0.0.1:8000
```

`checkpoints/` incluye un modelo de ejemplo de cada tipo, así que la Arena funciona nada más
arrancar. Para reentrenar desde la línea de comandos:

```bash
.venv/bin/python scripts/prepare_data.py --download          # ~3 s tras la descarga
.venv/bin/python scripts/train_classifier.py                 # ~10 s
.venv/bin/python scripts/train_generator.py --iterations 6000 --batch-size 64 --eval-interval 500   # ~8 min en CPU
.venv/bin/python -m pytest -q                                # 32 pruebas
```

Para desarrollar el frontend con recarga en caliente, ejecuta `cd frontend && npm install && npm run dev`
(en http://localhost:5173, con el proxy de `/api` apuntando al backend en el puerto 8000).

## Arquitectura

```
React + Vite + Tailwind + Recharts  ──HTTP/JSON + SSE──▶  FastAPI (backend/phantomlink/api.py)
                                                        ├─ generator/  MiniGPT: model, train, sampling, checkpoint
                                                        ├─ classifier.py + features.py  RandomForest
                                                        ├─ arena.py    generar → validar → clasificar → resumir
                                                        ├─ jobs.py     un entrenamiento a la vez, parada, eventos
                                                        └─ data.py     normalización y división del dataset
```

Endpoints principales: `POST /api/arena`, `POST /api/inspect/generator`, `POST /api/inspect/classifier`,
`POST /api/train/{generator|classifier}`, `POST /api/train/stop`, `GET /api/jobs/{id}/events` (SSE),
`GET /api/status`, `GET /api/models`, `POST /api/models/activate`, `GET /api/challenge`.
La documentación interactiva está en `/docs`.

Protecciones: todas las configuraciones se validan (con un tope de ~2M de parámetros, 20.000 iteraciones
y 500 árboles), solo puede haber un entrenamiento simultáneo (HTTP 409 si se intenta otro),
los entrenamientos se pueden detener, los modelos que divergen (loss no finita) no se guardan
y los checkpoints se cargan con `torch.load(weights_only=True)`.

## Datos

[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
(UCI, id 967, CC BY 4.0): 235.795 URLs.

- **Etiquetas**: en el CSV original `label = 1` es **legítimo** y `label = 0` es **phishing**.
  Se mapea una única vez a `is_phishing`.
- **Solo se usa el nombre del dominio.** Se descartan los atributos del contenido de la página,
  porque la demo nunca visita sitios.
- **Fuga eliminada**: el 100 % de los hosts legítimos empieza por `www.`, frente al 41 % de los de
  phishing, así que se quita `www.`. También se quita el sufijo público (TLD): los dominios generados
  se muestran con el TLD reservado `.test` (RFC 2606), así que el detector juzga lo mismo en
  entrenamiento y en la Arena.
- **División por dominio registrado** (eTLD+1, Public Suffix List incluida, sin red) mediante un
  hash estable: 70/15/15. Ningún dominio registrado aparece en dos conjuntos (lo comprueba un assert).
  Las plataformas de alojamiento (`web.app`, `firebaseapp.com`, `repl.co`…) cuentan como un solo
  grupo cada una, lo que hace la evaluación más estricta.

| Conjunto | Nombres | Phishing | Legítimos |
|---|--:|--:|--:|
| train | 139.897 | 47.902 | 91.995 |
| val | 38.863 | 19.276 | 19.587 |
| test | 34.091 | 14.589 | 19.502 |

Tras eliminar IPs, nombres con etiquetas contradictorias y duplicados quedan 212.851 nombres.
Los detalles están en `data/processed/split_report.json`.

## Resultados (checkpoints incluidos, semilla 42)

**Detector, test no visto (umbral 0,5):** accuracy 0,833 · precision 0,878 · recall 0,709 · F1 0,785 · AUC 0,862.
Matriz de confusión: VN 18.070 · FP 1.432 · FN 4.245 · VP 10.344.
La señal más importante es el número de etiquetas (subdominios), seguida de dígitos y entropía.

**Generador:** loss de validación 3,75 → 2,70 (perplejidad 42,5 → 14,9) en 6.000 iteraciones
(~8 min en CPU). El token de estilo funciona como condicionamiento real. Con T = 0,8 y 500
nombres por estilo:

| Estilo pedido | Marcados por el detector | P media | Subdominios medios | Dígitos medios |
|---|--:|--:|--:|--:|
| legit | 1 % | 0,30 | 1,00 | 0,00 |
| phish | 47 % | 0,59 | 1,57 | 0,49 |

El generador aprendió por su cuenta patrones del phishing alojado en plataformas
(`….web`, `f0554.xsph`) y de suplantación de marca (`…-fb-co-jp.amazocn…`).

**Break it** (learning rate 1,0, sin clipping): la loss se dispara de 3,7 a más de 100, con normas de
gradiente de unas 70, y después se estanca en ≈3,3. El detector *Break it* (árboles sin límite y 2.000
ejemplos) sobreajusta: accuracy de 0,94 en train frente a 0,83 en validación.

## Limitaciones

- El detector solo analiza la **forma del nombre**: no ve el TLD, el contenido, los certificados
  ni la reputación. Que diga **«no sospechoso» no significa que el dominio sea seguro**. Con el
  umbral por defecto, alrededor del 29 % del phishing de test pasa sin ser detectado.
- El dataset tiene sesgos de recogida. Se han neutralizado los conocidos (`www.`, TLD, URL completa),
  pero puede haber otros.
- El generador es diminuto y puede reproducir nombres que ya existen en el dataset (la Arena los
  marca como «memorizado»).
- En la Arena, el estilo pedido al generador no es una verdad de referencia: un dominio «estilo
  phish» marcado como sospechoso indica que los dos modelos coinciden, no que se haya verificado
  phishing. Por eso no se presenta como accuracy. El modo «Tú contra la IA» usa etiquetas reales.
- No hay aprendizaje por refuerzo ni reentrenamiento adversarial, y la aplicación nunca hace
  peticiones de red a los dominios generados.

## Estructura

```
backend/phantomlink/   API, modelos, datos, Arena, jobs
scripts/               prepare_data.py · train_classifier.py · train_generator.py
tests/                 pytest (dataset sintético diminuto + comprobaciones sobre el real)
checkpoints/           generator_default.pt · classifier_default.joblib
frontend/              React 19 · Vite · Tailwind CSS 4 · Recharts
data/processed/        split_report.json (domains.csv se regenera con prepare_data.py)
```

## Licencias y créditos

Código: desarrollado desde cero para este proyecto. Datos: PhiUSIIL Phishing URL Dataset —
Prasad, A. & Chandra, S. (2024), UCI Machine Learning Repository, CC BY 4.0.
