# PHANTOMLINK — Documentación técnica completa y diario de desarrollo

> Documento interno. Vive en `/home/kali/Claude/DOCUMENTATION.md`, **fuera** del repositorio
> `phantomlink/` (el repositorio git no lo contiene).
> Objetivo: que una persona **sin conocimientos técnicos** entienda qué se ha construido, por qué
> se tomó cada decisión y cómo se hizo paso a paso; y que una persona técnica encuentre aquí
> todos los detalles (cifras, parámetros, problemas encontrados y cómo se resolvieron).
>
> El proyecto está terminado. El diario conserva las decisiones y mediciones del desarrollo.
> El apartado 11 sirve como manual de uso de la aplicación.

---

## Índice

1. [Qué es PHANTOMLINK, explicado sin tecnicismos](#1-qué-es-phantomlink-explicado-sin-tecnicismos)
2. [Glosario para no técnicos](#2-glosario-para-no-técnicos)
3. [Documentos de partida y reglas del proyecto](#3-documentos-de-partida-y-reglas-del-proyecto)
4. [Arquitectura general](#4-arquitectura-general)
5. [Diario de desarrollo paso a paso](#5-diario-de-desarrollo-paso-a-paso)
6. [Los datos: de dónde salen y cómo se preparan](#6-los-datos-de-dónde-salen-y-cómo-se-preparan)
7. [Modelo B: el detector (Random Forest)](#7-modelo-b-el-detector-random-forest)
8. [Modelo A: el generador (Mini Transformer)](#8-modelo-a-el-generador-mini-transformer)
9. [La Arena: cómo interactúan los dos modelos](#9-la-arena-cómo-interactúan-los-dos-modelos)
10. [El backend (servidor FastAPI)](#10-el-backend-servidor-fastapi)
11. [El frontend (interfaz)](#11-el-frontend-interfaz)
    - 11.A [Manual de interacciones: cada control explicado](#11a-manual-de-interacciones-cada-control-explicado)
        - [AI Arena](#11a1-ai-arena)
        - [Model Inspector · generador](#11a2-model-inspector--generador-cómo-decide-cada-carácter)
        - [Model Inspector · detector](#11a3-model-inspector--detector-por-qué-lo-marca)
        - [Training Lab · generador](#11a4-training-lab--generador)
        - [Training Lab · detector](#11a5-training-lab--detector)
        - [Checkpoints](#11a6-training-lab--checkpoints)
    - 11.B [Comparación con el proyecto de referencia: MLP con embeddings frente a Random Forest](#11b-comparación-con-el-proyecto-de-referencia-mlp-con-embeddings-frente-a-random-forest)
    - 11.C [Comparación de los modelos de deep learning](#11c-comparación-de-los-modelos-de-deep-learning)
12. [Pruebas automatizadas](#12-pruebas-automatizadas)
13. [Registro de decisiones (y por qué)](#13-registro-de-decisiones-y-por-qué)
14. [Problemas encontrados y cómo se resolvieron](#14-problemas-encontrados-y-cómo-se-resolvieron)
15. [Limitaciones honestas](#15-limitaciones-honestas)
16. [Cómo reproducir todo desde cero](#16-cómo-reproducir-todo-desde-cero)
17. [Estructura de archivos](#17-estructura-de-archivos)

---

### Cómo leer este documento

| Apartados | Pregunta que responden |
|---|---|
| 1–3 | Qué demuestra el laboratorio, qué significan sus términos y de dónde salió la idea. |
| 4–5 | Qué piezas tiene la aplicación y cómo se construyeron. El diario sigue el orden del desarrollo. |
| 6–8 | Qué ejemplos estudia cada modelo y cómo aprende. El detector aprende a clasificar nombres. El generador aprende a predecir el siguiente carácter. |
| 9–10 | Qué ocurre al pedir una ronda y cómo se comunican la pantalla y el servidor local. |
| 11–11.C | Cómo usar las tres pantallas, interpretar sus cifras y comparar los modelos con los del proyecto de referencia. Puedes empezar aquí con los modelos incluidos. |
| 12–17 | Cómo se comprobó el proyecto, qué decisiones y límites tiene, cómo arrancarlo y dónde encontrar cada pieza. |

## 1. Qué es PHANTOMLINK, explicado sin tecnicismos

Imagina dos programas que han aprendido cosas leyendo ejemplos, en lugar de que alguien les
escriba reglas a mano:

- **El Generador (🔴)** ha leído unos 140.000 nombres de dominio reales (la parte de una
  dirección web como `tienda-ejemplo` en `tienda-ejemplo.com`). Ha aprendido "cómo suenan" y
  ahora es capaz de **inventar nombres nuevos letra a letra**. Se le puede pedir que invente
  nombres "con estilo legítimo" o "con estilo de phishing" (phishing = webs fraudulentas que
  imitan a otras para robar datos).
- **El Detector (🔵)** ha aprendido, a partir de ejemplos etiquetados, a **estimar si un
  nombre se parece estructuralmente a los usados en phishing** (longitud, números, guiones,
  palabras como "login" o "secure", etc.).

En la **Arena** se enfrentan: el generador inventa, por ejemplo, 20 nombres; el detector los
juzga uno a uno y da una probabilidad. El usuario puede cambiar la "creatividad" del
generador (temperatura) o la "sensibilidad" del detector (umbral) y ver qué ocurre.

Puntos importantes de seguridad y honestidad:

- Todos los nombres generados se muestran con la terminación **`.test`**, que es un dominio
  reservado oficialmente (RFC 2606) que **nunca puede existir en Internet**. Además, se
  muestran como texto no clicable y la aplicación **nunca se conecta** a ninguna dirección.
- "No sospechoso" **no significa seguro**. El detector solo mira la forma del nombre, no la
  web. Es un prototipo educativo, no un antivirus.
- Ninguna cifra ni gráfico está inventado: todo sale de los modelos reales.

## 2. Glosario para no técnicos

| Término | Qué significa aquí |
|---|---|
| Modelo | Programa que ha aprendido a partir de ejemplos (en vez de reglas escritas a mano). |
| Entrenar | El proceso de mostrarle ejemplos al modelo y corregirle poco a poco. |
| Dataset | Colección de ejemplos. Aquí: 235.795 direcciones web etiquetadas como legítimas o phishing. |
| Etiqueta | La "respuesta correcta" de cada ejemplo (legítimo / phishing). |
| Train / Validation / Test | Tres montones separados de ejemplos: uno para aprender, otro para vigilar el aprendizaje y otro, guardado bajo llave, para el examen final. |
| Fuga de datos (*data leakage*) | Cuando el modelo "hace trampa" porque ve en el examen algo que ya había visto, o una pista que no tendría en la vida real. Da notas infladas y falsas. |
| Transformer | El tipo de red neuronal en el que se basan ChatGPT y similares. Aquí usamos uno diminuto (~107.000 parámetros frente a miles de millones). |
| Parámetro | Cada uno de los "números ajustables" que el modelo modifica al aprender. |
| Carácter / token | La unidad que el generador produce en cada paso. Aquí, una letra, un dígito, un guion o un punto. |
| Loss (pérdida) | Medida de cuánto se equivoca el modelo. Cuanto más baja, mejor. |
| Perplexity (perplejidad) | Otra forma de expresar la loss: "entre cuántas opciones duda el modelo de media". Un modelo que no sabe nada duda entre ~42 caracteres; el nuestro, entre ~16-18. |
| Temperatura | Control de creatividad: baja = nombres previsibles; alta = nombres más raros e incoherentes. |
| Top-K | Solo se permite elegir entre los K caracteres más probables en cada paso. |
| Semilla | Número que fija el azar: con la misma semilla se obtienen exactamente los mismos resultados. |
| Atención | Mecanismo del Transformer que decide en qué letras anteriores "fijarse" para elegir la siguiente. Podemos visualizarlo. |
| Random Forest | Un conjunto de muchos "árboles de decisión" (cuestionarios de sí/no aprendidos automáticamente) que votan. |
| Umbral | A partir de qué probabilidad el detector dice "sospechoso" (por defecto 0,5). |
| Accuracy / Precision / Recall / F1 | Métricas de acierto. *Precision*: de lo que marca como sospechoso, cuánto lo era. *Recall*: de lo sospechoso, cuánto detecta. *F1*: equilibrio entre ambas. |
| AUC | Capacidad de ordenar bien (sospechosos por encima de legítimos), independiente del umbral. 0,5 = azar, 1 = perfecto. |
| Matriz de confusión | Tabla con aciertos y errores de cada tipo (falsos positivos, falsos negativos…). |
| Checkpoint | Archivo con un modelo ya entrenado, para no tener que volver a entrenarlo. |
| API / backend | El "motor" que corre en el servidor y ejecuta los modelos. |
| Frontend | La parte visual que se ve en el navegador. |
| SSE (Server-Sent Events) | Técnica para que el servidor envíe datos en directo al navegador (las gráficas de entrenamiento en tiempo real). |

## 3. Documentos de partida y reglas del proyecto

- `PHANTOMLINK.md` (contexto): idea, inspiración (proyecto "Generador de Excusas"), propuesta
  de producto.
- `INSTRUCTIONS.md` (**la fuente de verdad**). Reglas clave que guían todas las decisiones:
  - Dos modelos entrenados desde cero: Transformer decoder-only a nivel de carácter (PyTorch)
    y `RandomForestClassifier` (scikit-learn).
  - Prohibido: modelos preentrenados, APIs externas, generación por plantillas, aprendizaje
    por refuerzo, reentrenamiento adversarial, ataques a sistemas reales, peticiones de red a
    los dominios generados, servicios de pago, copiar código de `Generador_Excusas`.
  - Datos: PhiUSIIL (UCI id 967), solo características extraíbles localmente del dominio,
    separación train/val/test **por dominio registrado**.
  - **No fabricar métricas, predicciones ni visualizaciones**.
  - Prioridades: P0 datos + 2 modelos + persistencia + integración → P1 Arena, controles,
    entrenamiento desde la UI, métricas reales → P2 atención/probabilidades → P3 estética.
  - Ante problemas de alcance: reducir complejidad, conservar la funcionalidad real.

Nota: no se ha consultado ni reutilizado código de `Generador_Excusas`; todo el código se ha
escrito desde cero para este proyecto.

## 4. Arquitectura general

```
            Navegador (React + Vite + Tailwind + Recharts)
       AI Arena  |  Model Inspector  |  Training Lab
                         │  
                         │  HTTP (JSON)
                         │  + SSE (telemetría en directo)
                         ▼
                 FastAPI  (backend/phantomlink/api.py)
          ┌──────────────┼──────────────────┐
          ▼              ▼                  ▼
   Generador (PyTorch)  Arena         Detector (scikit-learn)
   MiniGPT char-level   valida +      RandomForest + 19
   entrenamiento,       clasifica     características léxicas
   muestreo, trazas                   métricas, contribuciones
          │                                 │
   checkpoints/*.pt                 checkpoints/*.joblib

---------------------------------------------------------------------------
  data/processed/domains.csv  (preparado desde PhiUSIIL)
```

Una sola aplicación backend, un frontend y dos modelos. Sin bases de datos, microservicios ni
Docker obligatorio. En producción, el propio FastAPI sirve el frontend compilado, de modo que
todo arranca con un único comando.

## 5. Diario de desarrollo paso a paso

### Paso 0 — Lectura de requisitos y comprobación del entorno (19/11/2026)
- Equipo: Kali Linux, 12 núcleos, 3,8 GB de RAM, sin GPU (CUDA no disponible). Esto confirma
  que **todo debe funcionar en CPU** (requisito del proyecto) y condiciona tamaños de modelo.
- Faltaban PyTorch, scikit-learn y Node.js;
  - Solución Python: entorno virtual propio en `phantomlink/.venv` (no toca el sistema).
    PyTorch instalado en su versión **solo CPU** (mucho más ligera). Versiones resultantes:
    torch 2.15.0+cpu, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, FastAPI 0.165.0.
  - Solución Node.js: descarga del binario oficial v24.22.3 (LTS) en `~/.local/opt/`, sin
    instalación de sistema. Hay que añadirlo al `PATH` para usar `npm`.

### Paso 1 — Descarga y análisis del dataset
- Descargado `PhiUSIIL_Phishing_URL_Dataset.csv` (56 MB, 235.795 filas, 56 columnas) desde UCI.
- **Hallazgo 1 (crítico): el significado de las etiquetas está "al revés" de lo intuitivo.**
  En este CSV `label = 1` significa **legítimo** y `label = 0` significa **phishing**
  (134.850 legítimos, 100.945 phishing). Se comprobó mirando ejemplos (`www.uni-mainz.de`
  tiene label 1; `ethereum-uma.com` label 0). Si se hubiera asumido 1 = phishing, todo el
  detector funcionaría al revés. Solución: en un único lugar del código
  (`data.py`) se crea la columna `is_phishing = (label == 0)` y en todo el resto del proyecto
  se usa solo esa columna, sin ambigüedad.
- **Hallazgo 2 (crítico): fuga de etiqueta por el prefijo `www.`.** El **100 %** de los
  dominios legítimos empieza por `www.`, frente a solo el **41 %** de los de phishing. Un
  modelo aprendería "si no tiene www → phishing" y sacaría buena nota sin entender nada (y
  fallaría en la vida real). Solución: se elimina `www.` de todos los dominios antes de
  cualquier otra cosa.
- **Hallazgo 3: también `https` es 100 % en legítimos** frente a 49 % en phishing. No nos
  afecta porque solo usamos el nombre del dominio, nunca la URL completa.
- El dataset incluye muchas columnas que requieren haber visitado la web (título, formularios,
  iframes, número de imágenes…). **Se descartan todas**: nuestra demo no visita webs, así que
  el modelo no las tendría al hacer predicciones.

### Paso 2 — Preparación de datos (`backend/phantomlink/data.py`, `scripts/prepare_data.py`)
Ver [sección 6](#6-los-datos-de-dónde-salen-y-cómo-se-preparan) para el detalle completo.
Resultado: 212.851 nombres únicos en `data/processed/domains.csv` + informe
`data/processed/split_report.json`.

### Paso 3 — Detector Random Forest (`backend/phantomlink/features.py`, `classifier.py`)
Ver [sección 7](#7-modelo-b-el-detector-random-forest). Primer resultado en el conjunto de test
nunca visto: **accuracy 0,833, precision 0,878, recall 0,709, F1 0,785, AUC 0,862**.

### Paso 4 — Generador Mini Transformer (`backend/phantomlink/generator/`)
Ver [sección 8](#8-modelo-a-el-generador-mini-transformer). Primera prueba de 1.000
iteraciones (~32 s en CPU): la perplejidad de validación baja de 42,3 (no sabe nada) a 17,7.

### Paso 5 — Arena, gestor de entrenamientos y API
Ver secciones [9](#9-la-arena-cómo-interactúan-los-dos-modelos) y
[10](#10-el-backend-servidor-fastapi).

### Paso 6 — Pruebas automatizadas
31 pruebas, todas pasan. Detectaron dos fallos reales (ver
[sección 14](#14-problemas-encontrados-y-cómo-se-resolvieron)).

### Paso 7 — Checkpoints de ejemplo
- `checkpoints/classifier_default.joblib` (2,0 MB): 100 árboles, profundidad 12, 30.000
  dominios de entrenamiento, semilla 42 (~10 s).
- `checkpoints/generator_default.pt` (0,5 MB): 2 capas, 4 cabezas, 64 dimensiones, batch 64,
  6.000 iteraciones, learning rate 0,001, semilla 42. Tardó **490 s (~8 min) en CPU**.
  Perplejidad de validación final **14,89** (mejor loss de validación 2,685). Ver sección 8.

### Paso 8 — Experimento: ¿funciona de verdad el estilo?
Antes de construir la interfaz se midió si pedir «estilo phish» o «estilo legit» cambia algo
(500 nombres por estilo y temperatura, juzgados por el detector). Ver la tabla en la sección 8.
Conclusión: **sí, claramente** (1 % frente a 47 % marcados como sospechosos con T = 0,8).

### Paso 9 — Endpoint «Tú contra la IA»
Se añadió `GET /api/challenge`, que devuelve dominios del conjunto **test** con su etiqueta real,
equilibrados (mitad y mitad). Motivo: con los dominios generados no existe una «respuesta
correcta», así que solo con datos reales se puede puntuar de forma honesta a la persona y a la IA.

### Paso 10 — Frontend (React 19 + Vite 8 + Tailwind CSS 4 + Recharts 3)
Tres pantallas (ver sección 11). Se compila a `frontend/dist`, que sirve el propio FastAPI.

### Paso 11 — Pruebas en un navegador real
Con Playwright (herramienta de desarrollo, no forma parte de la aplicación) y el Chromium del
sistema, se automatizó: abrir la app → iniciar un duelo → jugar a «Tú contra la IA» → abrir un
dominio en el Inspector → ir al Training Lab. Se hicieron capturas a 1400 px (escritorio) y a
390 px (móvil). Sin errores en la consola. También se lanzaron entrenamientos reales **desde la
interfaz** (preset Fast: 16.160 parámetros, 400 iteraciones, ~3 s; preset Break it) y se comprobó
que las gráficas se actualizan en directo y que el modelo se guarda y se activa al terminar.
Después se reactivó el generador por defecto.

### Paso 12 — Validación de los presets con datos reales
Cada texto de «qué esperamos que ocurra» se comprobó entrenando el preset y, cuando la realidad
no coincidía, se corrigió el preset o el texto (ver sección 13, D16–D17, y sección 14).

### Paso 13 — Ajustes de diseño responsive
En móvil, la tabla de la Arena (4 columnas) desbordaba la pantalla: la página medía 442 px en una
pantalla de 390. Ahora cada dominio es una tarjeta de dos líneas en pantallas estrechas y las
rejillas de estadísticas pasan de 3–5 a 2–3 columnas. Verificado: 390 px exactos, sin
desplazamiento horizontal.

### Paso 14 — Vídeo de demostración automático
`scripts/record_demo.py` (herramienta de desarrollo) abre la aplicación real con Playwright, sigue
el guion de `PHANTOMLINK.md` y graba `docs/demo.webm` (**42 s, 1280×720**; borrado después y sustituido por
los vídeos del paso 16):
0–4 s, presentación de las dos IAs · 4–9 s, primer duelo y decisiones · 9–17 s, se sube la
temperatura a 1,4 y se repite (en la grabación, los «estilo phish» marcados pasaron del 40 % al
80 %, visible en el historial) · 17–29 s, Inspector generando carácter a carácter y mapas de
atención · 29–42 s, entrenamiento real del preset Fast con las gráficas actualizándose.
El script solo hace clic y desplaza la página: todo lo que aparece lo calculan los modelos.
Para grabar hizo falta el ffmpeg propio de Playwright (`python -m playwright install ffmpeg`).
Ese ffmpeg no sirve para convertir a MP4: si hay un ffmpeg normal
instalado, el script genera también `docs/demo.mp4`. Si no, se puede convertir con
`ffmpeg -i docs/demo.webm -c:v libx264 -pix_fmt yuv420p docs/demo.mp4`.
El vídeo es una base.

### Paso 15 — README y script de arranque
`README.md` (sí se sube al repositorio) y `run.sh`, que prepara los datos si faltan, compila el
frontend si falta y arranca el servidor en http://127.0.0.1:8000.

---

### Paso 16 — Vídeos
`scripts/record_social.py` graba dos versiones con rótulos, foco, zoom y un cursor visible, todo
inyectado en la página real. Resultado: `docs/demo_16x9.mp4` (1920×1080, 43 s) y
`docs/demo_9x16.mp4` (1080×1920, 45 s, con el diseño móvil real), en H.264 con barra de progreso.
- Las cifras de los rótulos se leen de las respuestas de la API durante la propia grabación. En
  ambas versiones, los «estilo phish» marcados pasan del 40 % al 80 % al subir la temperatura de
  0,8 a 1,4.
- **Problema 1**: la grabadora de vídeo de Playwright ignora la escala de alta densidad y pinta la
  página pequeña en una esquina de un lienzo gris («se ve como en una ventana»). **Solución**:
  capturar los fotogramas con el *screencast* de Chrome DevTools a resolución real (~36 fps) y
  montarlos con ffmpeg respetando el instante de cada fotograma.
- **Problema 2**: el vídeo 16:9 entrena un modelo Fast en directo, y el 9:16 se grabó con ese modelo
  activo (mostraba 16.160 parámetros). **Solución**: restaurar el generador original después de
  **cada** grabación.
- ffmpeg completo (binario estático) instalado en `~/.local/opt/ffmpeg/` (sin sudo). No incluye
  `drawtext`, por eso los textos se hacen en la página.

### Paso 17 — Música original para los dos vídeos
**Decisión: componer la música por código en lugar de descargarla.** El requisito era «sin
copyright, gratis». Con una pista descargada, la licencia solo se puede comprobar leyendo una
página web, y las licencias «libres» varían: CC BY obliga a poner el crédito, algunas webs
restringen ciertos usos y las condiciones pueden cambiar. Una pista sintetizada desde cero no
depende de ninguna obra ajena, así que no hay derechos de terceros que comprobar.

`scripts/compose_music.py` (numpy + scipy) genera la pista y la añade a cada vídeo:
- **Estilo**: electrónica ambient, sin voces, en La menor a **100 BPM**, con la progresión
  Am–F–C–G (dos compases por acorde).
- **Capas**: pad cálido (sierras con armónicos limitados y tres voces ligeramente desafinadas por
  nota), sub-bajo con efecto de «bombeo» en cada pulso, arpegio de notas punteadas en corcheas,
  bombo suave y hi-hats solo en el clímax, y una reverb por convolución con ruido decreciente.
- **Estructura sincronizada con el guion** (escalada a la duración de cada vídeo):

  | Tramo del vídeo | Qué hace la música | Nivel medido (RMS) |
  |---|---|--:|
  | Título (0–3 s) | Solo el pad, entrando poco a poco | −19 dB |
  | Las dos IAs y primer duelo (3–13 s) | Entran el bajo, el arpegio y el bombo; el pad se va abriendo | −15 → −9 dB |
  | Experimento de temperatura (13–22 s) | **Clímax**: hi-hats, pad más brillante, arpegio más fuerte | −7,5 dB |
  | Inspector y entrenamiento (22–38 s) | Meseta más relajada | −10,5 dB |
  | Tarjeta de cierre (38 s–final) | Fuera la percusión; resuelve en La menor y se desvanece | −13 → −16 dB |

- **Volumen**: música de fondo a **−21 LUFS** con un pico de −13,7 dBFS (sin saturación). Se
  aplica una **ganancia fija**. La primera versión usaba un normalizador dinámico (`loudnorm`),
  que aplanaba la curva hasta dejar un rango de solo 2 LU; con ganancia fija el rango es de 6,6 LU y
  la subida del clímax se conserva.
- **Verificación**: medición EBU R128, espectrograma (se ven el bajo desde ~3 s, el bombo desde ~6 s
  y los hi-hats entre 13 y 24 s) y comprobación de que la imagen de los vídeos **no se recodificó**
  (mismo MD5 del flujo de vídeo).
- **Limitación**: la pista no se ha podido **escuchar** durante el desarrollo; solo se comprobó con
  mediciones. El juicio estético es humano. Casi toda la energía está por debajo de 700 Hz: en
  altavoces de móvil, que apenas reproducen graves, se oirán sobre todo el pad y el arpegio.
- LinkedIn reproduce los vídeos en silencio por defecto, así que la música es un extra: el vídeo
  se entiende sin sonido gracias a los rótulos.

## 6. Los datos: de dónde salen y cómo se preparan

**Fuente:** PhiUSIIL Phishing URL Dataset, UCI Machine Learning Repository, id 967, licencia
CC BY 4.0 (uso y redistribución libres citando la fuente).

**Pipeline de normalización** (`data.normalize_host`), en este orden:

1. Minúsculas, quitar espacios, punto final y puerto (`:8080`).
2. Descartar direcciones IP (no tienen nombre que analizar): 608 filas, junto a las no
   interpretables.
3. Separar el dominio con `tldextract` usando la **Public Suffix List incluida en el paquete**
   (`suffix_list_urls=()`), es decir, **sin conexión a Internet**. Así se sabe que en
   `cavernas.org.br` el sufijo es `org.br` y el nombre registrado es `cavernas`.
4. Quitar todos los `www.` iniciales (también casos dobles como `www.www.ejemplo.com`).
5. **Quitar el sufijo público (TLD)**. Motivo: el generador produce nombres bajo `.test`. Para
   que el detector juzgue lo mismo en entrenamiento y en la Arena, ambos trabajan **solo con el
   nombre** (subdominios + dominio). Coste: el detector pierde la pista del TLD (p. ej. `.xyz`
   o `.top` son más frecuentes en phishing), lo que baja algo sus métricas. Se acepta a cambio
   de coherencia y honestidad.
6. Solo se aceptan caracteres `a-z 0-9 - .` (alfabeto de 38 símbolos compartido por ambos
   modelos).

Ejemplo: `www.kuerennkayccato-co-jp.kuerennkayccato.bsutgy.top` → nombre
`kuerennkayccato-co-jp.kuerennkayccato.bsutgy`, dominio registrado `bsutgy.top`.

**Limpieza:**
- Nombres que aparecen con ambas etiquetas (ambiguos): 457 nombres eliminados por completo.
- Duplicados (mismo nombre normalizado): 20.736 eliminados, se conserva uno.

**División train/validation/test por dominio registrado** (requisito anti-fuga):
- Cada dominio registrado (eTLD+1, sección ICANN de la PSL) se asigna a un conjunto mediante un
  hash SHA-256 estable: 70 % train, 15 % validación, 15 % test. Ventajas: determinista (no
  depende del orden de las filas ni de una semilla), y **todos los subdominios de un mismo
  dominio registrado caen en el mismo conjunto**. El script comprueba con un `assert` que no
  hay ningún dominio registrado en dos conjuntos.
- Decisión deliberadamente estricta: plataformas de alojamiento gratuito muy usadas por phishing
  (`web.app` 5.527 filas, `firebaseapp.com` 5.418, `repl.co` 3.498, `weeblysite.com` 2.889…)
  cuentan como un único grupo cada una. Así el modelo **no puede memorizar** una plataforma en
  train y "reconocerla" en test. Consecuencia: los porcentajes de phishing varían entre
  conjuntos (train 34 %, val 50 %, test 43 %) y la evaluación es más dura (más realista).

| Conjunto | Filas | Phishing | Legítimos | Dominios registrados |
|---|--:|--:|--:|--:|
| train | 139.897 | 47.902 | 91.995 | 118.860 |
| val | 38.863 | 19.276 | 19.587 | 25.511 |
| test | 34.091 | 14.589 | 19.502 | 25.162 |
| **Total** | **212.851** | | | |

Comando: `python scripts/prepare_data.py --download` (descarga, ~3 s de procesado).

## 7. Modelo B: el detector (Random Forest)

**Qué es, en sencillo:** cientos de "cuestionarios" automáticos (¿tiene más de 3 dígitos? ¿más
de 2 puntos?...). Cada árbol responde una probabilidad y se hace la media.

**Características** (`features.py`) — 19, todas calculadas solo a partir del texto del nombre:

| Característica | Significado |
|---|---|
| length | Número de caracteres |
| n_labels | Número de partes separadas por puntos |
| max_label_len / mean_label_len | Longitud de la parte más larga / media |
| n_digits / digit_ratio | Cantidad / proporción de dígitos |
| n_hyphens / hyphen_ratio | Cantidad / proporción de guiones |
| entropy | Entropía de Shannon: lo "aleatorio" que parece el texto |
| distinct_ratio | Caracteres distintos / longitud |
| vowel_ratio | Proporción de vocales entre las letras |
| max_consonant_run / max_digit_run / max_repeat_run | Rachas más largas de consonantes / dígitos / mismo carácter |
| repeated_bigram_ratio | Proporción de pares de letras repetidos |
| letter_digit_switches | Cambios letra↔número (p. ej. `a1b2`) |
| suspicious_tokens | Nº de patrones léxicos presentes (login, secure, verify, pay, wallet, web, app…) |
| has_punycode | Contiene `xn--` (dominios internacionalizados, usados en homógrafos) |
| starts_with_digit | Empieza por número |

Los patrones léxicos **no son reglas**: solo se cuentan; el bosque decide cuánto importan.

**Entrenamiento** (`classifier.train`):
- Muestra estratificada de `train_size` dominios del conjunto train (por defecto 30.000,
  configurable hasta ~140.000).
- **Crecimiento incremental** (`warm_start`): se añaden árboles en ~20 tandas y tras cada una
  se mide en validación. Así la interfaz puede mostrar en directo cómo evolucionan las métricas
  al añadir árboles. Es entrenamiento real, no una animación.
- Pesos de clase `balanced` calculados **una sola vez** a partir de la muestra (ver problema 3
  en la sección 14).
- Al final: métricas en validación y en **test** (nunca usado antes), matriz de confusión,
  histograma de probabilidades por clase, barrido de umbrales (0,05–0,95), importancia de
  características por impureza y por **permutación** (más fiable: mide cuánto empeora el F1 si
  se desordena esa característica), y tiempo de inferencia por cada 1.000 dominios.

**Explicación por dominio** (`classifier.contributions`): para un nombre concreto se recorre el
camino de decisión en cada árbol y se atribuye a cada característica el cambio de probabilidad
que produce (método de Saabas). La suma `probabilidad base + contribuciones` coincide
**exactamente** con la predicción del modelo (lo verifica una prueba automatizada). También se
devuelven los votos individuales de cada árbol (dispersión = desacuerdo del bosque) y la mediana
de cada característica por clase en train, como contexto.

**Persistencia:** `ClassifierBundle` (modelo + configuración + nombres de características +
versión del extractor + métricas + historial + semántica de etiquetas + versión de
scikit-learn) guardado con `joblib`. Al cargar se comprueba que la versión del extractor de
características es compatible.

**Resultados del checkpoint por defecto** (100 árboles, profundidad 12, hoja mínima 2,
balanced, umbral 0,5, 30.000 dominios, semilla 42, ~10 s de entrenamiento):

| Conjunto | Accuracy | Precision | Recall | F1 | AUC |
|---|--:|--:|--:|--:|--:|
| Validación | 0,860 | 0,928 | 0,778 | 0,846 | 0,899 |
| **Test** | **0,833** | **0,878** | **0,709** | **0,785** | **0,862** |

Matriz de confusión en test (umbral 0,5): 18.070 legítimos bien clasificados, 1.432 falsos
positivos, 4.245 phishing no detectados, 10.345 phishing detectados.

Importancia (impureza) destacada: `n_labels` 0,29, `entropy` 0,10, `n_digits` 0,09,
`max_digit_run` 0,08. Por permutación, `n_labels` domina (0,10) seguida de
`suspicious_tokens` (0,014). Interpretación: el número de subdominios es la señal más fuerte
(típico de phishing alojado en plataformas, `algo.web`, `x.y.z`).

Lectura honesta: ~29 % del phishing de test **no** se detecta solo con el nombre. Es lo
esperable de un análisis puramente léxico, y es un mensaje del proyecto.

## 8. Modelo A: el generador (Mini Transformer)

**Qué es, en sencillo:** un "autocompletado" diminuto. Dado el principio de un nombre, calcula la
probabilidad de cada posible siguiente carácter, elige uno y repite hasta decidir terminar.

**Tokenizador** (`generator/tokenizer.py`): 42 símbolos = 4 especiales (`<pad>` relleno,
`<legit>` y `<phish>` para indicar el estilo, `<eos>` fin) + 38 caracteres. Cada ejemplo de
entrenamiento es: `<estilo> c1 c2 … cn <eos>`. El token de estilo es **la única** forma de
condicionar la generación (no hay plantillas).

**Arquitectura** (`generator/model.py`), escrita con operaciones básicas de PyTorch (no se usa
`nn.Transformer` ni pesos preentrenados), para poder exponer las matrices de atención reales:
- Embeddings de token + embeddings de posición aprendidos.
- N bloques: LayerNorm → atención causal multicabeza (implementación propia: Q, K, V, máscara
  triangular para no "ver el futuro", softmax) → conexión residual → LayerNorm → MLP (4×, GELU)
  → conexión residual.
- LayerNorm final y capa de salida que comparte pesos con el embedding (*weight tying*).
- Configuración por defecto: 2 capas, 4 cabezas, 64 dimensiones, contexto 64, dropout 0,1 →
  **106.880 parámetros** (dentro del objetivo de 100k–200k).

**Corpus:** nombres del conjunto train (139.127 tras descartar 770 de más de 63 caracteres) para
aprender; nombres de validación (38.039) para medir. Nunca se usa test.

**Entrenamiento** (`generator/train.py`):
- AdamW (weight decay 0,01 solo en matrices), warmup lineal (5 %) + decaimiento coseno hasta el
  10 % del learning rate, *gradient clipping* configurable (0 = desactivado).
- Telemetría real cada `log_interval` iteraciones: loss de train, learning rate, **norma del
  gradiente** (medida antes del recorte), tokens/s, ms por iteración, memoria (RSS del proceso;
  y memoria CUDA si hay GPU). Cada `eval_interval`: loss en train y validación promediadas
  sobre 20 batches, **perplejidad**, y 3 dominios de muestra por estilo (para ver cómo aprende).
- Detección de divergencia: si la loss deja de ser un número finito, se detiene y **no** se
  guarda el modelo roto.
- Parada a petición del usuario en cualquier iteración.
- Reproducible: semillas en PyTorch, NumPy y en generadores de números aleatorios dedicados.

**Muestreo** (`generator/sampling.py`):
- Temperatura (0 = elegir siempre el más probable), top-K (0 = sin límite), longitud máxima,
  semilla. Generación en lote (varios nombres a la vez).
- Se prohíbe emitir tokens especiales y terminar en el primer paso (evita nombres vacíos).

**Inspección paso a paso** (`trace`): genera un nombre (opcionalmente forzando un prefijo, o
puntuando un nombre completo escrito por el usuario) y devuelve, para cada carácter: el
contexto, el carácter elegido, su probabilidad según el modelo y según la distribución
realmente muestreada (tras temperatura y top-K), la entropía (incertidumbre) en bits y los 8
candidatos más probables. Además las **matrices de atención reales** de cada capa y cabeza y las
**activaciones** del flujo residual tras cada bloque. Técnica: tras generar, una sola pasada
causal sobre la secuencia completa da exactamente las mismas distribuciones que se tenían en
cada paso (gracias a la máscara causal; verificado por una prueba).

**Persistencia:** `.pt` con formato `phantomlink-generator/1`, configuración, pesos, tokenizador,
historial y métricas. Se carga con `torch.load(weights_only=True)` (seguro: no ejecuta código).

**Primera prueba (1.000 iteraciones, batch 32, ~32 s):**

| Iteración | Loss train | Loss val | Perplejidad val | Ejemplos estilo phish |
|--:|--:|--:|--:|---|
| 0 | 3,74 | 3,75 | 42,3 | `3f7zonjew3wwpq3xoc…` (basura aleatoria) |
| 250 | 2,76 | 3,01 | 20,2 | `oussdintera.dh-bolakeeb` |
| 500 | 2,79 | 2,91 | 18,4 | `malllesom.webdy` |
| 999 | 2,91 | 2,90 | 18,2 | `soupebo-gtsu` |

Ya aparecen patrones aprendidos de verdad: el estilo phish produce sufijos tipo `.web`, `.wapp`,
más puntos y guiones (imitando el phishing alojado en `web.app`), mientras el estilo legítimo
produce palabras compuestas pronunciables (`nelandumaring`, `cordictcs`).

Observación: con la misma semilla, ambos estilos a veces producen el mismo primer nombre (el
condicionamiento aún es débil con poco entrenamiento). Por eso la Arena usa semillas distintas
para cada estilo.

**Checkpoint por defecto (6.000 iteraciones, batch 64):**

| Iteración | Loss train | Loss val | Perplejidad val | Ejemplos estilo phish |
|--:|--:|--:|--:|---|
| 0 | 3,75 | 3,75 | 42,5 | basura aleatoria |
| 1.000 | 2,76 | 2,85 | 17,4 | `menvandiciresty` |
| 2.000 | 2,77 | 2,79 | 16,2 | `asteollres-104798.web`, `sarmerivere-co-jp.web` |
| 4.000 | 2,66 | 2,73 | 15,4 | `f0646498.xsph` |
| 5.999 | 2,69 | 2,74 | 15,5 | `firesoftappeace`, `mispy-om` |
| final (20 batches) | | 2,70 | **14,9** | |

Se aprecian patrones aprendidos (no programados): `-co-jp` (suplantación de dominios japoneses),
`.web` (phishing alojado en `web.app`), `f0646498.xsph` (subdominios numéricos de `xsph.ru`).

**¿Funciona el condicionamiento por estilo? (experimento, 500 nombres por fila, top-K 20)**

| Temperatura | Estilo | Válidos | Marcados por el detector | P media | Subdominios medios | Dígitos medios | Longitud media | Ya en el dataset |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 0,5 | legit | 100 % | 1 % | 0,31 | 1,00 | 0,00 | 10,5 | 5 |
| 0,5 | phish | 100 % | 49 % | 0,63 | 1,79 | 0,58 | 17,7 | 37 |
| 0,8 | legit | 100 % | 1 % | 0,30 | 1,00 | 0,00 | 11,4 | 10 |
| 0,8 | phish | 100 % | 47 % | 0,59 | 1,57 | 0,49 | 16,4 | 14 |
| 1,2 | legit | 100 % | 5 % | 0,32 | 1,02 | 0,00 | 11,9 | 6 |
| 1,2 | phish | 100 % | 49 % | 0,60 | 1,53 | 0,44 | 16,5 | 1 |

Lectura: el estilo cambia de verdad la estructura (más subdominios, dígitos y longitud en
«phish»). A más temperatura hay menos memorización (de 37 nombres ya existentes a 1). Incluso con
T = 1,2 todos los nombres son sintácticamente válidos.

**Presets del generador (medidos):**
- *Fast*: 1 capa, 2 cabezas, 32 dims → 16.160 parámetros, 400 iteraciones en ~3 s, val loss ≈3,0
  (perplejidad ≈20).
- *Balanced*: la configuración de referencia, con 2.000 iteraciones.
- *Break it*: la primera versión usaba learning rate 0,3. **Resultado real**: no divergía, solo se
  estancaba (loss ≈3,3 y un pico inicial de val loss de 7,3). Se probó 1,0: la loss **explota de
  3,7 a 170**, la norma del gradiente llega a ~69 y luego se estanca en ≈3,3 sin recuperarse
  (un modelo sano llega a ≈2,7). Se eligió 1,0 por ser más ilustrativo y se reescribió el texto
  con lo observado. No llega a NaN porque Adam normaliza el tamaño de cada paso.

## 9. La Arena: cómo interactúan los dos modelos

`backend/phantomlink/arena.py`, siguiendo los 5 pasos de `INSTRUCTIONS.md`:

1. **Generar** N nombres (1–200). Mezcla configurable: mitad phish / mitad legit, o solo uno.
2. **Validar sintácticamente** (reglas RFC 1035/1123): etiquetas de 1–63 caracteres, sin guion al
   principio o final, sin `..`, solo caracteres permitidos. Se indica el motivo del rechazo.
3. **Clasificar** los válidos con el detector.
4. **Decisiones individuales**: probabilidad, sospechoso/no sospechoso según el umbral, si el
   nombre ya existe en el dataset (memorización: se indica en qué conjunto), duplicados en la
   ronda, log-probabilidad media del generador.
5. **Resumen agregado**: generados, válidos, únicos, ya existentes, sospechosos, por estilo
   (cuántos de los "estilo phish" marca el detector), tiempos de generación y clasificación.

Importante: que un nombre "estilo phish" sea marcado como sospechoso es **acuerdo entre dos
modelos**, no un acierto verificado. Por eso no se llama "accuracy" a ese número.

## 10. El backend (servidor FastAPI)

`backend/phantomlink/api.py`. Endpoints:

| Método y ruta | Función |
|---|---|
| GET `/api/status` | Modelos activos, métricas, dispositivo, trabajo en curso, informe del dataset |
| GET `/api/presets` | Presets Fast / Balanced / Break it y valores por defecto |
| GET `/api/models` · POST `/api/models/activate` | Listar y activar checkpoints (de ejemplo o generados desde la UI) |
| POST `/api/train/generator` · `/api/train/classifier` | Iniciar entrenamiento (respuesta 202 con id) |
| POST `/api/train/stop` | Detener el entrenamiento en curso |
| GET `/api/jobs/current` · `/api/jobs/{id}` | Estado de un trabajo |
| GET `/api/jobs/{id}/events` | **SSE**: telemetría en directo (repite desde `since`) |
| POST `/api/generate` | Generar nombres |
| POST `/api/classify` | Clasificar nombres escritos por el usuario |
| POST `/api/arena` | Ronda completa de duelo |
| POST `/api/inspect/generator` | Traza paso a paso con probabilidades, atención y activaciones |
| POST `/api/inspect/classifier` | Características, contribuciones y votos de los árboles |
| GET `/api/classifier/report` · `/api/generator/report` | Métricas e historiales completos (con recálculo por umbral) |

Protecciones (requisito "límites de recursos, validar configuraciones, evitar entrenamientos
concurrentes"):
- Validación con Pydantic: capas 1–6, cabezas 1–8, dimensiones 16–256 y divisibles por cabezas,
  contexto 16–64, iteraciones 10–20.000, batch 4–256, learning rate (0, 1], tope de ~2 millones
  de parámetros, árboles 1–500, profundidad 1–40, rondas de Arena ≤ 200, etc.
- **Un solo entrenamiento a la vez** (si no, error 409). No se puede cambiar de modelo mientras
  se entrena.
- Hilos de PyTorch limitados a 4 (configurable con `PHANTOMLINK_THREADS`) para que la interfaz
  siga respondiendo durante un entrenamiento.
- Los identificadores de checkpoints se validan para impedir acceder a rutas arbitrarias.
- Los entrenamientos lanzados desde la UI se guardan en `runs/` y se activan automáticamente al
  terminar (o al detenerlos); si divergen, no se guardan.
- La entrada del usuario (p. ej. `https://www.Ejemplo.com/ruta`) se normaliza igual que el
  dataset, solo con procesado de texto: **nunca se resuelve ni se contacta**.

## 11. El frontend (interfaz)

Puedes empezar en **AI Arena** y pulsar **Iniciar duelo** con los modelos incluidos. Haz clic en un nombre de la ronda para pasar al **Model Inspector** y ver cómo lo puntúan ambos modelos. **Training Lab** permite entrenar otras versiones y compararlas con las de ejemplo.

| Acción | Qué estás cambiando |
|---|---|
| Elegir temperatura, top-K o semilla | Cómo se usa el generador ya entrenado. Pulsa **Iniciar duelo** o **Generar** para obtener una salida con los valores nuevos. |
| Mover el umbral de la Arena | La decisión aplicada a las puntuaciones de la ronda visible. Por ejemplo, una puntuación de 0,62 se marca con umbral 0,50 y queda sin marcar con umbral 0,70. |
| Cambiar un campo de Training Lab | La configuración de la próxima ejecución. **Entrenar** inicia el aprendizaje con esos valores. |
| Pulsar **activar** en Checkpoints | La versión guardada que se usará en las siguientes consultas de la aplicación. |

En **machine learning (ML)**, un modelo aprende patrones a partir de ejemplos. El Random Forest de PhantomLink aprende preguntas sobre 19 medidas del nombre, como su longitud. En **deep learning (DL)**, una rama de ML, se encadenan capas de una red neuronal que aprenden sus propias representaciones. El Transformer del proyecto convierte caracteres en números, combina información de posiciones anteriores y predice qué podría venir después. En ambos casos, **entrenar** es el proceso de aprender y **hacer inferencia** es utilizar lo aprendido para resolver un caso nuevo.

La interfaz está escrita en React y dibuja las gráficas con Recharts. El navegador envía peticiones al servidor local y recibe el progreso del entrenamiento mediante SSE, una conexión que entrega mensajes sucesivos. Los cálculos de los modelos se ejecutan en el servidor.

## 11.A Manual de interacciones: cada control explicado

En la cabecera, el rojo identifica al generador y el azul al detector. Lee la leyenda de cada gráfica: en unas distingue los estilos solicitados y en otras indica el sentido de una contribución o identifica una serie de datos. Las cifras históricas del manual corresponden a los modelos y condiciones de las pruebas que las produjeron.

### 11.A.0 Elementos comunes a las tres pantallas

**Pestañas (AI Arena · Model Inspector · Training Lab).** Cambian de pantalla sin perder lo que
había en las demás: las tres siguen montadas por detrás. La app recuerda la última pestaña
abierta.

**Insignias de la cabecera** (se actualizan cada 4 s):

| Insignia | Qué dice |
|---|---|
| `GEN 106.880 params` | Tamaño del generador **activo**. Si entrenas uno nuevo, este número cambia (p. ej. a 16.160 con el preset Fast). Es la forma más rápida de saber qué modelo estás usando. |
| `DET 100 árboles` | Tamaño del detector activo. |
| `CPU` / `CUDA` | Indica si el servidor detecta CUDA, la plataforma de cálculo para GPU NVIDIA. El dispositivo de una ejecución concreta se indica en el registro de entrenamiento del generador. |
| `● entrenando generator/classifier` | Aparece mientras hay un entrenamiento en curso. Solo puede haber uno a la vez. |
| `backend desconectado` | El servidor no responde. Hay que arrancarlo con `./run.sh`. |

**Pie de página.** Es un recordatorio permanente: el sufijo reservado `.test` permite mostrar los nombres como texto. Incluso cuando «Tú contra la IA» toma nombres de datos reales, la aplicación trabaja con sus caracteres y nunca visita los sitios.

---

### 11.A.1 AI Arena

Antes de pulsar **Iniciar duelo**, decide qué quieres observar. Para estudiar la generación, cambia un ajuste y ejecuta otra ronda con la semilla fija. Para estudiar el detector, mueve su umbral sobre la ronda que ya está a la vista. La Arena compara el estilo solicitado con el juicio del detector. Las etiquetas verificadas del conjunto de datos se usan en «Tú contra la IA» y en las métricas de test.

#### Tarjeta roja — IA generadora

**MiniGPT · capas · cabezas · parámetros.** Identifica la arquitectura activa. Las capas procesan la secuencia una tras otra. Cada capa reparte su trabajo de atención entre varias cabezas en paralelo. Los parámetros son los números ajustados durante el aprendizaje. **ppl val** resume cuánto le sorprenden, en promedio, los caracteres reales de los nombres de validación. Una perplejidad menor indica mejor predicción del siguiente carácter sobre ejemplos comparables.

**Temperatura (creatividad)** · rango 0–2 · insignia «tiempo real»
- *Concepto*: antes de elegir cada letra, el modelo calcula una probabilidad para cada carácter
  posible. La temperatura «aplana» o «afila» esa lista antes de sortear. Con T = 1 se usa la
  distribución tal cual; con T < 1 se refuerzan los favoritos; con T > 1 se reparten las opciones.
  Con T = 0 no hay sorteo: siempre gana el favorito.
- *Efecto*: se aplica al próximo duelo. La etiqueta «tiempo real» indica que puedes reutilizar el modelo entrenado. Con T = 1, las probabilidades originales se conservan antes de aplicar el filtro top-K.
- *Ejemplo real* (letra siguiente a `secure-`, estilo phish):

  | Temperatura | P(«s») | P(«c») | P(«a») | Incertidumbre (entropía) |
  |--:|--:|--:|--:|--:|
  | 0,5 | 30,7 % | 15,4 % | 10,7 % | 3,50 bits |
  | 1,0 | 13,1 % | 9,3 % | 7,8 % | 4,56 bits |
  | 1,5 | 8,6 % | 6,8 % | 6,1 % | 4,90 bits |

  Y en nombres completos (estilo phish, semilla 1):
  - **T = 0**: los 8 nombres salen idénticos (`saison-co-jp.amazon.com.jp.azzzzzzzzz`). Sin azar,
    el modelo sigue las elecciones que considera más probables y acaba repitiendo `z` hasta el límite de longitud.
  - **T = 0,5**: `saisancard.co.jp.jp.amazij`, `berlonger`, `paildinshaakckjnw546.web`.
  - **T = 1,0**: `berloogin`, `phissoladerm.web`, `oflootd.ncs.byecolling`.
  - **T = 2,0** sin top-K: `pviqbuea.wew.rhpdoglxx024xbxfmbonno96b21oem8uzm` (casi ruido).

  La temperatura puede afectar tanto a la variedad como a la cantidad de nombres válidos. Para observarlo, compara «únicos» y «válidos» entre dos duelos con el mismo estilo, tamaño y semilla.

**Top-K** · rango 0–42 (0 = sin límite) · «tiempo real»
- *Concepto*: en cada paso solo se permite sortear entre los K caracteres más probables. El resto
  queda prohibido.
- *Efecto*: corta las opciones raras. Un K bajo limita la variedad de cada paso. K = 1 reduce el sorteo al favorito salvo empates y K = 0 elimina el filtro. A T = 0 se elige el favorito directamente.
- *Ejemplo real* (tras `secure-`, T = 1): con K = 3 solo pueden salir «s» (43,5 %), «c» (30,8 %) y
  «a» (25,7 %). «p» y «b», que tenían un 5,9 % y un 5,8 %, pasan a 0 %. La incertidumbre cae de
  4,56 a 1,55 bits.

**Longitud máxima** · 4–63
- *Concepto*: el máximo de caracteres por nombre. Si el modelo no ha decidido terminar antes, se le
  obliga a cerrar.
- *Efecto*: nombres más cortos. No cambia lo que el modelo «piensa», solo dónde se le corta.
- *Aviso*: si el generador activo tiene un contexto menor (p. ej. 16, entrenado en el Lab), la
  longitud máxima no puede superar contexto − 1. Si la supera, la Arena muestra el error «max_len
  no puede superar 15…»; basta con bajar el deslizador.

**Dominios por ronda** · 4–100
- *Efecto*: cuántos nombres se inventan en cada duelo. Con más nombres, cada caso individual pesa menos en el porcentaje. Si solo hay cuatro nombres válidos de un estilo, una decisión cambia su barra en 25 puntos porcentuales.

**Estilo solicitado** (mitad y mitad · phish · legit)
- *Concepto*: el primer «token» que recibe el generador es una etiqueta de estilo. El modelo
  aprendió durante el entrenamiento cómo son los nombres de cada tipo.
- *Efecto*: «mitad y mitad» pide la mitad de cada estilo (si el número es impar, phish recibe uno
  más). Medido con 500 nombres por estilo a T = 0,8: el detector marcó el 1 % de los «legit» y el
  47 % de los «phish».
- *Importante*: el estilo es **lo que se pidió**, no una verdad comprobada.

**Semilla** y casilla **«incrementar la semilla en cada ronda»**
- *Concepto*: el número que fija el azar. Misma semilla + mismos ajustes + mismo modelo = exactamente
  los mismos nombres.
- *Efecto*: con la casilla marcada (por defecto), cada nueva ronda suma 1 a la semilla y salen
  nombres distintos. Desmárcala para repetir una ronda idéntica y comparar solo el efecto de otro
  ajuste (por ejemplo, el umbral).
- *Ejemplo*: con la semilla 42 la primera ronda siempre empieza por `mintanotel.test` y
  `oodect-meds-fb-co-jp.amazocn.ccncn.woork.test`.

#### Botón central — ⚡ INICIAR DUELO — N DOMINIOS
Ejecuta el ciclo completo en el servidor: generar → validar → clasificar → resumir. Se desactiva
mientras trabaja o si falta alguno de los dos modelos.

#### Tarjeta azul — IA detectora

**Random Forest · árboles · prof. · 19 características.** Cada árbol recorre un camino de preguntas sobre medidas del nombre. La profundidad máxima limita cuántos niveles de preguntas puede tener. La puntuación del bosque es la media de las puntuaciones de los árboles. **F1 test** resume el equilibrio entre cuántas alarmas son correctas y cuánto phishing se detecta. Es la media armónica de precision y recall, por lo que queda penalizado si cualquiera de las dos cae.

**Umbral de decisión (sensibilidad)** · 0,05–0,95 · «tiempo real»
- *Concepto*: el detector da una probabilidad entre 0 y 1. El umbral es la línea a partir de la
  cual esa probabilidad se convierte en «SOSPECHOSO».
- *Efecto en la Arena*: al moverlo, las decisiones se recalculan al instante sobre las
  probabilidades ya obtenidas: cambian las insignias de cada fila, la marca blanca de las barras,
  los contadores sospechosos / no sospechosos y las barras por estilo. **No cambian**: la
  probabilidad de cada dominio, la «P media», el histograma de confianza ni las filas ya guardadas
  en el historial, que conservan el umbral con el que se jugó cada ronda. En «Tú contra la IA» se toma el umbral del momento en que respondes cada caso. Las respuestas ya reveladas quedan registradas con su decisión anterior.
- *Ejemplo real* (los 34.091 dominios de test):

  | Umbral | Phishing detectado (recall) | Acierta cuando acusa (precision) | Falsos positivos | Phishing no detectado |
  |--:|--:|--:|--:|--:|
  | 0,2 | 99,2 % | 43,9 % | 18.521 | 113 |
  | 0,3 | 88,3 % | 55,6 % | 10.278 | 1.713 |
  | 0,5 | 70,9 % | 87,8 % | 1.432 | 4.245 |
  | 0,7 | 66,1 % | 95,1 % | 497 | 4.944 |
  | 0,8 | 63,2 % | 97,5 % | 241 | 5.376 |

**Accuracy · precision · recall · AUC (cuatro cifras de la tarjeta).** Son las métricas del
detector activo en los dominios reales de test, **calculadas con el umbral con el que se entrenó**
(0,5). No cambian al mover el deslizador de la Arena; para ver cómo cambian con el umbral, usa
«Explorar el umbral» en el Training Lab.
- accuracy 0,833: acierta 83 de cada 100 dominios.
- precision 0,878: de cada 100 que marca, 88 eran phishing.
- recall 0,709: de cada 100 phishing, detecta 71.
- AUC 0,862: si coges un phishing y un legítimo al azar, en el 86 % de los casos le da más
  probabilidad al phishing.

#### Lista de resultados

| Elemento | Significado |
|---|---|
| `nombre.test` | El dominio inventado. Es texto, no un enlace. `.test` es un dominio reservado que no existe en Internet. Si el nombre es largo, se corta con «…». |
| `● memorizado` | El nombre exacto existe en el dataset; al pasar el ratón se ve en qué conjunto. La coincidencia puede venir de un patrón aprendido o del azar. Por sí sola no prueba memorización, especialmente si el nombre pertenece a validación o test. Pasa más con temperatura baja (con T = 0,5, 37 de 500 nombres; con T = 1,2, solo 1). |
| `duplicado` | Ya había salido en esta misma ronda. |
| Nombre tachado + `inválido: motivo` | No cumple las reglas de los dominios (etiqueta vacía `..`, guion al principio o al final, etiqueta de más de 63 caracteres…). No se le pasa al detector. |
| Insignia `phish` / `legit` | El estilo que se le pidió al generador. |
| Barra + número | Probabilidad que da el detector. La raya blanca vertical es el umbral: si la barra lo pasa, la barra se pone roja. |
| `SOSPECHOSO` / `NO SOSPECHOSO` | La decisión: probabilidad ≥ umbral. |
| `gen X ms · det Y ms` | Tiempo real que tardaron en generar y en clasificar toda la ronda (unos 50 ms y 25 ms en CPU). |
| **Clic en un dominio** | Abre el Inspector con ese nombre ya cargado: el generador lo «puntúa» y el detector lo explica. |

#### Resultados del experimento
- **generados / válidos / únicos**: cuántos se pidieron, cuántos pasaron la validación y cuántos
  eran distintos entre sí. «Ya en dataset» cuenta apariciones, incluidas las repetidas. «Únicos» incluye las cadenas inválidas.
- **sospechosos / no sospechosos / P media**. Las dos decisiones se cuentan solo entre los nombres válidos. P media promedia las puntuaciones de esos nombres. Puedes tener la misma media con muchas puntuaciones intermedias o con varias muy bajas y muy altas. El histograma permite distinguirlo.
- **Barras por estilo**. El denominador es el número de nombres válidos de ese estilo y el numerador, cuántos superan o igualan el umbral. Si lees `3/8 marcados · 38 %`, hubo tres alertas entre ocho nombres válidos de ese estilo. Los duplicados cuentan como apariciones.
- **Distribución de confianza**: histograma de las probabilidades de la ronda, en 10 tramos y
  apilado por estilo. Si el rojo se acumula a la derecha y el azul a la izquierda, los dos modelos
  «coinciden» en la forma. El eje horizontal agrupa puntuaciones (0,6 representa desde 0,6 hasta menos de 0,7) y el vertical cuenta nombres. Pasa el cursor por una barra para ver los recuentos por estilo.

#### Historial de rondas
- Guarda las últimas 12 rondas **solo en memoria**: se pierde al recargar la página.
- **Gráfica**: % de «phish» marcados (rojo), % de «legit» marcados (azul) y % de válidos (verde
  discontinuo) por ronda.
- **Tabla**: #, temperatura, top-K (∞ = sin límite), umbral, semilla, % válidos, % nuevos (apariciones que no coinciden con el dataset sobre todas las generadas), phish→sospechoso y legit→sospechoso. Un guion significa que no hubo válidos de ese estilo. Los puntos conectan rondas con datos disponibles.
- *Uso típico*: cambia un solo ajuste entre ronda y ronda y observa qué columna se mueve.

#### 🧠 Tú contra la IA
- **Jugar / Nueva partida**: carga 10 dominios **reales** del conjunto de test (que el detector
  nunca vio), 5 de phishing y 5 legítimos. Los nombres se muestran con `.test` en lugar de su
  dominio real.
- **Sospechoso / No sospechoso**: tu respuesta. Después se revelan la respuesta de la IA (con su
  probabilidad), calculada con el umbral actual de la Arena, y la **etiqueta real** del dataset.
- **Siguiente →** aparece cuando has respondido y abre el próximo caso. Al contestar el décimo verás quién ganó o si hubo empate.
- **Marcador** (tus aciertos y los de la IA) y la tira de ✓/✗ por dominio. Cada fracción cuenta aciertos sobre casos respondidos. El resultado corresponde a esa partida equilibrada de diez casos.
- *Por qué con datos reales*: con los dominios inventados no existe una respuesta correcta; aquí
  sí la hay.

---

### 11.A.2 Model Inspector — Generador («¿cómo decide cada carácter?»)

#### Controles

| Control | Concepto y efecto |
|---|---|
| **Estilo** | Igual que en la Arena: qué etiqueta recibe el modelo al empezar. |
| **Temperatura / Top-K / Semilla** | Igual que en la Arena. Aquí sirven para ver su efecto en cada letra. |
| **Texto inicial forzado** | Caracteres que se imponen al modelo; después continúa solo. Ejemplo: escribe `secure-` y verás cómo completa. Solo admite letras, números, `-` y `.`. Cualquier otro carácter da error. |
| **Puntuar este texto completo** | No genera nada: mide cuánto «se lo esperaba» el modelo, letra a letra (el botón pasa a llamarse **Puntuar**). |
| **Generar / Puntuar** | Ejecuta. La insignia «generado en X ms» indica cuánto tardó. |

**Perplejidad observada al pulsar Puntuar.** Cuanto menor es la cifra, más probabilidad asignó el generador a los caracteres observados.

| Texto | Estilo | Perplejidad |
|---|---|--:|
| `secure-login` | phish | **5,89** |
| `secure-login` | legit | 6,91 |
| `gardenstudio` | legit | **6,60** |
| `gardenstudio` | phish | 7,93 |
| `xq9zk2vv` | legit | 124,25 |

En estas mediciones, el generador esperaba más `secure-login` bajo la condición phish y `gardenstudio` bajo legit. A `xq9zk2vv` le asignó poca probabilidad bajo la condición legit.

La puntuación usa las probabilidades originales del modelo para cada carácter del texto impuesto dado lo que le precede. También cuenta el final. En ML esto se llama *teacher forcing*. La semilla solo interviene en sorteos, y la temperatura y top-K solo alteran la distribución de muestreo. Por eso puedes cambiar las barras de color en modo **Puntuar** sin cambiar la perplejidad del mismo texto y estilo. Una perplejidad de 8 equivale a una probabilidad media geométrica de 1/8 para los tokens de esa secuencia. Sirve para comparar qué secuencia esperaba más el generador bajo condiciones comparables. La decisión del detector se calcula aparte.

#### Reproductor paso a paso
- **▶ Reproducir / ⏸ Pausa**: avanza solo, un carácter cada 0,38 s. Al generar se reproduce
  automáticamente; al puntuar, no.
- **← / →** detienen la reproducción y retroceden o avanzan un paso. Para volver al principio tras llegar al final, retrocede con ← o selecciona el primer paso.
- **paso X/Y**: en qué decisión estás. Y = número de caracteres + 1, porque la última decisión es
  «terminar».
- **perplejidad · log-prob · válido**: la sorpresa media del nombre completo, la suma de los
  logaritmos de las probabilidades de cada elección y si el nombre cumple las reglas de los
  dominios.

#### El texto que se va escribiendo
- `<phish>` / `<legit>` (recuadro rojo): el token de estilo con el que arranca.
- El último carácter, en **rojo subrayado**: el que se está eligiendo en este paso (⏹ = terminar).
- **Fondo azul de cada letra anterior**: cuánta atención le presta el modelo (media de las cabezas
  de la capa seleccionada) para elegir la letra actual. Más azul = más la está mirando.
- **Clic en una letra**: salta a ese paso.

#### Candidatos para el paso N
- Los 8 caracteres más probables en este paso, con el contexto (lo escrito hasta ahora).
- **Barra gris**: probabilidad «pura» del modelo (T = 1).
- **Barra de color**: probabilidad real del sorteo, tras aplicar temperatura y top-K. La del
  carácter elegido es roja si aparece entre los ocho candidatos. Un carácter forzado puede aparecer subrayado en el texto aunque no figure en esa lista o aunque su probabilidad de muestreo sea cero.
- `a % → b %`: el paso de una a otra. Ejemplo real: tras `secure-`, «s» pasa de 13,1 % a 30,7 %
  con T = 0,5.
- **Entropía (bits)**: incertidumbre del paso. 0 bits = completamente seguro; ~5,4 bits = duda entre
  unas 42 opciones equiprobables como referencia teórica. El conjunto realmente disponible es menor porque hay símbolos reservados que el generador bloquea. La entropía resume toda la distribución, incluidas las opciones fuera de las ocho filas visibles.

#### Gráfica «incertidumbre por paso»
- **Rojo**: entropía del modelo. **Azul discontinuo**: entropía del sorteo real. **Verde**:
  probabilidad original del modelo para el carácter elegido o impuesto. El verde va de 0 a 1. Las otras dos curvas se miden en bits, aunque compartan el eje vertical.
- Los picos marcan los momentos de duda, por ejemplo el inicio de una palabra nueva tras un `-` o
  un `.`. Los valles, letras casi obvias, como la `n` tras `logi`.
- **Clic en un punto**: salta a ese paso. La línea blanca vertical marca el paso actual.

#### Mapas de atención
- **Botones «capa 1 / capa 2 / …»** seleccionan el bloque. Los números de capas y cabezas dependen del modelo activo. El incluido tiene dos capas y cuatro cabezas por capa.
- Cada cuadro muestra una cabeza. Una cabeza calcula pesos sobre el contexto y los usa para combinar información de las posiciones disponibles. Una celda brillante indica un peso alto en esa cabeza. La predicción final combina también lo que aportan las demás cabezas, capas y transformaciones.
- Cada **fila** es la posición que decide; cada **columna**, una posición a la que puede mirar. Más
  brillo = más atención. El triángulo superior está vacío porque el modelo **no puede mirar al
  futuro** (máscara causal). Cada fila suma 1. El **marco blanco** es la fila del paso actual. Al
  pasar el ratón se ve el valor exacto.
- *Ejemplo de forma real*: para `login-secure` son 2 capas × 4 cabezas × matrices de 13 × 13 (el
  token de estilo + 12 caracteres).

#### Activaciones
- **Botones `embeddings` / `block 1` / `block 2`**: la representación interna de cada posición al
  entrar, tras el primer bloque y tras el segundo.
- Cada **fila** es una posición (el carácter, a la izquierda; ◆ = token de estilo) y cada
  **columna**, una dimensión del vector (64 con el modelo incluido). La fila del paso seleccionado se resalta. Un embedding es una lista de números aprendidos para representar el carácter y aquí se le añade información sobre su posición. **Rojo** = valor positivo, **azul** = negativo; la
  intensidad es la magnitud. Cada mapa tiene su propia escala ±, indicada debajo. Lee el número al pasar el cursor sobre una celda para comparar entre capas.
- Si ves varias columnas con patrones similares, estás viendo valores que varían juntos en esa secuencia. Para saber qué representan harían falta más observaciones y pruebas.

#### Norma L2 del flujo residual
- Una línea por etapa, desde la entrada hasta la salida de cada bloque. Muestra cómo cambia el tamaño del vector interno de cada posición. Cada bloque añade una corrección al vector que recibe y algunas componentes pueden reducirse al combinarse. El eje horizontal recorre las posiciones (◆ indica el estilo) y el vertical muestra la raíz de la suma de cuadrados de los valores. Una norma grande indica magnitud del vector.

---

### 11.A.3 Model Inspector — Detector («¿por qué lo marca?»)

**Caja de texto + Analizar (o Enter).** Puedes pegar un nombre, una variante con `.test` o una dirección completa. Escribe `https://www.ejemplo.com/ruta` y verás `ejemplo` como **Nombre analizado**. El servidor trabaja en minúsculas, retira la ruta y, cuando reconoce el sufijo, elimina el TLD y el prefijo `www.`. Si un nombre contiene puntos internos, comprueba el texto normalizado para saber qué partes llegaron al modelo. Si la sintaxis es inválida, aparece un error. La insignia en milisegundos indica el tiempo del cálculo de contribuciones y votos.

**P(sospechoso), insignia y barra.** Cada árbol hace preguntas sobre la forma del texto y termina en una hoja, un grupo de ejemplos de entrenamiento. La proporción ponderada de phishing en esa hoja es la puntuación del árbol. El bosque promedia esas puntuaciones. Lee 0,73 como 73 puntos sobre 100 en la escala del modelo. Para interpretar 0,73 como una frecuencia de peligro comprobada habría que medir la calibración del modelo en la población donde se usa. La raya blanca marca el **umbral guardado en el detector**. Si en la Arena usas 0,70 y el Inspector usa 0,50, una puntuación de 0,62 puede recibir decisiones distintas. Al comparar, comprueba también que el texto normalizado coincide y que estás usando el mismo checkpoint.

**Base + suma de contribuciones.** La base es la puntuación media en la raíz de los árboles, antes de hacer preguntas. Cada vez que un árbol baja por una rama se anota cuánto cambió la proporción de phishing y se atribuye ese cambio a la característica que originó la pregunta. Se promedian los cambios de todos los árboles. Por ejemplo, con una base de 0,50, una contribución de +0,18 y otra de −0,06, el resultado es 0,62. Al sumar las 19 contribuciones completas se recupera la puntuación, con posibles diferencias en pantalla por redondeo y por las filas omitidas.

| Elemento | Cómo leerlo |
|---|---|
| **Votos de los árboles** | El eje horizontal agrupa puntuaciones de árboles por décimas y el vertical cuenta árboles. Barras concentradas indican acuerdo. Barras repartidas indican discrepancia. El color se asigna por el inicio de cada tramo respecto al umbral, así que un tramo puede incluir votos a ambos lados. |
| **Contribución de cada característica a esta decisión** | Muestra las 12 contribuciones de mayor magnitud. A la derecha del cero, en rojo, suben la puntuación. A la izquierda, en azul, la bajan. Pasa el cursor para ver el valor de la medida y cuánto sumó o restó. El aporte depende del camino que siguió ese nombre por los árboles. |
| **Tabla de características** | Muestra las ocho contribuciones de mayor magnitud. «Valor» es la medida del nombre actual. La mediana es el punto medio de los valores observados en cada grupo de entrenamiento. «Contrib.» muestra el efecto de la medida en esta predicción. Por ejemplo, un valor de tres dígitos y una contribución de +0,04 son magnitudes distintas. |
| **Importancia global de las características** | Mira el comportamiento del bosque en conjunto. En azul, Gini estima cuánto ayudó cada característica a separar las clases en los árboles. En rojo, permutación mide cuánto cae el F1 de validación al mezclar los valores de esa característica entre nombres. Tienen escalas e interpretaciones distintas. Dos medidas parecidas pueden sustituirse entre sí y hacer que su importancia individual por permutación parezca pequeña. |
| **Distribución de probabilidades en test por etiqueta real** | El eje horizontal agrupa puntuaciones y el vertical cuenta casos. Azul y rojo corresponden a las etiquetas del conjunto de datos. Si ambas etiquetas abundan en el mismo tramo, el detector les asignó puntuaciones similares. Usa «Explorar el umbral» para ver cuántas falsas alarmas y omisiones se producen con un corte concreto. |

La **entropía** de la tabla del detector describe la variedad de caracteres dentro de un texto. La **entropía** del panel del generador describe cuán repartidas están las probabilidades para la siguiente elección. Las dos se miden en bits y responden a preguntas diferentes. El apartado [7](#7-modelo-b-el-detector-random-forest) recoge las 19 medidas.

---

### 11.A.4 Training Lab — Generador

Antes de cambiar un valor, identifica los conjuntos de datos. **Train** contiene los nombres que aprende el modelo. **Validación** permite medir cómo responde a otros ejemplos durante el ajuste. **Test** se reserva para la evaluación final del detector. La unidad de aprendizaje del generador es el siguiente carácter. Una iteración toma un lote de nombres, intenta predecir sus siguientes caracteres, mide el error y actualiza los pesos. Una ronda de Arena, en cambio, solo consulta modelos ya entrenados.

**Presets (⚡ Fast · ⚖️ Balanced · 💥 Break it).** Rellenan el formulario con valores probados y
cada uno explica lo que se observó al entrenarlo. **No entrenan solos**. Puedes editar los valores que cargan y luego pulsar **Entrenar**. Si vuelves a pulsar un preset, se vuelven a cargar sus valores.

| Preset | Qué se mide de verdad |
|---|---|
| Fast | 16.160 parámetros, 400 iteraciones en ~3 s, loss de validación ≈3,0 (perplejidad ≈20) |
| Balanced | La arquitectura de referencia (~107k parámetros) con 2.000 iteraciones (el modelo de ejemplo usó 6.000 y llegó a perplejidad 14,9) |
| Break it | Learning rate 1,0 sin clipping: la loss se dispara de 3,7 a ~170, la norma del gradiente llega a ~69 y después se estanca en ≈3,3 |

**Bloque «Arquitectura — requiere reentrenar».** Cambiarlo crea un modelo distinto desde cero.

| Campo | Concepto | Efecto | Metáfora |
|---|---|---|---|
| **Capas** (1–6) | Cuántos bloques Transformer se apilan | Más capas = más capacidad, pero entrena más despacio y puede sobreajustar | Pisos de un edificio: cada uno refina lo que viene de abajo |
| **Cabezas de atención** (1–8) | Cuántas «miradas» paralelas tiene cada capa | Con 64 dimensiones, cuatro cabezas trabajan con 16 dimensiones cada una. Ocho trabajan con ocho cada una. A ancho fijo, se reparte la misma representación entre más cabezas. La dimensión de embeddings tiene que ser divisible entre el número de cabezas (si no, sale un aviso rojo y el botón se desactiva) | Varios lectores que leen la misma frase buscando cosas distintas |
| **Dim. embeddings** (16–256) | Cuántos números describen cada carácter dentro de la red | El factor que más pesa: los parámetros crecen con su cuadrado | La resolución de una foto |
| **Contexto (tokens)** (16–64) | Cuántos caracteres puede ver a la vez | También limita la longitud de los nombres: con 16, los nombres de más de 15 caracteres **se quitan del corpus** y la Arena no podrá pedir más de 15 | El tamaño de la ventana por la que mira |

**Contador de parámetros.** Es exacto (misma fórmula que el modelo). Se pone rojo si pasa de 2
millones, que es el límite del servidor. Ejemplos: 1 capa / 32 dims / contexto 64 = 16.160;
2 capas / 64 dims = 106.880.

**Parámetros de entrenamiento** (se pueden cambiar sin tocar la arquitectura, pero igualmente
requieren entrenar de nuevo):

| Campo | Concepto | Efecto | Metáfora |
|---|---|---|---|
| **Dropout** (0–0,5) | Pone temporalmente a cero una fracción aleatoria de valores internos durante el entrenamiento | Reduce la memorización. Demasiado alto hace que aprenda más despacio | Estudiar tapando partes del libro para no aprender de memoria |
| **Learning rate (log₁₀)** (0,00001–1) | El tamaño de cada corrección. El deslizador va en escala logarítmica: cada paso multiplica, no suma | Bajo = aprende lento pero estable. Alto = rápido hasta que se pasa de largo y explota (ver Break it) | El tamaño de los pasos al bajar una montaña a ciegas |
| **Batch size** (4–256) | Cuántos nombres mira antes de cada corrección | Un lote mayor suele dar una estimación menos ruidosa del gradiente y necesita más memoria. El rendimiento depende del equipo. | Preguntar a 4 o a 256 personas antes de cambiar de opinión |
| **Iteraciones** (10–20.000) | Cuántas correcciones hará | Más = mejor hasta que se estanca. En el modelo de ejemplo la validación casi no mejora después de ~5.000 | Horas de práctica |
| **Grad clip** (0 = off) | Tope a la norma del gradiente antes de cada corrección | Evita que una corrección enorme lo rompa todo. La gráfica de |∇| muestra el valor **antes** del recorte | El limitador de velocidad de un coche |
| **Semilla** | Fija los pesos iniciales y el orden de los ejemplos | Sirve para comparar ensayos en el mismo entorno. Cambios de equipo o librerías pueden afectar la reproducibilidad. | La partida guardada |
| **Warmup (fracción)** | Porcentaje inicial en el que el learning rate sube desde casi 0; después baja en forma de coseno hasta el 10 % | Evita sobresaltos al principio, cuando los pesos aún son aleatorios | Calentar antes de correr |
| **Evaluar cada** | Cada cuántas iteraciones se intenta medir la validación y se generan 3 muestras por estilo. Las evaluaciones intermedias coinciden con un paso de registro, cada 10 iteraciones por defecto. Usa múltiplos de 10 para que coincidan. | Más frecuente = curvas más detalladas, pero un entrenamiento algo más lento | Cada cuánto te pesas cuando estás a dieta |

**▶ Entrenar generador.** Lanza el entrenamiento real. Está desactivado si ya hay otro en curso o si la dimensión no es divisible por las cabezas. El servidor comprueba también los demás límites y muestra un error si rechaza el formulario. Al terminar, el modelo se
guarda en `runs/` y **pasa a ser el activo en toda la app**. Si la loss diverge (deja de ser un
número finito), no se guarda y el modelo anterior sigue activo.

**■ Detener.** Para en la siguiente iteración. **Lo entrenado hasta ese momento se guarda y se
activa**, aunque esté a medias. Para volver al modelo original, usa Checkpoints.

**Registro de etapas.** Mensajes con la hora: construcción del corpus, nº de parámetros y
dispositivo, divergencia (en rojo), «Guardado y activado …» (en verde) y el estado final.

**Insignias de estado.** `● entrenando`, `completed`, `stopped` y `failed`. A la derecha:
`entrenamiento en directo` si las gráficas vienen del entrenamiento actual, o `historial del modelo activo (nombre)` si muestran el historial guardado en el checkpoint. La última ejecución puede seguir visible bajo la etiqueta de directo tras terminar, mientras sus eventos estén cargados. Comprueba el estado y el mensaje «Guardado y activado» para saber qué versión quedó activa.

**Corpus, parámetros y dispositivo.** El encabezado del entrenamiento indica cuántas secuencias entraron en train y validación tras aplicar el límite de contexto. También indica el dispositivo elegido y cuántos pesos está ajustando esa ejecución. El contador junto al formulario corresponde a la configuración que estás preparando.

**Siete indicadores.**

| Indicador | Qué significa | Valor de referencia del modelo de ejemplo |
|---|---|---|
| iteración | Paso actual | 5.999 |
| loss train | Error en el último lote de entrenamiento (oscila mucho) | ~2,69 |
| loss val | Error en nombres que el modelo no ha visto | ~2,74 |
| perplexity | e^(loss val): «entre cuántas letras duda» | ~15,5 (empezó en 42,5) |
| \|∇\| | Norma del gradiente antes del recorte. Mide la magnitud de la señal que orienta la actualización y se pone roja si pasa de 5 | ~0,9 |
| tokens/s | Velocidad real | ~13.000–18.000 en esta CPU |
| memoria | RAM del proceso (o CUDA) y ms por iteración | ~700 MB · ~30–60 ms |

**Seis gráficas en directo.** El eje horizontal es la iteración. Pasa el cursor por una curva para leer el valor del punto. Los indicadores resumen los últimos datos disponibles y la validación se actualiza con menos frecuencia que la pérdida del lote.
1. **Loss**: train por lote (rojo fino, ruidoso), train evaluado (rojo grueso) y validación (azul).
   La entropía cruzada penaliza dar poca probabilidad al carácter correcto. Si train y validación bajan en conjunto, el modelo suele predecir mejor también fuera de train. Si validación empeora de forma sostenida mientras train mejora, hay indicios de sobreajuste.
2. **Perplejidad en validación**: la misma información en una escala más intuitiva.
3. **Norma del gradiente** (escala logarítmica): saltos verticales iguales representan multiplicaciones del valor. Examina los picos junto con la pérdida para distinguir un lote difícil de una inestabilidad sostenida.
4. **Learning rate**: rampa de warmup y caída en coseno. Sirve para comprobar que la planificación
   es la esperada.
5. **Tokens por segundo**: baja si el ordenador está ocupado con otra cosa.
6. **Tiempo por iteración y memoria**: las dos curvas comparten eje y usan unidades distintas. Compara cada curva consigo misma. Un pico en milisegundos indica un paso más lento. El valor RSS muestra cuánta RAM ocupa el proceso completo. Si se usa CUDA, el indicador de memoria también muestra la memoria asignada por PyTorch en la GPU.

**Muestras de la última evaluación.** Tres nombres de cada estilo generados durante el
entrenamiento. Al principio son ruido (`3f7zonjew3wwpq3…`); en el modelo de ejemplo, en la
iteración 2.000 ya salían `asteollres-104798.web` o `sarmerivere-co-jp.web`.

---

### 11.A.5 Training Lab — Detector

En el detector el entrenamiento consiste en construir árboles a partir de ejemplos etiquetados. Cada pregunta divide los ejemplos de una rama en dos grupos. Una **hoja** es el grupo final que queda tras las preguntas. Como muchos árboles aprenden de muestras y subconjuntos de variables distintos, promediar sus predicciones suele dar un resultado más estable.

**Presets.** Fast (20 árboles poco profundos, 10.000 dominios, F1 en test ≈0,78), Balanced (la
referencia, F1 ≈0,785, AUC ≈0,86) y Break it (árboles sin límite, hojas de 1 muestra, sin pesos y
solo 2.000 dominios: accuracy en train 0,94 frente a 0,83 en validación, F1 en test ≈0,75 →
sobreajuste).

| Campo | Concepto | Efecto | Metáfora |
|---|---|---|---|
| **Número de árboles** (1–500) | Cuántos árboles votan | Más = predicción más estable, con mejoras pequeñas a partir de ~25 en este problema; el tiempo crece en proporción | El tamaño del jurado |
| **Profundidad máxima** (vacío = sin límite) | Cuántas preguntas seguidas puede hacer cada árbol | Más profundo = reglas más finas y más riesgo de memorizar | La longitud de un cuestionario |
| **Muestras mínimas por hoja** (1–200) | Cuántos ejemplos tiene que cubrir cada respuesta final | Más alto = reglas más generales y menos ruido | No sacar conclusiones de un solo caso |
| **Pesos de clase** | `balanced` da más peso a la clase minoritaria (en train hay un 34 % de phishing) | Sin pesos, tiende a decir «legítimo» más a menudo | Dar más tiempo de palabra a quien está en minoría en un debate |
| **Umbral de decisión** | Se guarda con el modelo; con él se calculan las métricas de validación y de test y es el umbral por defecto en el Inspector | **No cambia los árboles**, solo el corte | La nota mínima para aprobar |
| **Tamaño del conjunto de entrenamiento** | Cuántos dominios de train se usan (muestra estratificada) | Más datos = mejor generalización, a cambio de más tiempo | Cuántos casos has estudiado |
| **Semilla** | Fija la muestra y el azar de los árboles | Resultados reproducibles | La partida guardada |

**▶ Entrenar detector / ■ Detener.** Igual que en el generador. Si lo detienes, se guarda el
bosque con los árboles construidos hasta ese momento (como mínimo una tanda) y se evalúa en test.

**Gráfica «métricas en validación a medida que se añaden árboles».** El bosque crece en unas 20
tandas y tras cada una se mide en validación: F1 (azul), precision (verde), recall (rojo), AUC
(morado) y accuracy en train (gris discontinuo). Con el modelo de ejemplo, las curvas casi no se
mueven a partir de los 25 árboles. La curva gris usa accuracy y las curvas de validación usan otras métricas, así que la distancia entre líneas no mide directamente sobreajuste. Busca si el rendimiento en train mejora mientras las métricas de validación se estancan o empeoran. Cuando puedas, contrasta la misma métrica en ambos conjuntos.

**Panel de test (no visto).** Consulta las definiciones de accuracy, precision, recall y AUC en la tarjeta azul de la Arena. F1 baja si precision o recall se queda atrás. Las cinco métricas, la matriz de confusión y una línea de resumen
(F1 y AUC en validación, filas usadas, segundos de entrenamiento y milisegundos por cada 1.000
dominios).

**Matriz de confusión** (umbral 0,5, modelo de ejemplo):

| | Predicho legítimo | Predicho sospechoso |
|---|--:|--:|
| **Real legítimo** | 18.070 aciertos | 1.432 falsos positivos (falsas alarmas) |
| **Real phishing** | 4.245 phishing no detectado | 10.344 phishing detectado |

Las casillas azules son aciertos y las rojas, errores; la intensidad del color es proporcional a la
cantidad.

**Explorar el umbral (sin reentrenar).** Mueve el umbral de 0,05 en 0,05 y el servidor recalcula
las métricas reales de test. Aparece la insignia «umbral X (recalculado)». Las métricas que dependen del corte y la matriz pueden cambiar. AUC se obtiene del orden de las puntuaciones y conserva su valor. La curva de la derecha muestra precision (verde), recall (rojo) y F1 (azul)
para todos los umbrales, con una línea blanca en el actual. Es una consulta sobre test. El Inspector usa el umbral guardado con el modelo y la Arena tiene su propio deslizador. Si estás seleccionando un corte para un sistema real, haz la selección con validación y reserva test para la comprobación final.
Ejemplos en la tabla del umbral de la Arena (11.A.1).

---

### 11.A.6 Training Lab — Checkpoints
- Dos listas: generador (`.pt`) y detector (`.joblib`).
- `ejemplo` = incluido en el repositorio (`checkpoints/`); `run` = entrenado desde la interfaz
  (`runs/`, fuera de git). Junto a cada uno se muestra su tamaño.
- **activo** = el que usan las siguientes operaciones de Arena e Inspector. Vuelve a pulsar **Iniciar duelo**, **Generar**, **Puntuar** o **Analizar** para obtener resultados con la nueva versión. Las salidas ya calculadas pueden seguir a la vista.
- **activar** = cambia al instante el modelo de toda la app. Está desactivado mientras hay un
  entrenamiento en curso.

---

### 11.A.7 Recetas de experimentos (para probar en 5 minutos)

1. **Temperatura y muestreo.** En el Inspector, escribe `jardin-` y activa **Puntuar**. Haz dos consultas con temperaturas 0,5 y 1,5, manteniendo el estilo. Compara la barra gris y la de color en el mismo paso y observa cómo la entropía del muestreo responde al cambio.
2. **Repetición.** En Arena, selecciona un único estilo y ejecuta un duelo con T = 0. Mira cuántas salidas únicas hay. Repite con T = 1 y la misma semilla. Examina por separado «duplicado» y «memorizado»: uno compara dentro de la ronda y el otro contra los datos.
3. **Qué estilo esperaba más el texto.** En el Inspector, puntúa el mismo texto con estilo phish y con estilo legit. Una perplejidad menor en un estilo indica que esa condición asignó más probabilidad al texto. Para una decisión de clasificación, consulta el panel del detector.
4. **Romper el modelo.** En el Training Lab, Break it → Entrenar: la loss explota y el |∇| se pone
   rojo. Después, Checkpoints → activa `generator_default.pt`.
5. **Sobreajuste del detector.** Break it del detector → la accuracy en train se separa de la de
   validación.
6. **Coste de una alarma sensible.** Explorar el umbral en 0,2 frente a 0,7: cuántas falsas
   alarmas cuesta cazar cada phishing de más.
7. **Explicar un caso.** Escribe tu propio dominio en el detector y mira cuánto suma o resta cada medida en ese caso.

### 11.A.8 Alcance de la interfaz
- El historial de rondas vive en la página y se pierde al recargar. El progreso de un entrenamiento en curso viene del servidor y el navegador vuelve a suscribirse al reabrir la página.
- El formulario usa la configuración del servidor para `weight_decay` (0,01), `log_interval` (10) y la selección automática de dispositivo. El script `scripts/train_generator.py` ofrece opciones adicionales.
- Para recuperar un modelo anterior, actívalo en **Checkpoints**.

---

## 11.B Comparación con el proyecto de referencia: MLP con embeddings frente a Random Forest

> **Fuente y alcance.** Lo que se dice del proyecto «Generador de Excusas» procede **solo** de su
> descripción en `PHANTOMLINK.md`. No se ha leído ni reutilizado su código, como exige
> `INSTRUCTIONS.md`. Cuando se habla de rasgos generales de una técnica (por ejemplo, «un MLP
> necesita…»), se trata de propiedades conocidas de esa familia de modelos, **no de una
> verificación de su implementación concreta**.

### Qué hace cada clasificador

Un **MLP** o perceptrón multicapa transforma las representaciones numéricas del texto en puntuaciones mediante varias operaciones encadenadas. Sus pesos se corrigen con el gradiente. Un **Random Forest** aprende muchas secuencias de preguntas sobre medidas que ya hemos definido y combina sus respuestas. En **multietiqueta**, una misma frase puede recibir varias categorías a la vez. En **clasificación binaria**, la decisión tiene dos opciones.

| | Generador de Excusas | PHANTOMLINK |
|---|---|---|
| **Tarea** | Asignar a la frase que escribe el usuario una o varias categorías (multietiqueta), que luego condicionan la excusa generada | Decidir si un nombre de dominio se parece a los de phishing (binaria: sí / no, con probabilidad) |
| **Lugar en el flujo** | **Antes** del generador: Clasificador → Generador → Excusa | **Después** del generador: Generador → Clasificador → Evaluación |
| **Modelo** | Red neuronal: capa de *embeddings* + perceptrón multicapa (MLP), en PyTorch | `RandomForestClassifier` (conjunto de árboles de decisión), en scikit-learn |
| **Entrada** | El texto, convertido en vectores aprendidos (*embeddings*) | 19 características **calculadas a mano** sobre el nombre (longitud, dígitos, guiones, entropía, partes…) |
| **Cómo aprende** | Descenso del gradiente: muchas pasadas corrigiendo pesos poco a poco | Cada árbol divide los datos con las preguntas que mejor separan las clases; no hay gradiente ni épocas |
| **Datos** | Sintéticos; su README reconoce que son repetitivos y que hay fuga entre entrenamiento y validación | Reales (PhiUSIIL, 212.851 nombres), con variantes con y sin `www.` unificadas y los conjuntos separados por dominio registrado |

### Diferencias de fondo

**1. Quién decide qué mirar.**
- En el **MLP con embeddings**, la red aprende sola qué rasgos del texto importan: cada palabra o
  carácter se convierte en un vector cuyos números se ajustan durante el entrenamiento. Es potente
  con texto libre, donde sería imposible enumerar a mano todos los rasgos.
- En el **Random Forest** de PHANTOMLINK, **una persona decide los rasgos** (las 19
  características) y el bosque decide cuánto pesa cada uno. Es adecuado aquí porque un nombre de
  dominio es corto y su estructura (partes, dígitos, guiones) se puede describir bien a mano.
- *Metáfora*: el MLP es un aprendiz al que das textos y que descubre por sí mismo en qué fijarse;
  el bosque es un comité de inspectores con una lista de comprobación, que solo decide qué casillas
  pesan más.

**2. Explicabilidad.**
- **Random Forest**: cada predicción se descompone **exactamente** en contribuciones por
  característica (`base + Σ = probabilidad`, verificado por una prueba automatizada), y se pueden
  ver los votos de cada árbol. Puedes describir cuánto aportaron las medidas al recorrer los árboles en ese caso.
- **MLP con embeddings** (rasgo general de la técnica): se pueden visualizar sus embeddings y
  activaciones, como hace el proyecto de referencia, pero esos números no tienen un significado
  directo. Explicar una decisión concreta requiere técnicas aproximadas (gradientes, atribuciones).
- *Metáfora*: el bosque es un tique de compra con cada línea detallada; el MLP es un chef que te
  deja ver la cocina, pero no te da la receta escrita.

**3. Datos y fiabilidad de las métricas.**
- Con datos sintéticos repetitivos y con fuga entre conjuntos (según el README del proyecto de
  referencia), las métricas de validación tienden a salir **optimistas**: el modelo puede acertar
  por haber visto casi lo mismo.
- PHANTOMLINK evalúa en un conjunto de **test** separado por dominio registrado y nunca usado para
  decidir nada durante el entrenamiento. Por eso sus cifras (F1 0,785, AUC 0,862) son modestas,
  pero defendibles.
- *Metáfora*: aprobar un examen con las mismas preguntas del cuaderno de ejercicios frente a uno
  con preguntas nuevas.

**4. Coste y estabilidad del entrenamiento.**
- **Random Forest**: unos 5–10 s en CPU, pocos hiperparámetros, no diverge (no hay learning rate que
  pueda explotar) y apenas necesita preprocesado: no hay que normalizar las escalas.
- **MLP** (rasgo general): necesita elegir learning rate, épocas y tamaño, puede sobreajustar o
  volverse inestable y suele requerir más ajustes, a cambio de mayor flexibilidad con datos
  complejos. En PHANTOMLINK esa parte «delicada» la cubre el generador Transformer: el preset Break
  it muestra lo que pasa con un learning rate excesivo.

**5. Límites de cada enfoque.**
- **Random Forest** de PHANTOMLINK: solo sabe lo que dicen sus 19 características. Si dos nombres
  tienen la misma longitud, partes y dígitos pero significados distintos (`paypal-login` frente a
  `garden-shop`), solo lo distinguen señales como el recuento de palabras sospechosas. No entiende
  el significado ni detecta el parecido con una marca (`paypa1`) más allá de esos rasgos.
- **MLP con embeddings** (general): puede captar relaciones entre caracteres o palabras sin
  enumerarlas, pero con pocos datos o datos repetitivos tiende a memorizar en lugar de generalizar.

### Resumen

| Criterio | MLP + embeddings (Excusas) | Random Forest (PHANTOMLINK) |
|---|---|---|
| Rasgos | Aprendidos por la red | Diseñados a mano (19) |
| Tipo de salida | Multietiqueta (categorías) | Binaria con probabilidad |
| Explicación por predicción | Indirecta (visualizar activaciones) | Exacta (contribuciones + votos) |
| Entrenamiento | Gradiente, sensible a hiperparámetros | Construcción de árboles, segundos, estable |
| Datos de evaluación | Sintéticos, con fuga reconocida | Reales, separados por dominio registrado |
| Fortaleza principal | Flexibilidad con texto libre | Transparencia y robustez con pocos rasgos claros |
| Riesgo principal | Memorizar datos repetitivos | Quedarse ciego a lo que sus rasgos no miden |

**Conclusión.** No hay un modelo «mejor» en abstracto: cada proyecto eligió lo que encaja con su
entrada. Para frases libres, aprender los rasgos (embeddings + MLP) es razonable; para nombres de
dominio cortos con estructura medible, un Random Forest sobre características explícitas ofrece
algo que el proyecto de referencia no tiene: **poder explicar cada decisión de forma exacta** y
evaluarla con datos reales sin fuga. Además, PHANTOMLINK combina dos familias distintas (red
neuronal en el generador y árboles en el detector), mientras que el proyecto de referencia usa dos
redes neuronales. Los dos generadores, en cambio, son de la misma familia: Transformers pequeños a
nivel de carácter entrenados desde cero.

## 11.C Comparación de los modelos de deep learning

> Misma regla que en 11.B: lo que se dice del proyecto de referencia procede **solo** de
> `PHANTOMLINK.md`. Los datos que ese documento no da (tamaño exacto, hiperparámetros, detalles de
> la arquitectura) aparecen como **«no especificado»**; no se suponen.

Inventario de redes neuronales:
- **Generador de Excusas**: dos. Un clasificador (embeddings + MLP) y un generador (Mini-GPT,
  Transformer decoder-only a nivel de carácter).
- **PHANTOMLINK**: una, el generador (Mini-GPT a nivel de carácter). El otro modelo es un Random
  Forest, que no es deep learning (ver 11.B).

### 11.C.1 Los dos generadores (Transformer frente a Transformer)

**Autorregresivo** significa que cada carácter añadido pasa a formar parte del contexto para decidir el siguiente. **Causal** significa que la predicción solo puede usar posiciones disponibles hasta ese punto. **Multicabeza** significa que varias operaciones de atención se hacen en paralelo. Una **conexión residual** conserva la representación que llega a un bloque y le suma lo que el bloque calcula. La **normalización** ajusta la escala de los valores internos para facilitar el aprendizaje. En el Inspector puedes recorrer estas operaciones por posición con el reproductor, los mapas y la gráfica de norma L2.

Comparten familia: Transformer *decoder-only*, entrenado desde cero, que predice el siguiente
carácter de forma autorregresiva, con atención causal multicabeza, embeddings, conexiones
residuales y normalización (las cinco piezas que `PHANTOMLINK.md` enumera para el proyecto de
referencia y que también tiene este). Ninguno usa un modelo preentrenado. Las diferencias están en
**qué generan, cómo se condicionan, con qué datos aprenden y cómo se evalúan**.

| | Generador de Excusas | Generador de PHANTOMLINK |
|---|---|---|
| **Qué produce** | Una excusa: frases en español | Un nombre de dominio: cadenas cortas de `a-z 0-9 - .` (máx. 63 caracteres) |
| **Condicionamiento** | Las categorías que detecta el clasificador previo (el mecanismo exacto no está especificado) | Un token de estilo (`<legit>` o `<phish>`) al principio de la secuencia, elegido por el usuario |
| **Tamaño y configuración** | No especificado | 106.880 parámetros: 2 capas, 4 cabezas, 64 dimensiones, contexto 64, vocabulario 42 (38 caracteres + 4 especiales), dropout 0,1, weight tying |
| **Datos** | Sintéticos; el README reconoce que son repetitivos y que hay fuga entre entrenamiento y validación | 139.127 nombres reales (train) y 38.039 de validación, separados por dominio registrado |
| **Entrenamiento** | Desde el navegador. Curvas de entrenamiento y validación. Previsiones del tiempo de entrenamiento, heurísticas según su README | Desde el navegador o un script. AdamW + warmup + coseno + clipping configurables, telemetría por SSE (loss, perplejidad, \|∇\|, lr, tokens/s, memoria), parada y detección de divergencia. Sin previsiones de tiempo (recortadas a propósito) |
| **Visualización** | Probabilidades del siguiente carácter, matrices de atención, embeddings y activaciones, curvas, efecto de temperatura y learning rate | Probabilidades (modelo frente a muestreo real), entropía por paso, atención por capa y cabeza, activaciones por capa, norma del flujo residual, reproducción paso a paso y puntuación de un texto propio |
| **Evaluación de la salida** | Loss y validación sobre datos sintéticos | Perplejidad de validación (42,5 → 14,9) **más** una evaluación externa: otro modelo independiente (el detector) juzga las salidas, y se mide cuánto se memoriza |

**1. La dificultad de la tarea no es comparable.**
Generar una frase coherente en español exige gramática, concordancia y sentido a lo largo de
decenas de caracteres. Generar un dominio solo exige patrones locales: sílabas pronunciables,
guiones y dígitos en sitios plausibles. `PHANTOMLINK.md` lo plantea así: «generar cadenas cortas
con estructuras reconocibles es mucho más factible que crear textos largos de calidad desde cero».
Con un modelo de ~100.000 parámetros, el generador de excusas compite contra una tarea que suele
desbordar a un modelo pequeño; el de PHANTOMLINK juega en una tarea a su medida.
- *Metáfora*: pedir a un aprendiz de pintor un retrato realista frente a pedirle un logotipo
  geométrico. Con la misma formación, el logotipo le saldrá mucho más convincente.

**2. El condicionamiento: quién decide el «tono».**
- En el de referencia, el condicionamiento viene de **otro modelo**: si el clasificador se
  equivoca de categoría, el generador escribe una excusa del tipo equivocado. Los errores se
  encadenan.
- En PHANTOMLINK, el estilo lo elige el usuario y entra como primer token, así que el error del
  primer modelo no se propaga al segundo. Su efecto está **medido**: con 500 nombres por estilo a
  T = 0,8, el detector marca el 1 % de los «legit» y el 47 % de los «phish», y los «phish» tienen
  más partes (1,57 frente a 1,00) y más dígitos (0,49 frente a 0,00).
- *Metáfora*: en uno, un intérprete le dice al actor qué papel hacer (y si traduce mal, el papel
  está mal); en el otro, el director se lo dice directamente.

**3. Datos sintéticos frente a reales.**
Con datos sintéticos y repetitivos, un Transformer pequeño tiende a **reproducir las plantillas**
con las que se crearon los datos: la loss baja mucho, pero lo aprendido son las plantillas, no el
lenguaje. Con nombres reales hay mucha más variedad y la loss se queda más alta (2,70 de
validación), pero lo que aprende son patrones del mundo real: `-co-jp` (suplantación de dominios
japoneses), `.web` (phishing alojado en plataformas) o subdominios numéricos.
- *Metáfora*: aprender un idioma repitiendo las 20 frases del libro de texto frente a escuchar la
  radio. Con lo primero hablas «perfecto» solo esas frases.

**4. Cómo se sabe si el generador es bueno.**
- La loss y la perplejidad dicen cuánto acierta el siguiente carácter, no si el resultado sirve.
  Con fuga entre entrenamiento y validación (reconocida en el proyecto de referencia), además
  salen optimistas.
- PHANTOMLINK añade dos medidas que no dependen de la propia red:
  - un **juez independiente** (el Random Forest) que puntúa las salidas;
  - una **tasa de memorización**: cuántos nombres generados existen literalmente en el dataset.
    Baja de 37/500 con T = 0,5 a 1/500 con T = 1,2, y con T = 0 todo sale igual.
- *Metáfora*: no basta con que el alumno saque buena nota en su propio examen de práctica; hace
  falta un examinador externo y comprobar que no ha copiado.

**5. Inspección: lo mismo, más afinado.**
Los dos permiten ver probabilidades, atención y activaciones; es la gran fortaleza del proyecto de
referencia y PHANTOMLINK la mantiene. Lo que este añade:
- separa la probabilidad **del modelo** de la **realmente muestreada** (tras temperatura y top-K),
  que es lo que de verdad decide cada letra;
- permite **puntuar un texto propio** (perplejidad por estilo);
- garantiza que la traza coincide exactamente con la generación: una prueba automatizada comprueba
  que la atención es causal y que cada fila suma 1.

### 11.C.2 Las dos redes del proyecto de referencia frente a la red de PHANTOMLINK

El proyecto de referencia usa también deep learning para clasificar (embeddings + MLP).
Comparación de su clasificador con nuestro Transformer, en rasgos **generales** de cada
arquitectura:

| | MLP con embeddings | Transformer (MiniGPT) |
|---|---|---|
| **Orden de la entrada** | Un MLP no tiene, por sí mismo, noción de posición: depende de cómo se combinen los embeddings antes de entrar (no especificado en el proyecto) | Embeddings de posición + atención causal: sabe qué carácter va antes de cuál |
| **Relación entre partes** | Mezcla la información de la entrada en capas densas | La atención decide, para cada posición, en qué otras fijarse; se puede visualizar |
| **Salida** | Una puntuación por categoría | Una distribución sobre el siguiente carácter, repetida paso a paso |
| **Uso típico** | Clasificar una entrada completa | Generar o puntuar secuencias |

PHANTOMLINK usa deep learning **solo donde aporta algo** (generar secuencias) y un modelo clásico
donde la transparencia importa más (decidir y explicar por qué).
- *Metáfora*: usar el bisturí para operar y la regla para medir, en lugar de usar el bisturí
  también para medir.

### 11.C.3 Resumen

| Criterio | Generador de Excusas | PHANTOMLINK |
|---|---|---|
| Redes neuronales | 2 (MLP + Transformer) | 1 (Transformer) + 1 modelo clásico |
| Tarea generativa | Frases en español (difícil para un modelo pequeño) | Dominios cortos (a la medida de un modelo pequeño) |
| Condicionamiento | Salida de otro modelo (los errores se encadenan) | Token de estilo elegido por el usuario (efecto medido) |
| Datos | Sintéticos, con fuga reconocida | Reales, separados por dominio registrado |
| Evaluación de la salida | Métricas internas de la propia red | Métricas internas + juez externo + memorización |
| Inspección | Probabilidades, atención, embeddings, activaciones | Lo mismo + muestreo real, entropía, puntuación de textos y trazas verificadas por pruebas |

**Lo que el proyecto de referencia hace igual de bien o mejor.** Explora más en profundidad las
representaciones internas: visualiza los embeddings del clasificador, que PHANTOMLINK no tiene
porque su detector no es una red neuronal, y la proyección 2D de embeddings se recortó por tiempo.
Además, ya estaba terminado y publicado cuando PHANTOMLINK empezó.

## 12. Pruebas automatizadas

`pytest` (configurado en `pytest.ini`). Usan un mini-dataset sintético (fixture `tiny_dataset`)
para ser rápidas e independientes del CSV real, más una prueba sobre el dataset real si existe.
Cubren: normalización (www, sufijos, puertos, IPs), determinismo de la división, validez
sintáctica, valores de características, causalidad y normalización de la atención, semillas
reproducibles, coherencia de las trazas, guardado/carga de ambos modelos, que el entrenamiento
reduce la loss, parada de entrenamientos, métricas y contribuciones exactas del Random Forest,
semántica de etiquetas, y la API completa (estado, Arena, normalización de entradas,
inspectores, validación, un solo entrenamiento simultáneo, parada, SSE).

Estado: **32/32 pruebas superadas** (~6 s sin otros procesos; ~27 s si hay un entrenamiento en
paralelo). La prueba número 32 verifica el endpoint «Tú contra la IA»: los dominios salen del
conjunto test, con etiqueta real y equilibrados.

Además de pytest, se hicieron pruebas manuales automatizadas en el navegador (paso 11) y una prueba
de seguridad de rutas: `GET /assets/../../../../etc/passwd` devuelve 404.

## 13. Registro de decisiones (y por qué)

| # | Decisión | Motivo |
|---|---|---|
| D1 | Columna interna `is_phishing` mapeada en un único sitio | El CSV usa 1 = legítimo; evitar errores de interpretación |
| D2 | Eliminar `www.` | Fuga de etiqueta (100 % vs 41 %) |
| D3 | Eliminar el TLD y usar `.test` para mostrar | Coherencia entre entrenamiento y Arena; dominios imposibles de visitar |
| D4 | Agrupar por eTLD+1 ICANN (estricto) | Evitar que se memoricen plataformas de alojamiento entre conjuntos |
| D5 | Hash SHA-256 para asignar conjuntos | Determinista e independiente del orden de los datos |
| D6 | PSL incluida en `tldextract`, sin red | Reproducible y sin peticiones externas |
| D7 | Random Forest con `warm_start` | Permite ver métricas reales mientras se añaden árboles |
| D8 | Importancia por permutación además de impureza | La de impureza favorece variables continuas; la de permutación es más fiable |
| D9 | Atención implementada a mano | Exponer las matrices reales en el Inspector |
| D10 | Weight tying | Menos parámetros y mejor generalización en modelos pequeños |
| D11 | Inspección con una pasada causal | Mismos valores que paso a paso, más simple y rápido |
| D12 | No guardar modelos divergidos | Un modelo con NaN no puede generar nada |
| D13 | Semillas distintas por estilo en la Arena | Evitar que ambos estilos compartan el mismo azar |
| D14 | Un solo proceso FastAPI que sirve también el frontend | Arranque con un comando, sin infraestructura extra |
| D15 | `torch.load(weights_only=True)` | Cargar checkpoints sin ejecutar código arbitrario |
| D16 | «Break it» del generador con learning rate 1,0 | Con 0,3 solo se estancaba; con 1,0 la explosión se ve en las gráficas (medido) |
| D17 | Textos de los presets con cifras medidas | No prometer comportamientos que no se han comprobado |
| D18 | «Tú contra la IA» con datos de test reales | Con los dominios generados no hay verdad con la que puntuar |
| D19 | El umbral de la Arena se reaplica en el cliente | Respuesta instantánea, con la misma regla (p ≥ umbral) sobre las probabilidades reales |
| D20 | Semilla autoincremental (opcional) en la Arena | Rondas variadas por defecto y reproducibles al desmarcarla |
| D21 | Sin fuentes ni scripts externos en la UI | Funciona sin conexión y no hace peticiones a terceros |
| D22 | `domains.csv` no se sube al repositorio | Pesa 9 MB y se regenera de forma determinista con `prepare_data.py` (`run.sh` lo hace solo) |
| D23 | Música de los vídeos compuesta por código | Garantiza que no hay derechos de terceros, sin depender de licencias web que pueden cambiar |
| D24 | Ganancia fija en lugar de `loudnorm` dinámico | Mantener la curva intro → clímax → cierre (rango de 6,6 LU frente a 2 LU) |

## 14. Problemas encontrados y cómo se resolvieron

1. **Sin `sudo` ni Node.js/PyTorch instalados** → entorno virtual Python y binario de Node en la
   carpeta del usuario.
2. **`view` de PyTorch falla** sobre tensores no contiguos (al recortar `batch[:, 1:]`) →
   sustituido por `reshape`.
3. **Aviso de scikit-learn** al combinar `class_weight="balanced"` con `warm_start` (recalcula
   los pesos en cada tanda) → se calculan los pesos una vez con `compute_class_weight` y se
   pasan como diccionario. La opción `balanced_subsample` se retiró por el mismo motivo.
4. **Dominio `www.www.woodandthimble.com`** conservaba un `www.` (detectado por una prueba sobre
   el dataset real) → se eliminan todos los `www.` iniciales. El dataset pasó de 212.852 a
   212.851 filas y se reentrenó el clasificador.
5. **Prueba de parada demasiado rápida**: se detenía el entrenamiento antes de emitir la primera
   métrica → la prueba ahora espera a la primera métrica antes de detener.
6. **Reconexiones SSE duplicaban eventos**: el navegador reconecta solo y el servidor volvía a
   enviar todo desde el principio → el servidor respeta la cabecera `Last-Event-ID` y el cliente
   descarta los eventos que ya tiene (por su número de secuencia).
7. **«Break it» no rompía nada** (learning rate 0,3) → medido y cambiado a 1,0 (ver sección 8).
8. **Desbordamiento horizontal en móvil** (442 px en una pantalla de 390) → filas de la Arena
   convertidas en tarjetas y rejillas responsive (paso 13).
9. **Detalles visuales**: etiquetas «requiere reentrenar» que se solapaban (ahora es un bloque
   «Arquitectura»), eje del learning rate cortado (ahora usa notación científica), porcentajes de
   candidatos que se partían y etiquetas que faltaban en la gráfica de importancia.
10. **Reinicio del servidor**: un `pkill -f` cerró también la propia consola, porque el patrón
    coincidía con su línea de comandos. Lección: arrancar el servidor como proceso independiente.

## 15. Limitaciones honestas

- El detector analiza **solo la forma del nombre**: no ve la web, ni el TLD, ni certificados, ni
  reputación. "No sospechoso" ≠ seguro. Recall en test 0,71: ~29 % del phishing pasa.
- El dataset tiene sesgos de recogida (p. ej. el `www.`/`https` de los legítimos, procedentes de
  listas de sitios populares). Se han neutralizado los conocidos, pero puede haber otros.
- El generador es diminuto: produce nombres con "aspecto" plausible, no marcas reales coherentes.
  Puede reproducir nombres vistos en el dataset (se indica cuando ocurre).
- La "coincidencia" entre estilo solicitado y decisión del detector no es una medida de acierto.
- Las métricas de test cambian algo con la configuración (p. ej. el preset Fast da F1 ≈0,78) y
  la evaluación es estricta por la agrupación por plataformas: el porcentaje de phishing varía
  entre conjuntos (34 % / 50 % / 43 %).
- Las importancias por impureza favorecen las variables continuas; por eso se muestra también la
  importancia por permutación.
- Los tokens/s y los ms por iteración dependen de la carga del equipo (se ven picos cuando hay
  otros procesos activos).

## 16. Cómo reproducir todo desde cero

```bash
cd phantomlink
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py --download     # datos
.venv/bin/python scripts/train_classifier.py            # detector  (~10 s)
.venv/bin/python scripts/train_generator.py --iterations 6000 --batch-size 64 --eval-interval 500
.venv/bin/python -m pytest -q                           # pruebas
```
Arrancar la aplicación:

```bash
# (una vez) Node: si no está instalado, descarga el binario de nodejs.org y añádelo al PATH
cd frontend && npm install && npm run build && cd ..
./run.sh                     # http://127.0.0.1:8000  (documentación de la API en /docs)
```

Guía para una persona no técnica:
1. Abre una terminal en la carpeta `phantomlink`.
2. Ejecuta `./run.sh` y espera a ver «Uvicorn running on http://127.0.0.1:8000».
3. Abre esa dirección en el navegador.
4. En **AI Arena** pulsa «⚡ INICIAR DUELO». Sube la temperatura y repite para ver cómo cambian
   los resultados en el historial.
5. Haz clic en cualquier dominio para ver en el **Inspector** cómo se eligió cada letra.
6. En **Training Lab** pulsa «⚡ Fast» y luego «Entrenar generador» para ver un entrenamiento real
   en unos segundos. Prueba también «💥 Break it».
7. Para volver al modelo original: Training Lab → Checkpoints → `generator_default.pt` → activar.
8. Para pararlo todo, pulsa Ctrl+C en la terminal.

## 17. Estructura de archivos

```
phantomlink/
├── backend/phantomlink/
│   ├── paths.py          rutas del proyecto
│   ├── data.py           normalización, división y preparación del dataset
│   ├── features.py       19 características léxicas
│   ├── classifier.py     Random Forest: entrenamiento, métricas, contribuciones, persistencia
│   ├── generator/
│   │   ├── tokenizer.py  caracteres + tokens de estilo
│   │   ├── model.py      MiniGPT (atención causal propia)
│   │   ├── train.py      bucle de entrenamiento con telemetría
│   │   ├── sampling.py   generación y trazas de inspección
│   │   └── checkpoint.py guardado/carga .pt
│   ├── arena.py          duelo generador → validación → detector
│   ├── jobs.py           gestor de entrenamientos (uno a la vez, eventos)
│   ├── presets.py        Fast / Balanced / Break it
│   └── api.py            FastAPI
├── scripts/              prepare_data.py, train_classifier.py, train_generator.py
├── tests/                pruebas pytest
├── checkpoints/          modelos de ejemplo
├── data/processed/       dataset preparado + informe (data/raw no se sube)
├── frontend/
│   ├── src/api.js        cliente HTTP + SSE
│   ├── src/App.jsx       cabecera, pestañas y estado
│   ├── src/components/   ui.jsx, Heatmap.jsx
│   └── src/pages/        Arena, Challenge, Inspector, TrainingLab
├── docs/demo_16x9.mp4 · demo_9x16.mp4   vídeos para redes (43–45 s)
├── scripts/record_demo.py · record_social.py · compose_music.py   grabación y música de los vídeos
├── run.sh                arranque con un comando
├── README.md             documentación pública
├── requirements.txt · pytest.ini · .gitignore
├── runs/                 modelos entrenados desde la UI (no se suben)

```
