# DATA CARD — Dataset unificado de defectos en infraestructuras

**Versión:** 1.0 · **Fecha:** 2026-11-10 · **Generado con:** `python -m src.data.unify` (semilla 42)
**Formato:** YOLO (imágenes `.jpg` + etiquetas `.txt` con `clase cx cy w h` normalizados)
**Ubicación:** `data/unified/` (no está en el repositorio y **no se puede redistribuir**, ver §2)

---

## 1. Resumen

Dataset de detección de defectos en tres superficies industriales —hormigón, acero laminado y
pavimento— construido combinando tres datasets públicos con esquemas de anotación distintos. Se usa
para entrenar el módulo supervisado (YOLOv8) de InspectAI.

| | |
|---|---|
| Imágenes | 11.536 (8.077 train, 1.153 val, 2.306 test) |
| Cajas | 28.918 |
| Clases | 15 (5 hormigón, 6 acero, 4 carretera) |
| Imágenes negativas (sin cajas) | 820 (7,1 %) |
| Tamaño en disco | 1,1 GB (lado mayor ≤ 1.280 px) |

MVTec AD, el dataset del módulo de anomalías, **no forma parte de este dataset**: se usa por separado
(ver §2 y §3).

---

## 2. Fuentes y licencias

| Dataset | Origen | Qué se usa | Licencia | Entra en `data/unified` |
|---|---|---|---|---|
| **CODEBRIM** | Zenodo, [10.5281/zenodo.2620293](https://doi.org/10.5281/zenodo.2620293), v1.0 (2019) | `CODEBRIM_original_images.zip`: 1.590 imágenes y 1.052 XML con cajas | Licencia propia (`license.md` del registro): solo investigación y educación **no comerciales**, ver más abajo | Sí |
| **NEU-DET** | Página del autor Kechen Song (Northeastern University, China) | Versión NEU-DET: 1.800 imágenes 200×200 con cajas Pascal VOC | **No se ha encontrado una licencia explícita**; la página pide citar los artículos | Sí |
| **RDD2022** | Figshare, [10.6084/m9.figshare.21431547.v1](https://doi.org/10.6084/m9.figshare.21431547.v1) (CRDDC'2022) | Subconjunto Japón: `Japan/train` (10.506 imágenes con XML) | **CC BY 4.0** | Sí |
| **MVTec AD** | mvtec.com | 4 categorías (`carpet`, `grid`, `metal_nut`, `tile`) | **CC BY-NC-SA 4.0** (solo uso no comercial) | No (módulo de anomalías) |

### Condiciones de CODEBRIM que afectan al proyecto

Resumen de los términos de `license.md` (léelos completos antes de publicar nada):

1. Uso **solo para investigación y educación no comerciales**. Si el investigador trabaja para una empresa
   con ánimo de lucro, la empresa también queda sujeta a los términos.
2. **No se puede distribuir la base de datos ni versiones modificadas** (punto 4). `data/unified/` es una
   versión modificada: no debe subirse a GitHub, Hugging Face ni ningún otro sitio. Se puede compartir
   acceso con colegas solo si aceptan los mismos términos.
3. Los trabajos derivados que **no incluyan directamente los datos** (anotaciones adicionales, modelos
   entrenados) se pueden compartir **bajo la misma licencia**, es decir, solo para uso no comercial y
   educativo (punto 5).
4. Hay que **citar** el dataset y el artículo (punto 6).

Consecuencias prácticas:

- **La demo pública (fase 5.2) no debe incluir imágenes de CODEBRIM como ejemplos precargados**, porque
  eso es distribuir el dataset. Usar imágenes de RDD2022 (CC BY 4.0, con atribución) o propias.
- Los **pesos entrenados** que se publiquen heredan una restricción no comercial; debe constar en su descripción.
- **Qué es seguro y qué es interpretación.** Una demo gratuita y sin ingresos es no comercial en el
  sentido habitual. Que enseñar un proyecto personal cuente como "educación", y que mostrarlo a una
  empresa en una candidatura no sea "obtener un beneficio comercial" (punto 3), son lecturas razonables,
  pero la licencia no define esos términos. Lo que no depende de ninguna interpretación es la
  prohibición de redistribuir el dataset (punto 4). Para tener certeza, lo más sencillo es pedir permiso
  por escrito a los autores. Esta tarjeta no es asesoramiento legal.

### NEU-DET y MVTec

- **NEU-DET:** sin licencia explícita localizada; se trata como dato de investigación y se cita a los autores.
  Conviene escribir a los autores o comprobar la página de descarga antes de redistribuir nada.
- **MVTec AD:** CC BY-NC-SA 4.0, también solo uso no comercial, y los trabajos derivados deben compartirse
  con la misma licencia.

---

## 3. Composición

### 3.1 Imágenes por dataset y split

| Dataset | Imágenes origen | Usadas | train | val | test | Negativas |
|---|---|---|---|---|---|---|
| CODEBRIM | 1.590 | 1.042 | 730 | 104 | 208 | 26 |
| NEU-DET | 1.800 | 1.800 | 1.260 | 180 | 360 | 0 |
| RDD2022 Japón | 10.506 | 8.694 | 6.087 | 869 | 1.738 | 794 |
| **Total** | 13.896 | 11.536 | 8.077 | 1.153 | 2.306 | 820 |

Negativas por split (train / val / test): 574 / 82 / 164.

Cada dataset se reparte 70 / 10 / 20 por separado, así que los tres dominios aparecen en los tres
splits en la misma proporción.

### 3.2 Distribución de cajas por clase

| id | Clase | Origen | Familia (metadato) | train | val | test | total |
|---|---|---|---|---|---|---|---|
| 0 | `concrete_crack` | CODEBRIM `Crack` | grieta | 1.702 | 221 | 576 | 2.499 |
| 1 | `concrete_spalling` | CODEBRIM `Spallation` | fragmentacion | 1.310 | 212 | 366 | 1.888 |
| 2 | `concrete_efflorescence` | CODEBRIM `Efflorescence` | contaminacion | 582 | 65 | 179 | 826 |
| 3 | `concrete_exposed_bars` | CODEBRIM `ExposedBars` | corrosion | 1.049 | 189 | 262 | 1.500 |
| 4 | `concrete_corrosion_stain` | CODEBRIM `CorrosionStain` | corrosion | 1.115 | 159 | 273 | 1.547 |
| 5 | `steel_crazing` | NEU `crazing` | grieta | 481 | 73 | 135 | 689 |
| 6 | `steel_inclusion` | NEU `inclusion` | contaminacion | 710 | 95 | 206 | 1.011 |
| 7 | `steel_patches` | NEU `patches` | contaminacion | 617 | 94 | 170 | 881 |
| 8 | `steel_pitted_surface` | NEU `pitted_surface` | fragmentacion | 308 | 47 | 77 | 432 |
| 9 | `steel_rolled_in_scale` | NEU `rolled-in_scale` | corrosion | 434 | 72 | 122 | 628 |
| 10 | `steel_scratches` | NEU `scratches` | deformacion | 375 | 58 | 115 | 548 |
| 11 | `road_longitudinal_crack` | RDD2022 `D00` | grieta | 2.827 | 409 | 813 | 4.049 |
| 12 | `road_transverse_crack` | RDD2022 `D10` | grieta | 2.800 | 367 | 812 | 3.979 |
| 13 | `road_alligator_crack` | RDD2022 `D20` | grieta | 4.345 | 632 | 1.221 | 6.198 |
| 14 | `road_pothole` | RDD2022 `D40` | fragmentacion | 1.587 | 223 | 433 | 2.243 |
| | **Total** | | | 20.242 | 2.916 | 5.760 | 28.918 |

- **Desbalance:** la clase mayor (`road_alligator_crack`, 6.198 cajas) es 14 veces la menor
  (`steel_pitted_surface`, 432).
- **Desbalance entre dominios:** RDD2022 aporta el 57 % de las cajas (16.469), CODEBRIM el 29 % (8.260)
  y NEU el 14 % (4.189).
- **Estratificación:** cada clase queda repartida con una desviación máxima de 3 puntos porcentuales
  respecto a 70 / 10 / 20.

### 3.3 Características de las imágenes

| | CODEBRIM | NEU-DET | RDD2022 Japón |
|---|---|---|---|
| Superficie | hormigón de puentes | acero laminado en caliente | asfalto, cámara en vehículo |
| Resolución original | de 1200×1600 a 6000×4000 (26 tamaños) | 200×200, 1 canal | 540 a 1080 px (97 % a 600×600) |
| Mediana del área de una caja | 1,9 % de la imagen | 11,8 % | 1,9 % |
| Anotación original | Pascal VOC + flags por daño (multietiqueta) | Pascal VOC | Pascal VOC |

A 640 px de entrada, una parte importante de las cajas queda por debajo de 32 px de lado menor:
36,6 % en `concrete_crack`, 64,5 % en `road_transverse_crack` y 44,9 % en `road_pothole`
(detalle en `notebooks/02_unificacion_datasets.ipynb`, sección 6).

---

## 4. Proceso de preparación y decisiones

Todo el proceso está en `src/data/unify.py` y es reproducible. Cada decisión con su motivo:

| Decisión | Motivo |
|---|---|
| **Una clase por tipo de daño y dataset** (15 clases, "opción A") | Agrupar por morfología empeora el desbalance (de 14 a 1 pasa a 32 a 1), deja las familias dominadas por un dominio (`grieta` es un 82 % RDD2022) y pierde fidelidad con cada origen. La familia queda como metadato y la agrupación se puede probar más adelante sin reconvertir. Argumentación con datos en el notebook 02 |
| **CODEBRIM: una caja por cada flag de daño activo** | CODEBRIM es multietiqueta y YOLO no. Se duplican las cajas con las mismas coordenadas. Alrededor del 38 % de los objetos tiene más de una clase; se pasa de 5.233 objetos a 8.260 cajas |
| **CODEBRIM: se ignora el flag `Background`** | No es un daño. 52 objetos solo tenían ese flag o ninguno y no generan caja |
| **RDD2022: solo D00, D10, D20 y D40** | Son las del `label_map.pbtxt` oficial. `D43`, `D44` y `D50` (8.284 cajas, el 33,5 %) quedan fuera; sus significados no figuran en el mapa oficial |
| **Imágenes negativas conservadas con etiqueta vacía** | Las anotadas que se quedan sin cajas (26 de CODEBRIM y 794 de RDD2022) enseñan qué no es un daño y permiten medir falsas alarmas. Son el 7,1 % del total |
| **Se descartan 538 imágenes de CODEBRIM sin XML** | Se desconoce si tienen daños; usarlas como negativas podría enseñar que un daño real es fondo |
| **Se descartan 1.812 imágenes de RDD2022 con solo clases excluidas** | Tienen objetos anotados en el origen que no entran en la taxonomía. Se puede cambiar con `rdd2022.keep_excluded_only_as_negatives` |
| **Se descartan 10 imágenes de CODEBRIM con tamaño incoherente** | El XML declara 4608×3456 y el fichero mide 6000×4000, con otra relación de aspecto: las cajas no se pueden mapear |
| **Se descarta 1 caja de RDD2022** | Menor de `min_box_px` (2 px) |
| **Reescalado a 1.280 px de lado mayor** | Las imágenes de CODEBRIM llegan a 6000×4000 y YOLO las reducirá a 640. Reducir antes ahorra disco y tiempo de carga. Las etiquetas son relativas, así que no cambian |
| **Orientación EXIF** | En CODEBRIM hay imágenes con orientación EXIF 3, 6 y 8 y sus XML usan los píxeles tal como están guardados. Al recodificar sin EXIF las copias quedan en esa orientación, que es la que vio el anotador. Verificado visualmente |
| **Reparto 70 / 10 / 20 por dataset, estratificado** | Ningún dataset aporta un test con etiquetas (las 2.627 imágenes de `RDD2022/Japan/test` no tienen anotaciones). Test es el 20 %; la validación (10 %) la necesita YOLO para elegir checkpoint. Se estratifica por la clase más rara de cada imagen, para que las minoritarias aparezcan en los tres splits |

Las 2.627 imágenes sin etiquetas de `RDD2022/Japan/test` no entran en el dataset.

---

## 5. Limitaciones y sesgos conocidos

- **Posible fuga entre splits.** El reparto es por imagen. Los datasets no indican escena u objeto
  fotografiado, y la numeración de ficheros no agrupa imágenes parecidas (la correlación entre imágenes
  consecutivas es igual que entre imágenes al azar en RDD2022: 0,31 en ambos casos). Fotos casi
  idénticas pueden caer en splits distintos, así que **las métricas de test pueden ser optimistas**.
- **Un solo país para carretera.** RDD2022 se limita a Japón (cámara en vehículo, entorno urbano y
  rural japonés). No es representativo de otros pavimentos ni condiciones.
- **Dominios muy distintos.** NEU-DET son parches de 200×200 en escala de grises con defectos grandes;
  CODEBRIM y RDD2022 tienen defectos pequeños en imágenes grandes. Un buen resultado global no implica buen
  resultado en cada dominio: hay que reportar métricas por dataset.
- **Desbalance** de clases (14 a 1) y de dominios (RDD2022 el 57 % de las cajas).
- **Cajas duplicadas en CODEBRIM:** válidas, pero compiten por las mismas posiciones de predicción y
  pueden empeorar el aprendizaje de esas clases.
- **Calidad de las anotaciones.** No se ha revisado manualmente. En NEU-DET el 2 % de las cajas cubre más
  del 90 % de la imagen. En RDD2022 hay 794 imágenes sin objetos que se asumen sin daños.
- **Familias morfológicas provisionales.** Las asignaciones de `concrete_exposed_bars`,
  `steel_rolled_in_scale` y `steel_scratches` son discutibles y `deformacion` solo contiene una clase.
- **Licencias.** Ver §2: CODEBRIM impide distribuir el dataset y sus versiones modificadas.

## 6. Usos previstos y no previstos

**Previstos:** entrenar y evaluar un detector de defectos como proyecto de investigación y aprendizaje;
comparar enfoques supervisado y no supervisado.

**No previstos:**
- Redistribuir `data/unified/` o ejemplos de CODEBRIM (ver §2).
- Uso comercial sin comprobar las licencias de CODEBRIM, NEU-DET y MVTec AD.
- Decisiones de seguridad estructural reales: el modelo no está validado para inspección profesional.

## 7. Reproducibilidad

```bash
python -m src.data.unify --dry-run     # estadísticas y reparto, sin escribir
python -m src.data.unify --sanity 4    # conversión completa + hoja de comprobación
```

Entradas: `data/raw/{codebrim,neu_surface,rdd2022}`. Configuración: `configs/taxonomy.yaml` y
`configs/training_config.yaml` (`split`, `rdd2022`, `unify`). La salida es determinista con la semilla
`unify.seed` (42). Salidas: `data/unified/{images,labels}`, `dataset.yaml`, `manifest.csv`,
`class_distribution.csv`. Detalle técnico en `doc_tecnico.md` (§4.4 y §6.2).

## 8. Citas

- **CODEBRIM:** Martin Mundt, Sagnik Majumder, Sreenivas Murali, Panagiotis Panetsos y Visvanathan Ramesh,
  "Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with
  the COncrete DEfect BRidge IMage Dataset", Conference on Computer Vision and Pattern Recognition, 2019
  (cita exigida por la licencia).
- **RDD2022:** Deeksha Arya et al., "RDD2022: A multi-national image dataset for automatic road damage
  detection", [arXiv:2209.08538](https://arxiv.org/abs/2209.08538); datos en Figshare
  ([10.6084/m9.figshare.21431547.v1](https://doi.org/10.6084/m9.figshare.21431547.v1)), CC BY 4.0
  (exige atribución).
- **NEU-DET:** K. Song y Y. Yan, Applied Surface Science, 2013, y Y. He, K. Song, Q. Meng y Y. Yan,
  IEEE Transactions on Instrumentation and Measurement, 2020 (las dos que cita la página de los autores;
  comprobar los datos bibliográficos completos en esa página antes de publicar).
- **MVTec AD:** Paul Bergmann, Michael Fauser, David Sattlegger y Carsten Steger, "A Comprehensive
  Real-World Dataset for Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and Pattern
  Recognition, 2019.
