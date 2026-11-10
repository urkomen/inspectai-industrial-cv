"""Pipeline de unificacion de los tres datasets a formato YOLO.

Lee CODEBRIM, NEU-DET y RDD2022 de data/raw (anotaciones Pascal VOC), aplica la taxonomia de
configs/taxonomy.yaml y escribe en data/unified imagenes + etiquetas YOLO divididas en
train/val/test, junto con dataset.yaml, manifest.csv y class_distribution.csv.

Uso (desde la raiz del repo):
    python -m src.data.unify --dry-run       # solo estadisticas y reparto, no escribe nada
    python -m src.data.unify --sanity 4      # conversion completa + hoja de comprobacion
    python -m src.data.unify --limit 20 --out <carpeta>   # prueba rapida fuera de data/unified
"""
import argparse
import csv
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps

from src.data.taxonomy import load_taxonomy
from src.utils.paths import load_paths, load_training_config

SPLITS = ("train", "val", "test")
PREFIX = {"codebrim": "cb", "neu_surface": "neu", "rdd2022": "rdd"}
EXIF_ORIENTATION = 274
ROTATED = (5, 6, 7, 8)


@dataclass
class Sample:
    dataset: str
    stem: str
    image: Path
    width: int                # tamano segun el XML, en el que estan expresadas las cajas
    height: int
    transpose: bool           # el XML usa la imagen ya rotada segun EXIF
    orientation: int          # etiqueta EXIF de orientacion (1 = sin rotar)
    boxes: list = field(default_factory=list)   # (class_id, xmin, ymin, xmax, ymax) en px

    @property
    def name(self):
        return f"{PREFIX[self.dataset]}_{self.stem}"


# --- Lectura de anotaciones ---------------------------------------------------------------

def read_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    return root, int(size.findtext("width")), int(size.findtext("height"))


def voc_box(obj):
    bb = obj.find("bndbox")
    return tuple(int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))


def check_orientation(image_path, w, h):
    """(transpose, orientation) o None si el tamano del XML no cuadra con la imagen.

    transpose=False: el XML usa los pixeles tal como estan guardados. transpose=True: el XML
    usa la imagen ya rotada segun EXIF (ancho y alto intercambiados respecto al fichero).
    """
    with Image.open(image_path) as im:
        size, orientation = im.size, im.getexif().get(EXIF_ORIENTATION, 1)
    if size == (w, h):
        return False, orientation
    if size == (h, w) and orientation in ROTATED:
        return True, orientation
    return None


def build_sample(dataset, stem, image, w, h, boxes, st, ucfg, allow_empty):
    """Valida las cajas y crea el Sample, o None si la imagen se descarta (y lo anota en st)."""
    clean = []
    for cid, x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < ucfg["min_box_px"] or y1 - y0 < ucfg["min_box_px"]:
            st["cajas_descartadas_tamano"] += 1
            continue
        clean.append((cid, x0, y0, x1, y1))
    if not clean and not (allow_empty and ucfg["keep_negatives"]):
        st["imagenes_descartadas_sin_cajas"] += 1
        return None
    orient = check_orientation(image, w, h)
    if orient is None:
        st["imagenes_descartadas_tamano_incoherente"] += 1
        return None
    st["imagenes_ok"] += 1
    st["imagenes_negativas"] += not clean
    st["cajas_ok"] += len(clean)
    return Sample(dataset, stem, image, w, h, orient[0], orient[1], clean)


# --- Un recolector por dataset --------------------------------------------------------------

def collect_codebrim(raw, tax, ucfg, st):
    """Una caja YOLO por cada flag de dano activo (CODEBRIM es multietiqueta)."""
    images = {p.stem: p for p in sorted((raw / "images").glob("*.jpg"))}
    classes = tax.for_dataset("codebrim")
    st["imagenes_origen"] = len(images)
    annotated = set()
    for xml in sorted((raw / "annotations").glob("*.xml")):
        image = images.get(xml.stem)
        if image is None:
            st["xml_sin_imagen"] += 1
            continue
        annotated.add(xml.stem)
        root, w, h = read_voc(xml)
        boxes = []
        for obj in root.findall("object"):
            st["objetos_origen"] += 1
            defect, box = obj.find("Defect"), voc_box(obj)
            active = [c for c in classes
                      if defect is not None and defect.findtext(c.source_label, "0").strip() == "1"]
            if not active:
                st["objetos_sin_clase_de_dano"] += 1   # solo Background o sin ningun flag
            boxes.extend((c.id, *box) for c in active)
        sample = build_sample("codebrim", xml.stem, image, w, h, boxes, st, ucfg, allow_empty=True)
        if sample:
            yield sample
    st["imagenes_sin_xml_descartadas"] = len(images) - len(annotated)


def collect_neu(raw, tax, ucfg, st):
    images = {p.stem: p for p in sorted((raw / "IMAGES").glob("*.jpg"))}
    st["imagenes_origen"] = len(images)
    for xml in sorted((raw / "ANNOTATIONS").glob("*.xml")):
        image = images.get(xml.stem)
        if image is None:
            st["xml_sin_imagen"] += 1
            continue
        root, w, h = read_voc(xml)
        boxes = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            c = tax.lookup("neu_surface", name)
            if c is None:
                raise ValueError(f"Etiqueta NEU sin mapear en la taxonomia: {name!r} ({xml.name})")
            boxes.append((c.id, *voc_box(obj)))
        sample = build_sample("neu_surface", xml.stem, image, w, h, boxes, st, ucfg, allow_empty=False)
        if sample:
            yield sample


def collect_rdd(raw, tax, cfg, st):
    """Solo las clases de cfg['rdd2022']['classes']; el resto de cajas se descartan."""
    rcfg, ucfg = cfg["rdd2022"], cfg["unify"]
    used = set(rcfg["classes"])
    for country in rcfg["countries"]:
        base = raw / country / "train"
        images = {p.stem: p for p in sorted((base / "images").glob("*.jpg"))}
        st["imagenes_origen"] += len(images)
        for xml in sorted((base / "annotations" / "xmls").glob("*.xml")):
            image = images.get(xml.stem)
            if image is None:
                st["xml_sin_imagen"] += 1
                continue
            root, w, h = read_voc(xml)
            objects = root.findall("object")
            boxes = []
            for obj in objects:
                name = obj.findtext("name")
                if name not in used:
                    st["cajas_clase_excluida"] += 1
                    continue
                c = tax.lookup("rdd2022", name)
                if c is None:
                    raise ValueError(f"Clase RDD2022 sin mapear en la taxonomia: {name!r} ({xml.name})")
                boxes.append((c.id, *voc_box(obj)))
            only_excluded = bool(objects) and not boxes
            st["imagenes_solo_clase_excluida"] += only_excluded
            allow_empty = not objects or (only_excluded and rcfg["keep_excluded_only_as_negatives"])
            sample = build_sample("rdd2022", xml.stem, image, w, h, boxes, st, ucfg, allow_empty)
            if sample:
                yield sample


def collect_all(paths, tax, cfg, limit):
    ucfg = cfg["unify"]
    stats = defaultdict(Counter)
    collectors = {
        "codebrim": lambda: collect_codebrim(paths["raw"]["codebrim"], tax, ucfg, stats["codebrim"]),
        "neu_surface": lambda: collect_neu(paths["raw"]["neu_surface"], tax, ucfg, stats["neu_surface"]),
        "rdd2022": lambda: collect_rdd(paths["raw"]["rdd2022"], tax, cfg, stats["rdd2022"]),
    }
    samples = []
    for dataset, collect in collectors.items():
        found = list(collect())
        samples += found[:limit] if limit else found
    return samples, stats


# --- Reparto train/val/test -----------------------------------------------------------------

def split_samples(samples, ratios, seed):
    """Reparto por dataset estratificado por la clase mas rara de cada imagen.

    Cada grupo (dataset, clase mas rara; las imagenes sin cajas forman su propio grupo) se
    baraja con una semilla fija y se divide en las proporciones de `ratios`. Asi cada split
    contiene todos los dominios y las clases minoritarias aparecen en los tres. No hay forma
    de agrupar por escena u objeto fotografiado (los datasets no lo indican), asi que
    imagenes casi identicas pueden caer en splits distintos.
    """
    freq = Counter(b[0] for s in samples for b in s.boxes)
    groups = defaultdict(list)
    for s in samples:
        rarest = min((b[0] for b in s.boxes), key=lambda c: (freq[c], c)) if s.boxes else -1
        groups[(s.dataset, rarest)].append(s)
    rng = random.Random(seed)
    result = {split: [] for split in SPLITS}
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda s: s.stem)
        rng.shuffle(items)
        n_train = round(len(items) * ratios["train"])
        n_val = min(round(len(items) * ratios["val"]), len(items) - n_train)
        result["train"] += items[:n_train]
        result["val"] += items[n_train:n_train + n_val]
        result["test"] += items[n_train + n_val:]
    return result


def class_distribution(split_map, tax):
    """Cajas por (clase, split) y total, en el orden de ids."""
    rows = []
    for c in tax.classes:
        counts = {sp: sum(1 for s in split_map[sp] for b in s.boxes if b[0] == c.id) for sp in SPLITS}
        rows.append({"id": c.id, "name": c.name, "dataset": c.dataset, "family": c.family,
                     **counts, "total": sum(counts.values())})
    return rows


# --- Escritura --------------------------------------------------------------------------------

def out_size(w, h, max_side):
    if not max_side or max(w, h) <= max_side:
        return w, h
    scale = max_side / max(w, h)
    return max(1, round(w * scale)), max(1, round(h * scale))


def write_image(sample, dest, max_side, quality):
    tw, th = out_size(sample.width, sample.height, max_side)
    resize = (tw, th) != (sample.width, sample.height)
    if not resize and not sample.transpose and sample.orientation == 1:
        shutil.copyfile(sample.image, dest)       # sin cambios: copia byte a byte
        return
    # PIL no copia el EXIF al guardar: la copia queda con los pixeles en la orientacion del XML.
    with Image.open(sample.image) as im:
        if resize and not sample.transpose:
            im.draft("RGB", (tw, th))             # decodificacion JPEG ya reducida
        if sample.transpose:
            im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if resize:
            im = im.resize((tw, th), Image.LANCZOS)
        im.save(dest, "JPEG", quality=quality)


def write_label(sample, dest):
    w, h = sample.width, sample.height
    lines = [f"{cid} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
             f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}" for cid, x0, y0, x1, y1 in sample.boxes]
    dest.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def write_dataset(split_map, images_dir, labels_dir, ucfg, workers):
    jobs = []
    for split, items in split_map.items():
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        jobs += [(s, split) for s in items]

    def run(job):
        s, split = job
        write_image(s, images_dir / split / f"{s.name}.jpg", ucfg["max_side"], ucfg["jpeg_quality"])
        write_label(s, labels_dir / split / f"{s.name}.txt")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(run, jobs), 1):
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} imagenes escritas", flush=True)


def write_metadata(root, split_map, dist, tax, ucfg):
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "dataset", "split", "n_boxes", "orig_w", "orig_h", "out_w", "out_h"])
        for split, items in split_map.items():
            for s in items:
                ow, oh = out_size(s.width, s.height, ucfg["max_side"])
                w.writerow([s.name, s.dataset, split, len(s.boxes), s.width, s.height, ow, oh])
    with open(root / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dist[0]))
        w.writeheader()
        w.writerows(dist)
    # Sin clave "path": ultralytics toma la carpeta del yaml, asi el dataset es portable entre PCs.
    spec = {sp: f"images/{sp}" for sp in SPLITS}
    spec["names"] = dict(enumerate(tax.names))
    (root / "dataset.yaml").write_text(
        "# Generado por src/data/unify.py\n" + yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def sanity_sheet(root, images_dir, labels_dir, split_map, tax, per_dataset, seed, thumb=420):
    """Hoja PNG con cajas dibujadas sobre las imagenes ya escritas (una fila por dataset)."""
    rng = random.Random(seed)
    sheet = Image.new("RGB", (per_dataset * thumb, len(PREFIX) * thumb), "white")
    for row, dataset in enumerate(PREFIX):
        pool = [(sp, s) for sp, items in split_map.items() for s in items
                if s.dataset == dataset and s.boxes]
        for col, (split, s) in enumerate(rng.sample(pool, min(per_dataset, len(pool)))):
            im = Image.open(images_dir / split / f"{s.name}.jpg").convert("RGB")
            im.thumbnail((thumb, thumb))
            draw, (W, H) = ImageDraw.Draw(im), im.size
            for line in (labels_dir / split / f"{s.name}.txt").read_text().splitlines():
                cid, cx, cy, bw, bh = line.split()
                box = [(float(cx) - float(bw) / 2) * W, (float(cy) - float(bh) / 2) * H,
                       (float(cx) + float(bw) / 2) * W, (float(cy) + float(bh) / 2) * H]
                draw.rectangle(box, outline="red", width=2)
                draw.text((box[0] + 3, box[1] + 3), tax.names[int(cid)], fill="yellow")
            sheet.paste(im, (col * thumb, row * thumb))
    path = root / "sanity_check.png"
    sheet.save(path)
    return path


# --- Informes ----------------------------------------------------------------------------------

def print_stats(stats):
    for dataset, st in stats.items():
        print(f"\n[{dataset}]")
        for key, value in st.items():
            print(f"  {key:42s} {value:>7d}")


def print_split(split_map, dist):
    print("\nImagenes por split y dataset:")
    for dataset in PREFIX:
        counts = [sum(1 for s in split_map[sp] if s.dataset == dataset) for sp in SPLITS]
        print(f"  {dataset:12s} " + "  ".join(f"{sp}={n:5d}" for sp, n in zip(SPLITS, counts)))
    neg = [sum(1 for s in split_map[sp] if not s.boxes) for sp in SPLITS]
    print("  negativas    " + "  ".join(f"{sp}={n:5d}" for sp, n in zip(SPLITS, neg)))
    print("\nCajas por clase y split:")
    print(f"  {'id':>2s} {'clase':26s} {'train':>6s} {'val':>6s} {'test':>6s} {'total':>6s}")
    for r in dist:
        print(f"  {r['id']:>2d} {r['name']:26s} {r['train']:>6d} {r['val']:>6d} {r['test']:>6d} {r['total']:>6d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="solo estadisticas y reparto; no escribe")
    ap.add_argument("--limit", type=int, default=0, help="maximo de imagenes por dataset (pruebas)")
    ap.add_argument("--out", type=Path, help="carpeta de salida (por defecto, unified.root de paths.yaml)")
    ap.add_argument("--max-side", type=int, help="sobrescribe unify.max_side del config")
    ap.add_argument("--sanity", type=int, default=0, metavar="N", help="hoja de comprobacion con N imagenes por dataset")
    ap.add_argument("--overwrite", action="store_true", help="permitir escribir si ya hay imagenes en la salida")
    args = ap.parse_args(argv)

    paths, cfg, tax = load_paths(), load_training_config(), load_taxonomy()
    if args.max_side is not None:
        cfg["unify"]["max_side"] = args.max_side
    ucfg = cfg["unify"]

    samples, stats = collect_all(paths, tax, cfg, args.limit)
    print_stats(stats)
    split_map = split_samples(samples, cfg["split"], ucfg["seed"])
    dist = class_distribution(split_map, tax)
    print_split(split_map, dist)
    if args.dry_run:
        print("\n--dry-run: no se ha escrito nada")
        return

    root = args.out or paths["unified"]["root"]
    images_dir = args.out / "images" if args.out else paths["unified"]["images"]
    labels_dir = args.out / "labels" if args.out else paths["unified"]["labels"]
    if images_dir.exists() and any(images_dir.rglob("*.jpg")) and not args.overwrite:
        sys.exit(f"{images_dir} ya contiene imagenes; usa --overwrite para sobrescribir los ficheros con el mismo nombre")

    root.mkdir(parents=True, exist_ok=True)
    print(f"\nEscribiendo en {root}")
    write_dataset(split_map, images_dir, labels_dir, ucfg, ucfg["workers"])
    write_metadata(root, split_map, dist, tax, ucfg)
    if args.sanity:
        print("Hoja de comprobacion:", sanity_sheet(root, images_dir, labels_dir, split_map, tax, args.sanity, ucfg["seed"]))
    print("Hecho.")


if __name__ == "__main__":
    main()
