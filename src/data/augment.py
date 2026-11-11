"""Data augmentation especifica por dominio y sobremuestreo offline de clases minoritarias.

Genera copias aumentadas de imagenes de train que contienen clases con pocas cajas, hasta que cada
clase llega a augment.target_boxes_per_class. Cada dominio usa su propio pipeline de albumentations:
  - codebrim: luz, sombras y pequenos cambios de perspectiva (fotos de hormigon a distinta hora y angulo)
  - neu_surface: giros de 90 grados, flips y ruido de sensor (parches de acero en gris)
  - rdd2022: luz, llovizna simulada, perspectiva y desenfoque de movimiento (camara en un vehiculo)

Solo se aumentan imagenes de train: val y test no se tocan, asi no hay copias de una misma imagen
repartidas entre splits. Las copias van a images/train_aug y labels/train_aug, y dataset.yaml pasa a
listar las dos carpetas de train.

Uso (desde la raiz del repo):
    python -m src.data.augment --dry-run          # plan estimado, no escribe nada
    python -m src.data.augment                    # genera las copias en data/unified
    python -m src.data.augment --out <carpeta>    # genera solo las copias, en otra carpeta
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml

from src.data.taxonomy import load_taxonomy
from src.utils.paths import load_paths, load_training_config

PREFIX_DOMAIN = {"cb": "codebrim", "neu": "neu_surface", "rdd": "rdd2022"}
AUG_SPLIT = "train_aug"


@dataclass
class Item:
    name: str            # nombre de salida sin extension, p. ej. rdd_Japan_000007
    domain: str
    boxes: list          # (class_id, cx, cy, w, h) normalizados


def build_pipeline(domain, min_visibility, seed):
    """Pipeline de albumentations (cajas YOLO) del dominio indicado."""
    if domain == "codebrim":
        transforms = [
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            A.RandomGamma(gamma_limit=(80, 125), p=0.4),
            A.RandomShadow(shadow_intensity_range=(0.1, 0.3), p=0.2),   # sombra suave: la por defecto (0.5) es un triangulo muy marcado
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.Perspective(scale=(0.02, 0.05), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.HorizontalFlip(p=0.5),
        ]
    elif domain == "neu_surface":
        transforms = [
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(std_range=(0.01, 0.04), per_channel=False, p=0.6),   # sensor monocromo: mismo ruido en los 3 canales
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ]
    elif domain == "rdd2022":
        transforms = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomRain(slant_range=(-8, 8), drop_length=8, blur_value=3, brightness_coefficient=0.9,
                         rain_type="drizzle", p=0.3),   # llovizna: la lluvia por defecto dibuja rayas que imitan grietas
            A.RandomShadow(shadow_intensity_range=(0.1, 0.3), p=0.15),
            A.Perspective(scale=(0.02, 0.05), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.MotionBlur(blur_limit=5, p=0.2),
        ]
    else:
        raise ValueError(f"Dominio sin pipeline de augmentation: {domain!r}")
    bbox = A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=min_visibility)
    return A.Compose(transforms, bbox_params=bbox, seed=seed)


def read_label(path):
    boxes = []
    for line in path.read_text().splitlines():
        c, cx, cy, w, h = line.split()
        boxes.append((int(c), float(cx), float(cy), float(w), float(h)))
    return boxes


def load_train(root):
    """Items de train con al menos una caja (las negativas no aportan a ninguna clase)."""
    items = []
    for label in sorted((root / "labels" / "train").glob("*.txt")):
        boxes = read_label(label)
        if boxes:
            items.append(Item(label.stem, PREFIX_DOMAIN[label.stem.split("_")[0]], boxes))
    return items


def augment_item(item, image, pipeline):
    """(imagen, cajas) aumentadas, o None si la transformacion elimina todas las cajas."""
    out = pipeline(image=image, bboxes=[b[1:] for b in item.boxes], class_ids=[b[0] for b in item.boxes])
    if not out["bboxes"]:
        return None
    boxes = [(int(c), *(min(1.0, max(0.0, v)) for v in bb)) for c, bb in zip(out["class_ids"], out["bboxes"])]
    return out["image"], boxes


def oversample(items, n_classes, acfg, make_copy):
    """Elige copias hasta que cada clase llega al objetivo y devuelve (lista, cajas finales por clase).

    make_copy(item, k) crea la copia k de un item y devuelve sus cajas, o None si se descarta. Se procesan
    las clases de menos a mas cajas; entre las imagenes candidatas se prefiere la que menos copias lleva.
    """
    rng = random.Random(acfg["seed"])
    counts = Counter(b[0] for it in items for b in it.boxes)
    by_class = defaultdict(list)
    for it in items:
        for c in {b[0] for b in it.boxes}:
            by_class[c].append(it)
    copies, made = Counter(), []
    below = sorted((c for c in range(n_classes) if counts[c] < acfg["target_boxes_per_class"]),
                   key=lambda c: counts[c])
    for c in below:
        while counts[c] < acfg["target_boxes_per_class"]:
            pool = [it for it in by_class[c] if copies[it.name] < acfg["max_copies_per_image"]]
            if not pool:
                break
            fewest = min(copies[it.name] for it in pool)
            item = rng.choice([it for it in pool if copies[it.name] == fewest])
            copies[item.name] += 1
            boxes = make_copy(item, copies[item.name])
            if boxes is None:
                continue
            made.append((item, copies[item.name], boxes))
            for b in boxes:
                counts[b[0]] += 1
    return made, counts


def write_copy(root_out, item, k, image, boxes, quality):
    name = f"{item.name}_aug{k}"
    (root_out / "images" / AUG_SPLIT).mkdir(parents=True, exist_ok=True)
    (root_out / "labels" / AUG_SPLIT).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root_out / "images" / AUG_SPLIT / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes]
    (root_out / "labels" / AUG_SPLIT / f"{name}.txt").write_text("".join(lines), encoding="utf-8")
    return name


def draw(image, boxes, names, width=380):
    scale = width / image.shape[1]
    im = cv2.resize(image, (width, round(image.shape[0] * scale)))
    h, w = im.shape[:2]
    for c, cx, cy, bw, bh in boxes:
        p1, p2 = (round((cx - bw / 2) * w), round((cy - bh / 2) * h)), (round((cx + bw / 2) * w), round((cy + bh / 2) * h))
        cv2.rectangle(im, p1, p2, (0, 0, 255), 2)
        cv2.putText(im, names[c], (p1[0] + 2, p1[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return im


def check_sheet(root_src, root_out, made, names, tiles_per_domain=2, width=380):
    """Hoja PNG: por dominio, pares original / aumentada con sus cajas."""
    rows = []
    for domain in PREFIX_DOMAIN.values():
        pairs = []
        candidates = [m for m in made if m[0].domain == domain]
        step = max(1, len(candidates) // tiles_per_domain)
        for item, k, boxes in candidates[::step][:tiles_per_domain]:
            orig = cv2.imread(str(root_src / "images" / "train" / f"{item.name}.jpg"))
            aug = cv2.imread(str(root_out / "images" / AUG_SPLIT / f"{item.name}_aug{k}.jpg"))
            for im, bx in ((orig, item.boxes), (aug, boxes)):
                tile = draw(im, bx, names, width)
                canvas = np.full((width, width, 3), 255, np.uint8)
                canvas[:min(width, tile.shape[0]), :] = tile[:width]
                pairs.append(canvas)
        if pairs:
            rows.append(np.hstack(pairs))
    path = root_out / "augment_check.png"
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def update_dataset_yaml(root):
    path = root / "dataset.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec["train"] = ["images/train", f"images/{AUG_SPLIT}"]
    path.write_text("# Generado por src/data/unify.py; train ampliado por src/data/augment.py\n"
                    + yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def print_report(names, before, after, made):
    print(f"\n{'id':>2s} {'clase':26s} {'antes':>6s} {'despues':>8s}")
    for c, name in enumerate(names):
        mark = "  <- sobremuestreada" if after[c] != before[c] else ""
        print(f"{c:>2d} {name:26s} {before[c]:>6d} {after[c]:>8d}{mark}")
    per_domain = Counter(item.domain for item, _, _ in made)
    print("\nCopias a generar por dominio:", dict(per_domain), "| total:", len(made))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="plan estimado (sin aplicar transformaciones); no escribe")
    ap.add_argument("--out", type=Path, help="carpeta donde escribir las copias (por defecto, la misma de origen)")
    ap.add_argument("--overwrite", action="store_true", help="permitir escribir si train_aug ya tiene imagenes")
    args = ap.parse_args(argv)

    paths, cfg, tax = load_paths(), load_training_config(), load_taxonomy()
    acfg = cfg["augment"]
    src = paths["unified"]["root"]
    out = args.out or src
    items = load_train(src)
    names = tax.names
    before = Counter(b[0] for it in items for b in it.boxes)
    print(f"{len(items)} imagenes de train con cajas en {src}")

    if args.dry_run:
        made, after = oversample(items, len(names), acfg, lambda item, k: [b for b in item.boxes])
        print_report(names, before, after, made)
        print("\n--dry-run: estimacion suponiendo que no se pierde ninguna caja; no se ha escrito nada")
        return

    aug_dir = out / "images" / AUG_SPLIT
    if aug_dir.exists() and any(aug_dir.glob("*.jpg")) and not args.overwrite:
        sys.exit(f"{aug_dir} ya contiene imagenes; usa --overwrite para sobrescribir los ficheros con el mismo nombre")

    pipelines = {d: build_pipeline(d, acfg["min_visibility"], acfg["seed"]) for d in PREFIX_DOMAIN.values()}

    def make_copy(item, k):
        image = cv2.imread(str(src / "images" / "train" / f"{item.name}.jpg"))
        result = augment_item(item, image, pipelines[item.domain])
        if result is None:
            return None
        write_copy(out, item, k, result[0], result[1], acfg["jpeg_quality"])
        return result[1]

    made, after = oversample(items, len(names), acfg, make_copy)

    with open(out / "manifest_aug.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "source", "dataset", "n_boxes"])
        for item, k, boxes in made:
            w.writerow([f"{item.name}_aug{k}", item.name, item.domain, len(boxes)])
    if out == src:
        update_dataset_yaml(out)
    else:
        print("Aviso: con --out distinto del origen no se modifica dataset.yaml")
    print_report(names, before, after, made)
    print("Hoja de comprobacion:", check_sheet(src, out, made, names))
    print("Hecho.")


if __name__ == "__main__":
    main()
