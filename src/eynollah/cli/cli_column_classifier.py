"""Stream page column counts without running the layout pipeline."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import click
import cv2
import numpy as np

from ..column_classifier import classify_pages, crop_pages
from ..utils import is_image_filename


def image_paths(inputs):
    """Visit directories without accumulating the file list."""
    for source in inputs:
        if os.path.isdir(source):
            pending = [source]
            while pending:
                with os.scandir(pending.pop()) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                        elif entry.is_file() and is_image_filename(entry.name):
                            yield entry.path
        else:
            yield source


def read_image(path):
    try:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not decode image")
        return image, None
    except (OSError, ValueError, cv2.error) as exc:
        return None, str(exc)


def records_for_batch(paths, decoded, page_model, classifier_model):
    """Run both models once per batch, retaining input order and read errors."""
    valid = [(i, image) for i, (image, _) in enumerate(decoded) if image is not None]
    records = [None] * len(paths)
    if valid:
        cropped = crop_pages([image for _, image in valid], page_model)
        scores = np.asarray(classify_pages(cropped, classifier_model))
        if scores.ndim != 2 or scores.shape[0] != len(valid) or scores.shape[1] != 6:
            raise ValueError(f"unexpected classifier output shape: {scores.shape}")
        for (i, _), row in zip(valid, scores):
            if not np.all(np.isfinite(row)):
                raise ValueError("classifier returned non-finite scores")
            records[i] = {"image": paths[i], "columns": int(np.argmax(row)) + 1,
                          "confidence": float(np.max(row))}
    for i, path in enumerate(paths):
        if records[i] is None:
            records[i] = {"image": path, "error": decoded[i][1]}
    return records


def run(inputs, model_zoo, device, batch_size, workers, output):
    """Stream bounded batches and return the number of unreadable images."""
    cv2.setNumThreads(1)
    try:
        model_zoo.load_models('page', 'col_classifier', device=device)
        page_model = model_zoo.get('page')
        classifier_model = model_zoo.get('col_classifier')
        paths = iter(image_paths(inputs))
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while batch_paths := list(islice(paths, batch_size)):
                decoded = list(executor.map(read_image, batch_paths))
                for record in records_for_batch(batch_paths, decoded, page_model, classifier_model):
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                    failures += 'error' in record
                output.flush()
        return failures
    finally:
        model_zoo.shutdown()


@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('--batch-size', default=2, type=click.IntRange(min=1), help='Images per model call.')
@click.option('--workers', default=3, type=click.IntRange(min=1), help='Concurrent image readers.')
@click.option('--output', '-o', default='-', type=click.File('w', encoding='utf-8'),
              help='JSON Lines output file; - writes to stdout.')
@click.argument('images', nargs=-1, required=True, type=click.Path(exists=True))
@click.pass_context
def column_classifier_cli(ctx, batch_size, workers, output, images):
    """Classify columns in IMAGES or recursively scanned image directories."""
    try:
        failures = run(images, ctx.obj.model_zoo, ctx.obj.device, batch_size, workers, output)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if failures:
        click.echo(f'{failures} image(s) could not be decoded', err=True)
        raise SystemExit(1)
