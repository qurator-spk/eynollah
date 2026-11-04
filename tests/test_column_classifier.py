import io
import json

import cv2
import numpy as np
from click.testing import CliRunner

from eynollah.column_classifier import (
    crop_page_from_prediction,
    crop_pages,
    prepare_classifier_input,
    prepare_page_input,
)
from eynollah.cli.cli_column_classifier import image_paths, run
from eynollah.cli import main as eynollah_cli
import eynollah.cli.cli_column_classifier as column_cli


class PageModel:
    input_shape = (None, 448, 448, 3)

    def predict(self, batch, verbose=0):
        result = np.zeros((len(batch), 448, 448, 2), dtype=np.float32)
        result[..., 0] = 1
        result[:, 70:380, 60:390, 0] = 0
        result[:, 70:380, 60:390, 1] = 1
        return result


class ClassifierModel:
    def predict(self, batch, verbose=0):
        assert batch.shape[1:] == (448, 448, 3)
        assert np.all(batch[..., 0] == batch[..., 1])
        return np.tile([0.01, 0.94, 0.02, 0.01, 0.01, 0.01], (len(batch), 1))


class ModelZoo:
    def __init__(self):
        self.models = {'page': PageModel(), 'col_classifier': ClassifierModel()}
        self.loaded = None
        self.closed = False

    def load_models(self, *names, device=''):
        self.loaded = names

    def get(self, name):
        return self.models[name]

    def shutdown(self):
        self.closed = True


def test_page_crop_and_classifier_preparation():
    image = np.full((448, 448, 3), (20, 50, 90), dtype=np.uint8)
    old_page_input = cv2.GaussianBlur(image, (5, 5), 0)
    old_page_input = cv2.resize((old_page_input / 255.).astype(np.float16),
                                (448, 448), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(prepare_page_input(image, PageModel.input_shape), old_page_input)
    page, coord = crop_pages([image], PageModel())[0]
    assert coord == (64, 386, 54, 396)  # 5x5 dilation, three iterations
    assert page.shape == (322, 342, 3)
    prepared = prepare_classifier_input(page, coord)
    assert prepared.shape == (448, 448, 3)
    assert prepared.dtype == np.float16
    assert np.all(prepared[..., 0] == prepared[..., 1])


def test_page_crop_falls_back_to_full_image():
    image = np.ones((60, 80, 3), dtype=np.uint8)
    prediction = np.zeros((448, 448, 2), dtype=np.float32)
    prediction[..., 0] = 1
    page, coord = crop_page_from_prediction(image, prediction)
    assert coord == (0, 60, 0, 80)
    np.testing.assert_array_equal(page, image)


def test_streams_results_and_read_errors(tmp_path):
    folder = tmp_path / 'images'
    folder.mkdir()
    good = folder / 'good.png'
    bad = folder / 'bad.png'
    cv2.imwrite(str(good), np.ones((448, 448, 3), dtype=np.uint8) * 255)
    bad.write_bytes(b'not an image')
    paths = list(image_paths([str(folder)]))
    assert set(paths) == {str(good), str(bad)}

    output = io.StringIO()
    zoo = ModelZoo()
    failures = run([str(good), str(bad)], zoo, 'CPU', 2, 2, output)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert failures == 1
    assert zoo.loaded == ('page', 'col_classifier')
    assert zoo.closed
    assert records[0] == {'image': str(good), 'columns': 2, 'confidence': 0.94}
    assert records[1]['image'] == str(bad)
    assert records[1]['error'] == 'could not decode image'


def test_registered_subcommand_uses_parent_model_options(tmp_path, monkeypatch):
    image = tmp_path / 'image.png'
    image.write_bytes(b'image is handled by the mocked runner')

    def fake_run(inputs, model_zoo, device, batch_size, workers, output):
        assert inputs == (str(image),)
        assert model_zoo.model_basedir == tmp_path
        assert device == 'CPU'
        assert batch_size == 2
        output.write('{"image":"test","columns":2,"confidence":0.9}\n')
        return 0

    monkeypatch.setattr(column_cli, 'run', fake_run)
    result = CliRunner().invoke(eynollah_cli, [
        '-m', str(tmp_path), '-D', 'CPU', 'column-classifier', str(image),
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {'image': 'test', 'columns': 2, 'confidence': 0.9}
