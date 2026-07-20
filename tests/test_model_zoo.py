from eynollah.model_zoo import EynollahModelZoo
from eynollah.predictor import Predictor

def test_trocr1(
    model_dir,
):
    model_zoo = EynollahModelZoo(model_dir)
    try:
        model_zoo.load_models(('ocr', 'tr'))
        model = model_zoo.get('ocr')
        assert isinstance(model, Predictor)
        shape = model.input_shape
        assert len(shape) == 3
    except ImportError:
        pass

def test_cnnrnnocr1(
    model_dir,
):
    model_zoo = EynollahModelZoo(model_dir)
    try:
        model_zoo.load_models('ocr')
        model = model_zoo.get('ocr')
        assert isinstance(model, Predictor)
        shape = model.input_shape
        assert len(shape) == 2
        assert len(shape[0]) == 4
    except ImportError:
        pass
