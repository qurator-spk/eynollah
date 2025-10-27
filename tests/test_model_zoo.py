from pathlib import Path

from eynollah.model_zoo import EynollahModelZoo, TrOCRProcessor, VisionEncoderDecoderModel

testdir = Path(__file__).parent.resolve()
MODELS_DIR = testdir.parent

def test_trocr1():
    model_zoo = EynollahModelZoo(str(MODELS_DIR))
    model_zoo.load_model('trocr_processor')
    proc = model_zoo.get('trocr_processor', TrOCRProcessor)
    assert isinstance(proc, TrOCRProcessor)

    model_zoo.load_model('ocr', 'tr')
    model = model_zoo.get('ocr')
    assert isinstance(model, VisionEncoderDecoderModel)
    print(proc)

test_trocr1()
