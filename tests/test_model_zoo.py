from eynollah.model_zoo import EynollahModelZoo

def test_trocr1(
    model_dir,
):
    model_zoo = EynollahModelZoo(model_dir)
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        model_zoo.load_model('trocr_processor')
        proc = model_zoo.get('trocr_processor', TrOCRProcessor)
        assert isinstance(proc, TrOCRProcessor)
        model_zoo.load_model('ocr', 'tr')
        model = model_zoo.get('ocr', VisionEncoderDecoderModel)
        assert isinstance(model, VisionEncoderDecoderModel)
    except ImportError:
        pass
