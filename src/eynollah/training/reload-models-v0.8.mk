SHELL = bash -e

MODELS_SRC = models_eynollah
MODELS_DST = reloaded/models_eynollah


# eynollah-main-regions-aug-rotation_20210425
# eynollah-main-regions-aug-scaling_20210425
# eynollah-main-regions-ensembled_20210425
# eynollah-main-regions_20220314
# eynollah-main-regions_20231127_672_org_ens_11_13_16_17_18
# eynollah-tables_20210319

CURRENT_MODELS := 
CURRENT_MODELS += model_eynollah_page_extraction_20250915
CURRENT_MODELS += model_eynollah_reading_order_20250824
CURRENT_MODELS += modelens_e_l_all_sp_0_1_2_3_4_171024
CURRENT_MODELS += modelens_full_lay_1__4_3_091124
CURRENT_MODELS += modelens_table_0t4_201124
CURRENT_MODELS += modelens_textline_0_1__2_4_16092024
CURRENT_MODELS += model_eynollah_ocr_cnnrnn_20250930
CURRENT_MODELS += eynollah-binarization_20210425
CURRENT_MODELS += eynollah-column-classifier_20210425
CURRENT_MODELS += eynollah-enhancement_20210425
CURRENT_MODELS += eynollah-main-regions_20231127_672_org_ens_11_13_16_17_18

# tf (SavedModel format) for training
# onnx conversion for fast inference
all: tf onnx

tf-serving: $(CURRENT_MODELS:%=$(MODELS_DST)/%)
tf: $(CURRENT_MODELS:%=$(MODELS_DST)/%)
keras: $(CURRENT_MODELS:%=$(MODELS_DST)/%.keras)
hdf5: $(CURRENT_MODELS:%=$(MODELS_DST)/%.h5)
onnx: $(CURRENT_MODELS:%=$(MODELS_DST)/%.onnx)

# distinguish tf from tf-serving: target pattern is the same,
# so check if either is current goal, otherwise assumg tf
$(MODELS_DST)/%: FORMAT = $(or $(filter tf-serving,$(MAKECMDGOALS)), tf)
$(MODELS_DST)/%: $(MODELS_SRC)/%
	$(if $(and $(filter tf-serving,$(FORMAT)),$(findstring _ocr,$@)),$(warning skipping $@: OCR CTC decoder fails in TF-Serving) : )\
	eynollah-training convert \
		$(and $(wildcard $</config.json),--rebuild) \
		--in $< \
		--format $(FORMAT) \
		--out $@ \
	> $(notdir $<).$(FORMAT).log 2>&1 || { cat $(notdir $<).$(FORMAT).log; false; }

$(MODELS_DST)/%.keras: $(MODELS_SRC)/%
	eynollah-training convert \
		$(and $(wildcard $</config.json),--rebuild) \
		--in $< \
		--format keras \
		--out $@ \
	> $(notdir $<).keras.log 2>&1 || { cat $(notdir $<).keras.log; false; }

$(MODELS_DST)/%.h5: $(MODELS_SRC)/%
	eynollah-training convert \
		$(and $(wildcard $</config.json),--rebuild) \
		--in $< \
		--format hdf5 \
		--out $@ \
	> $(notdir $<).hdf5.log 2>&1 || { cat $(notdir $<).hdf5.log; false; }

$(MODELS_DST)/%.onnx: $(MODELS_SRC)/%
	$(if $(findstring _ocr,$@),$(warning skipping $@: OCR CTC decoder is buggy in ONNX) : )\
	eynollah-training convert \
		$(and $(wildcard $</config.json),--rebuild) \
		--in $< \
		--format onnx \
		--out $@ \
	> $(notdir $<).onnx.log 2>&1 || { cat $(notdir $<).onnx.log; false; }

compare: 
	for i in `find $(MODELS_DST) -mindepth 2`;do \
	 	n=$(MODELS_SRC)$${i#$(MODELS_DST)}; \
		du -bs $$n $$i ; \
	done

clear:
	rm -rf $(MODELS_DST)
