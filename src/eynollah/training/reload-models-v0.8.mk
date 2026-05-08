SHELL = bash -e

MODELS_SRC = models_eynollah
MODELS_DST = reloaded/models_eynollah


#	$(MODELS_DST)/eynollah-binarization_20210425 \
#	$(MODELS_DST)/eynollah-column-classifier_20210425 \
#	$(MODELS_DST)/eynollah-enhancement_20210425 \
#	$(MODELS_DST)/eynollah-main-regions-aug-rotation_20210425 \
#	$(MODELS_DST)/eynollah-main-regions-aug-scaling_20210425 \
#	$(MODELS_DST)/eynollah-main-regions-ensembled_20210425 \
#	$(MODELS_DST)/eynollah-main-regions_20220314 \
#	$(MODELS_DST)/eynollah-main-regions_20231127_672_org_ens_11_13_16_17_18 \
#	$(MODELS_DST)/eynollah-tables_20210319 \
#	$(MODELS_DST)/model_eynollah_ocr_cnnrnn_20250930 \

RELOADABLE_MODELS = \
	$(MODELS_DST)/model_eynollah_page_extraction_20250915 \
	$(MODELS_DST)/model_eynollah_reading_order_20250824 \
	$(MODELS_DST)/modelens_e_l_all_sp_0_1_2_3_4_171024 \
	$(MODELS_DST)/modelens_full_lay_1__4_3_091124 \
	$(MODELS_DST)/modelens_table_0t4_201124 \
	$(MODELS_DST)/modelens_textline_0_1__2_4_16092024

all: $(RELOADABLE_MODELS)

$(MODELS_DST)/%: $(MODELS_SRC)/%
	mkdir -p $@
	test -e $</config.json || exit 1
	eynollah-training train --force \
		with $</config.json \
		reload_weights=True \
		continue_training=False \
		dir_output=$(dir $@) \
		dir_of_start_model=$< \
	2>&1 | tee $(notdir $<).log
	cp $</config.json $@/config.json

compare: 
	for i in `find $(MODELS_DST) -mindepth 2`;do \
	 	n=$(MODELS_SRC)$${i#$(MODELS_DST)}; \
		du -bs $$n $$i ; \
	done


clear:
	rm -rf $(MODELS_DST)
