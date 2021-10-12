import transformersroberta

modeldata_path = r".\data\model"

model = transformersroberta.TransformersRoberta()
tokenizer = transformersroberta.TransformersTokenizerRoberta()

model.download_save("rinna/japanese-roberta-base", modeldata_path)
tokenizer.download_save("rinna/japanese-roberta-base", modeldata_path)
