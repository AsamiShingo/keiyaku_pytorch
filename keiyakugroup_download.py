import transformersbert
import transformersbertcolorful
import transformersroberta
import sys

modeldata_path = r".\data\model"

model_name = "roberta"
if len(sys.argv) >= 2:
    model_name = sys.argv[1]

if model_name == "bert":
    model = transformersbert.TransformersBert()
    tokenizer = transformersbert.TransformersTokenizerBert()
    model.download_save("cl-tohoku/bert-base-japanese-v2", modeldata_path)
    tokenizer.download_save("cl-tohoku/bert-base-japanese-v2", modeldata_path)
elif model_name == "bertcolorful":
    model = transformersbertcolorful.TransformersBertColorful()
    tokenizer = transformersbertcolorful.TransformersTokenizerBertColorful()
    model.download_save("colorfulscoop/bert-base-ja", modeldata_path)
    tokenizer.download_save("colorfulscoop/bert-base-ja", modeldata_path)
elif model_name == "roberta":
    model = transformersroberta.TransformersRoberta()
    tokenizer = transformersroberta.TransformersTokenizerRoberta()
    model.download_save("rinna/japanese-roberta-base", modeldata_path)
    tokenizer.download_save("rinna/japanese-roberta-base", modeldata_path)
else:
    print("model_name error(model_name={})".format(model_name))
    sys.exit(9)
    
