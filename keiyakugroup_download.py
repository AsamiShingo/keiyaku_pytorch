from transformersfactory import TransformersFactory
import sys

model_name = ""
if len(sys.argv) >= 2:
    model_name = sys.argv[1]

TransformersFactory.get_transfomers(model_name, True)
    
