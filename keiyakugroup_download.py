from keiyakumodelfactory import KeiyakuModelFactory
import sys

model_name = ""
if len(sys.argv) >= 2:
    model_name = sys.argv[1]

KeiyakuModelFactory.get_transfomers(model_name, True)
    
