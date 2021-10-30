import web.keiyakuweb
from keiyakumodelfactory import KeiyakuModelFactory
from web.keiyakuweb import init_web
import sys

if __name__ == "__main__":
    debugmode = True
    if len(sys.argv) >= 2 and sys.argv[1] == "release":
        debugmode = False

    init_web(debugmode)