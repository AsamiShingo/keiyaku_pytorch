import web.keiyakuweb
from keiyakumodelfactory import KeiyakuModelFactory

if __name__ == "__main__":
    KeiyakuModelFactory.get_keiyakumodel()
    web.keiyakuweb.app.run(debug=True)