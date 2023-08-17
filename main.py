import sys
from interface.app import App

if __name__ == '__main__':
    app = App(sys.argv)
    app.run()