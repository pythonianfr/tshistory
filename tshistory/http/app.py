from flask import Flask

from tshistory.http.server import httpapi


def make_app(tsa, apiclass=httpapi):
    app = Flask(__name__)
    api = apiclass(tsa)
    app.register_blueprint(
        api.bp
    )
    return app
