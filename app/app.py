import falcon

from app.translation.translation import TranslationAPI

app = falcon.App()
app.add_route("/translation", TranslationAPI())
