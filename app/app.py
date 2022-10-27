import falcon

from app.detection.detection import DetectionAPI
from app.translation.translation import TranslationAPI

app = falcon.App()
app.add_route("/translation", TranslationAPI())
app.add_route("/detection", DetectionAPI())
