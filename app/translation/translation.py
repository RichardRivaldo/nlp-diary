import json
from fnmatch import translate

import falcon
from app.translation.models.translation import Translation


class TranslationAPI:
    def __init__(self) -> None:
        self.translation = Translation()

    def on_post(self, req, resp):
        try:
            input_sentence = req.media.get("input_sentence")
            translated = self.translation.translate(input_sentence)

            resp.text = json.dumps(
                {"status": 200, "data": {"translated": translated}}, ensure_ascii=False
            )
            resp.status = falcon.HTTP_200
        except Exception as e:
            print(e)
            resp.text = json.dumps(
                {"status": 400, "data": {"error": "Invalid Request"}},
                ensure_ascii=False,
            )
