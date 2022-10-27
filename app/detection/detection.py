import json
import falcon
from app.detection.models.detection import Detection


class DetectionAPI:
    def __init__(self) -> None:
        self.detection = Detection()

    def on_post(self, req, resp):
        try:
            input_sentence = req.media.get("input_sentence")
            detected = self.detection.detect(input_sentence)

            resp.text = json.dumps(
                {"status": 200, "data": {"detected": detected}}, ensure_ascii=False
            )
            resp.status = falcon.HTTP_200
        except Exception as e:
            print(e)
            resp.text = json.dumps(
                {"status": 400, "data": {"error": "Invalid Request"}},
                ensure_ascii=False,
            )
