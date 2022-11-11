import json
import falcon
from app.classifier.models.classifier import EmotionClassifier
from app.classifier.models.classes import mapping


def pred_sentence_emotions(model, text, topn=5):
    max_len = 35
    with torch.no_grad():

        inputs = tokenizer.encode_plus(text,
                                       None,
                                       add_special_tokens=True,
                                       max_length=max_len,
                                       padding="max_length",
                                       truncation=True)
        ids = inputs["input_ids"]
        ids = torch.LongTensor(ids).cuda().unsqueeze(0)

        attention_mask = inputs["attention_mask"]
        attention_mask = torch.LongTensor(attention_mask).cuda().unsqueeze(0)

        output = model.forward(ids, attention_mask)[0]
        output = torch.sigmoid(output)

        probas, indices = torch.sort(output)

    probas = probas.cpu().numpy()[0][::-1]
    indices = indices.cpu().numpy()[0][::-1]

    for i, p in zip(indices[:topn], probas[:topn]):
        print(mapping[i]," --> ", p)

class ClassifierAPI:
    def __init__(self) -> None:
        self.classifier = EmotionClassifier()

    def on_post(self, req, resp):
        try:
            input_sentence = req.media.get("input_sentence")
            res = pred_sentence_emotions(self.classifier.model, input_sentence)

            resp.text = json.dumps(
                {"status": 200, "data": {"emotion": res}}, ensure_ascii=False
            )
            resp.status = falcon.HTTP_200
        except Exception as e:
            print(e)
            resp.text = json.dumps(
                {"status": 400, "data": {"error": "Invalid Request"}},
                ensure_ascii=False,
            )
