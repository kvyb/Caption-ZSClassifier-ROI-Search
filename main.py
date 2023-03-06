import json
import requests
import base64
import argparse
from label import *

def get_caption(reference):

    # API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
    # headers = {"Authorization": f"Bearer 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"}
    # with open(reference, "rb") as f:
    #     data = base64.b64encode(f.read()).decode("utf-8")
    # body = {"image":data}
    # # body["parameters"] = {"min_length": 30, "num_beams": 4}
    # response = requests.request("POST", API_URL, headers=headers, data=json.dumps(body))
    # return json.loads(response.content.decode("utf-8"))
    image_label = get_label([reference])
    return image_label

def make_class_names(caption):
    classes = []
    # caption = caption[0]['generated_text']
    caption = caption[0]
    classes.append(caption)
    classes.append("not " + caption)
    return classes

def inference(subject, classes):
    API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"
    headers = {"Authorization": "Bearer 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"}
    classlist = ",".join(classes[:2])
    with open(subject, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    body = {"image":data}
    body["parameters"] = {"candidate_labels": classlist}
    response = requests.post(API_URL, headers=headers, data=json.dumps(body))
    return json.loads(response.content.decode("utf-8"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example program")

    # Add arguments to the parser
    parser.add_argument("--reference", help="Referenceimage", required=True)
    parser.add_argument("--subject", help="Subjectimage", required=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the function with the keyword arguments
    caption = get_caption(args.reference)
    classes = make_class_names(caption)
    inference = inference(args.subject, classes)
    print(inference)