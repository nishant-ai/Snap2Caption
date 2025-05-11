import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import triton_python_backend_utils as pb_utils
import base64, io
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.model = LlavaForConditionalGeneration.from_pretrained("/models/LLAVA_MERGED").cuda().eval()
        self.processor = AutoProcessor.from_pretrained("/models/LLAVA_MERGED")

    def execute(self, requests):
        responses = []
        for request in requests:
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()

            img_b64 = image_tensor[0][0].decode("utf-8")
            prompt_text = prompt_tensor[0][0].decode("utf-8")

            image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image"}]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.processor(text=prompt, images=[image], return_tensors="pt", padding=True).to(self.model.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=80)
            caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            out_tensor = pb_utils.Tensor("caption", np.array([caption.encode("utf-8")], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
