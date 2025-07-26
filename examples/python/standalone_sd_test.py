import os
os.environ["ORT_LOGGING_LEVEL"] = "0"
import onnxruntime as ort
ort.set_default_logger_severity(0)
import onnxruntime_genai as og
from PIL import Image
import numpy as np

# Create configuration object
config = og.Config('C:\\Users\\asingh13\\github\\Ashutosh\\onnxruntime-genai\\examples\\python\\sd-v15-onnx-int32')

# Configure OpenVINO Execution Provider for GPU
config.clear_providers()
config.append_provider("OpenVINO")
#device_type = getattr(args, 'openvino_device', 'CPU')
#config.set_provider_option("OpenVINO", "device_type", device_type)
config.set_provider_option("OpenVINO", "device_type", "CPU_FP32")

# Create model with the configured execution provider
m = og.Model(config)

p = og.ImageGeneratorParams(m)
p.set_prompt('a dog is running in the park')

t = og.generate_image(m, p)

t_np = t.as_numpy()
#t_np = np.asarray(t)
print(type(t_np))

# print(t_np.shape)
# print(t_np.dtype)
# print(type(t_np))

#t_pil = Image.fromarray(t_np.astype(np.uint8))
t_pil = [Image.fromarray(t_np[i]) for i in range(t_np.shape[0])]

for i in range(len(t_pil)):
    t_pil[i].save(f'test_image_py_{i}.png')
    print(f"Successfully generated image(s)")
