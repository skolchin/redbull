import cv2
import clip
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

input = cv2.VideoCapture("input/video1.mp4")
if not input.isOpened():
    print ("Error: Could not open video.")
    exit

input_length = int(input.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc_code = 'vp09'
fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
output = None

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(sam)

model, preprocess = clip.load("ViT-B/32", device=device)

# model.predict(
#     source="input/screen.png",
#     retina_masks=True,
#     augment=True,
#     conf=0.1,
#     classes=[39,41],
#     imgsz=640,
#     # show=True,
#     save=True,
#     stream=False,
#     project='test',
#     name='test',
# )

def input_iter():
    while True:
        ret, frame = input.read()
        if not ret:
            break
        yield frame

for frame in tqdm(input_iter(), total=input_length):

    for r in model.predict(
        source=frame,
        retina_masks=False,
        augment=True,
        conf=0.12,
        classes=[39],
        imgsz=640,
        verbose=False,
    ):
        result_frame = r.plot(
            labels=False,
            conf=False,
            probs=False,
            masks=False,
        )
        cv2.putText(
            result_frame,
            f'{len(r)} bottles detected',
            (10, 20),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.8,
            color=(0,0,255),
            thickness=1,
        )
        # cv2.imshow('Inference', result_frame)
        # if (cv2.waitKey(1) & 0xff) == 27:
        #     break

        if output is None:
            output = cv2.VideoWriter("output/video1_yolo.mp4", fourcc, 20, tuple(reversed(result_frame.shape[:2])))

        output.write(result_frame)

input.release()
if output is not None:
    for _ in range(40):
        output.write(result_frame)
    output.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
