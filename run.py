import cv2
import click
import warnings
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm
from pprint import pformat
from typing import Literal, Generator, cast

warnings.simplefilter('ignore', DeprecationWarning)

from lib.base import ImageT
from lib.yolo import YoloModel
from lib.yolo_sam import SamYoloModel
from lib.sam import SamModel

@click.command(context_settings={
    'ignore_unknown_options': True, 
    'allow_extra_args': True,
})
@click.option('-i', '--input', 'input_file',
              default='input/video2.mp4',
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-o', '--output', 'output_file',
              default=None,
              type=click.Path(exists=False, dir_okay=False, writable=True))
@click.option('-m', '--model', 'model_type',
              default='yolo',
              type=click.Choice(['yolo', 'yolo-sam', 'sam']))
@click.option('--fps', 'output_fps', 
              default=0,
              type=click.IntRange(0, 300))
@click.option('-s/-ns', '--show/--no-show', 'show_image', is_flag=True, default=True)
def main(
    input_file: str,
    output_file: str,
    model_type: Literal['yolo', 'yolo-sam', 'sam'],
    output_fps: int,
    show_image: bool,
):

    print(f'Input file: {input_file}')
    input = cv2.VideoCapture(input_file)
    if not input.isOpened():
        print ("Error: Could not open video")
        exit(1)

    input_length = int(input.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = input.get(cv2.CAP_PROP_FPS)

    fourcc_code = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)   # type:ignore
    output = None
    if output_file:
        print(f'Output file: {output_file}')

    ctx = click.get_current_context()
    model_args = {ctx.args[i][2:]: ctx.args[i+1] for i in range(0, len(ctx.args), 2)}
    print(f'Model type: {model_type}\nModel args: {pformat(model_args, indent=2, compact=True)}')

    match model_type:
        case 'yolo':
            model = YoloModel(args=model_args)
        case 'yolo-sam':
            model = SamYoloModel(args=model_args)
        case 'sam':
            model = SamModel(args=model_args)

    def input_iter(as_image: bool = False) -> Generator[ImageT, None, None]:
        while True:
            ret, frame = input.read()
            if not ret:
                break
            yield  \
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if as_image \
                else cast(npt.NDArray[np.uint8], frame)


    result_frame = None
    stop = False
    for frame in tqdm(input_iter(model.NEEDS_PILLOW), total=input_length):

        anns = model.predict(frame)
        result = model.draw_all(frame, anns) #type:ignore
        match result:
            case Image.Image():
                result_frame = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            case _:
                result_frame = np.array(result)

        if show_image:
            cv2.imshow('Inference', result_frame)

            if (cv2.waitKey(1) & 0xff) in [27, ord('Q'), ord('q')]:
                stop = True
                break

        if output_file:
            if output is None:
                output = cv2.VideoWriter(
                    output_file, 
                    fourcc, 
                    float(output_fps) or input_fps, 
                    tuple(reversed(result_frame.shape[:2])))
            output.write(result_frame)

    input.release()
    if output is not None and result_frame is not None:
        for _ in range(40):
            output.write(result_frame)
        output.release()

    if show_image and not stop:
        cv2.waitKey(0)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
