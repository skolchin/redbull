# Pythonic wrapper around opencv's VideoCapture().
# Source: https://github.com/postpop/videoreader, modified by kol

import sys
import cv2
import filetype
import numpy as np
from enum import StrEnum
from pathlib import Path
from datetime import timedelta
from typing import Tuple, Sequence

class MediaType(StrEnum):
    """ Media type """

    VIDEO = 'video'
    """ Videofile """

    DEVICE = 'device'
    """ Livecam device """

    IMAGE = 'image'
    """ Single image file """

    DIRECTORY = 'directory'
    """ Directory with image files """

class VideoReader:
    """ Pythonic wrapper around opencv's VideoCapture() """

    _filename: str | None
    _device: int | None
    _media_type: MediaType
    _skip: int
    _flags: int
    _frame_channels: int
    _vr: cv2.VideoCapture

    def __init__(self, filename: str | int | Path, skip: int = 0, flags: int = 0):
        """Open image, video file or stream """

        self._skip = skip or 0
        self._flags = flags or 0

        match filename:
            case str() | Path() if Path(filename).is_dir():
                # directory (probably with image files)
                self._filename = str(filename)
                self._device = None
                self._media_type = MediaType.DIRECTORY

            case str() | Path() if (filetype.guess_mime(filename) or '').startswith('image'):
                # image file
                self._filename = str(filename)
                self._device = None
                self._media_type = MediaType.IMAGE

                self._vr = PictureVideoCapture()
                if not self._vr.open(self._filename, self._flags):
                    raise IOError(f'Cannot open file {self._filename}')

            case str() | Path() if (filetype.guess_mime(filename) or '').startswith('video'):
                # video file
                self._filename = str(filename)
                self._device = None
                self._media_type = MediaType.VIDEO

                self._vr = cv2.VideoCapture()
                if not self._vr.open(self._filename, self._flags):
                    raise IOError(f'Cannot open file {self._filename}')

            case int():
                # livecam device
                self._filename = None
                self._device = int(filename)
                self._skip = 0
                self._media_type = MediaType.DEVICE

                self._vr = cv2.VideoCapture()
                if not self._vr.open(self._device, self._flags):
                    raise IOError(f'Cannot open device {self._device}')

            case _:
                raise ValueError(f'Unknown file type {filename}')

        # read one frame to see the size
        frame = self.read()
        if frame is None:
            raise IOError(f'Cannot read from given file or device')

        self._frame_channels = int(frame.shape[-1])
        self._seek(self._skip)

    def __len__(self):
        """ Length (number of frames) """
        return self.frame_count

    def __getitem__(self, index):
        """Now we can get frame via self[index] and self[start:stop:step]."""
        if isinstance(index, slice):
            return (self[ii] for ii in range(*index.indices(self.frame_count or 0)))
        elif isinstance(index, (list, tuple, range)):
            return (self[ii] for ii in index)
        else:
            return self.read(index)

    def __repr__(self) -> str:
        if self._filename:
            return f"{self._filename} {self.frame_shape[0]}x{self.frame_shape[1]}@{self.frame_rate:1.2f} {self.frame_count} frames length"

        return f"video #{self._device} {self.frame_shape[0]}x{self.frame_shape[1]}@{self.frame_rate:1.2f}"

    def __iter__(self):
        return self

    def __next__(self):
        if (frame := self.read()) is not None:
            return frame
        raise StopIteration()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._vr.release()

    def read(self, frame_number: int | None = None) -> np.ndarray | None:
        """ Read next frame or frame specified by `frame_number` """
        if self._filename:
            is_current_frame = frame_number == self.position
            if frame_number is not None and not is_current_frame:
                self._seek(frame_number)
        elif frame_number is not None:
            raise ValueError('Arbitrary positioning supported only with video files')

        ret, frame = self._vr.read()
        return frame if ret else None

    def close(self):
        """" Close the reader """
        self._vr.release()

    def _reset(self):
        """ Re-initialize object """
        assert (fn := self._filename or self._device) is not None
        self.__init__(fn)

    def _seek(self, frame_number: int):
        """ Go to frame """
        self._vr.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    @property
    def filename(self) -> str | None:
        """ Filename, if opened from file """
        return self._filename

    @property
    def device(self) -> int | None:
        """ Device ID, if live cam is opened """
        return self._device

    @property
    def media_type(self) -> MediaType:
        """ Media type """
        return self._media_type

    @property
    def display_title(self) -> str:
        """ Some kind of identyfing caption """
        return self._filename if self._filename else f'video #{self._device}'

    @property
    def position(self) -> int | None:
        """ Position in a file (None for live cam) """
        return r if (r := int(self._vr.get(cv2.CAP_PROP_POS_FRAMES))) != -1 else None

    @property
    def frame_count(self) -> int | None:
        """ Total number of frames (None for live cam)"""
        return r if (r := int(self._vr.get(cv2.CAP_PROP_FRAME_COUNT))) != -1 else None

    @property
    def frame_rate(self) -> float:
        """ Frame rate """
        return self._vr.get(cv2.CAP_PROP_FPS)

    @property
    def duration(self) -> timedelta | None:
        """ File duration, None for live cam """
        if self.frame_count and self.frame_count > 0:
            return timedelta(seconds=self.frame_count / self.frame_rate)

        return None

    @property
    def frame_height(self) -> int:
        return int(self._vr.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_width(self) -> int:
        return int(self._vr.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_channels(self) -> int:
        return self._frame_channels

    @property
    def frame_shape(self) -> Tuple[int,int,int]:
        return (self.frame_height, self.frame_width, self.frame_channels)

    @property
    def fourcc(self) -> str:
        """ Decoded FOURCC code """
        return int((self._vr.get(cv2.CAP_PROP_FOURCC))) \
            .to_bytes(4, byteorder=sys.byteorder).decode()

    @property
    def fps(self) -> float:
        return self._vr.get(cv2.CAP_PROP_FPS)

class PictureVideoCapture(cv2.VideoCapture):
    """ Video capture emulator for single picture """

    _image: np.ndarray | None
    _pos: int

    def __init__(self):
        super().__init__()
        self._image = None
        self._pos = 0

    def open(self, filename: str | Path, apiPreference: int = 0) -> bool: # type:ignore
        self._image = cv2.imread(str(filename))
        self._pos = 0
        return self._image is not None

    def release(self):
        self._image = None

    def read(self) -> Tuple[bool, np.ndarray | None]: # type:ignore
        if self._image is not None and not self._pos:
            self._pos = 1
            return (True, self._image)
        
        return (False, None)

    def get(self, propId: int) -> float:
        match propId:
            case cv2.CAP_PROP_FRAME_COUNT:
                return 1

            case cv2.CAP_PROP_FRAME_HEIGHT:
                if self._image is None:
                    raise ValueError('Not opened')
                return self._image.shape[0]

            case cv2.CAP_PROP_FRAME_WIDTH:
                if self._image is None:
                    raise ValueError('Not opened')
                return self._image.shape[1]

            case cv2.CAP_PROP_FOURCC:
                return cv2.VideoWriter.fourcc(*'JPEG')
            
            case _:
                return 0

    def set(self, propId: int, value: float) -> bool:
        match propId:
            case cv2.CAP_PROP_POS_FRAMES:
                if value >=0 and value <= 1:
                    self._pos = int(value)
                    return True
                return False

            case _:
                return False
