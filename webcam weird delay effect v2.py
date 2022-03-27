import cv2
import numpy as np
from typing import Optional
import tkinter.filedialog as filedialog
import tkinter


class CircularCacheDelayAccess:
    def __init__(self, img_shape: tuple, cache_size: int):
        self.image_shape = img_shape
        self.cache_size = cache_size
        # some useful stuff
        self.multi_index_shape = (cache_size,) + img_shape
        self.image_size = int(np.prod(img_shape))
        self.size = cache_size * self.image_size
        # the index, going around in circles
        self.cache_index = 0
        self.cache = np.empty(self.size)
        # raveled_image_indices is a running index over a frame; it is the same thing as writing
        #   y, x, c = np.mgrid[0:height, 0:width, 0:3]
        #   raveled_image_indices = c + 3 * (x + width * y)
        # but it's a lot easier
        self.raveled_image_indices = np.arange(self.image_size)

    def fillCache(self, image: Optional[np.ndarray] = None):
        if image is None:
            self.cache = np.random.randint(0, 257, self.size)
        else:
            if image.shape != self.image_shape:
                raise ValueError(f'image shape must be the same as the cache image shape ({self.image_shape})')
            self.cache = np.tile(image.flatten(), self.cache_size)
        self.cache_index = 0

    def store(self, image: np.ndarray):
        if image.shape != self.image_shape:
            raise ValueError(f'image shape must be the same as the cache image shape ({self.image_shape})')
        self.cache_index = (self.cache_index + 1) % self.cache_size
        cIndex = self.image_size * self.cache_index
        self.cache[cIndex:cIndex + self.image_size] = image.flatten()

    def getFrame(self, delay_map: np.ndarray):
        # delay_map may either have shape == self.image_shape, or shape = (self.image_size,)
        if delay_map.shape != self.image_shape:
            if len(delay_map.shape) > 1 or len(delay_map) != self.image_size:
                raise ValueError(f'delay_map shape must either be the same as the cache image shape ({self.image_shape})'
                                 f'or a flattened vector of length ({self.image_size})')
        else:
            # if we arrived in here, that means delay_map.shape == self.image_shape
            delay_map = delay_map.flatten()
        if delay_map.dtype != int:
            raise ValueError('delay_map must have dtype int')
        max_delay = np.max(delay_map)
        if max_delay > self.cache_size:
            raise IndexError(f'delay_map should at no point exceed cache_size ({max_delay} > {self.cache_size})')

        frame = self.cache[self.image_size * (self.cache_index - delay_map) + self.raveled_image_indices]\
            .reshape(self.image_shape)

        return frame


class VideoRecorder:
    def __init__(self, fps: int, width: int, height: int):
        self.fps = fps
        self.width = width
        self.height = height
        self.out = None
        self.filetypes = (('Mp4', '*.mp4'), ('Avi', '*.avi'))

    def invoke_start_recording(self):
        '''
        not sure about the word 'invoke'. If the recorder is already recording it ignores this invoke.
        :return:
        '''
        print('invoke_start_recording')
        if self.out is None:
            tkinter.Tk().withdraw()
            file_out = filedialog.asksaveasfilename(filetypes=self.filetypes, defaultextension=self.filetypes[0])
            if file_out:
                if file_out[-3:] == 'mp4':
                    vid_code = 'MP4V'
                elif file_out[-3:] == 'avi':
                    vid_code = 'XVID'
                else:
                    vid_code = None

                if vid_code:
                    fourcc = cv2.VideoWriter_fourcc(*vid_code)
                    self.out = cv2.VideoWriter(file_out, fourcc, self.fps, (self.width, self.height))
                    # VideoWriter sends error codes instead of exceptions, so I need to figure out how to catch these.
                    print('recording began')

    def invoke_stop_recording(self):
        print('invoke_stop_recording')
        if self.out is not None:
            self.out.release()
            self.out = None
            print('recording stopped')

    @staticmethod
    def _astype_uint8(frame: np.ndarray):
        if frame.dtype is np.uint8:
            return frame
        return frame.astype(np.uint8)

    def invoke_write_frame(self, frame: np.ndarray):
        if self.out is not None:
            if frame.shape == (self.height, self.width, 3):
                self.out.write(VideoRecorder._astype_uint8(frame))
            elif frame.shape == (self.height, self.width):
                frame = VideoRecorder._astype_uint8(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                self.out.write(frame)
            else:
                raise ValueError(f'frame must either have shape {(self.height, self.width)} (grayscale) or '
                                 f'{(self.height, self.width, 3)} (bgr)')


def main():
    T = 5
    start_with_image = True
    preset = 0

    # Playing video from file:
    # cap = cv2.VideoCapture('vtest.avi')
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rec = VideoRecorder(fps, width, height)

    y, x, c = np.mgrid[0:height, 0:width, 0:3]
    if preset == 0:
        delay_map = x / width * (T * fps)
    elif preset == 1:
        delay_map = y / height * (T * fps)
    elif preset == 2:
        delay_map = abs(y / height - .5) * (T * fps)
    elif preset == 3:
        delay_map = (1 - y / height) * (T * fps) + 2*c
    elif preset == 4:
        delay_map = np.exp(-((x - width/2)**2 + (y - height/2)**2) / (2 * (height/6)**2)) * (T * fps) + 10 * c
    elif preset == 5:
        delay_map = 10*c
    elif preset == 6:
        raise NotImplementedError('set delay_file as a path to an image on your computer, and delete this line :)')
        delay_file = r''
        delay_map = T * fps * plt.imread(delay_file)
    delay_map_int = delay_map.astype(int)
    delay_map_frac = delay_map - delay_map_int
    delay_map_int_flat = delay_map_int.flatten()

    effect_win_name = 'effect'
    webcam_win_name = 'webcam'
    cv2.startWindowThread()
    cv2.namedWindow(effect_win_name)
    cv2.startWindowThread()
    cv2.namedWindow(webcam_win_name)

    cache_size = np.max(delay_map_int_flat) + 1
    cache = CircularCacheDelayAccess((height, width, 3), cache_size)

    if start_with_image:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cache.fillCache(frame.astype(int))
    else:
        cache.fillCache()

    currentFrame = 0
    while cv2.getWindowProperty(effect_win_name, 0) >= 0 and cv2.getWindowProperty(webcam_win_name, 0) >= 0:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Handles the mirroring of the current frame
        frame = cv2.flip(frame, 1)
        cache.store(frame.astype(int))

        """if currentFrame < fps * T:
            dm = delay_map * currentFrame / (fps * T)
            dmi = dm.astype(int)
            dmf = dm - dmi
        else:
            dmi = delay_map_int_flat
            dmf = delay_map_frac"""
        frame0 = cache.getFrame(delay_map_int_flat)
        frame1 = cache.getFrame(delay_map_int_flat+1)
        frame_interp = frame0 + delay_map_frac * (frame1 - frame0)
        """frame0 = cache.getFrame(dmi)
        frame1 = cache.getFrame(dmi + 1)
        frame_interp = frame0 + dmf * (frame1 - frame0)"""
        frameOut = frame_interp.astype(np.uint8)

        # # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Saves image of the current frame in jpg file
        # name = 'frame' + str(currentFrame) + '.jpg'
        # cv2.imwrite(name, frame)

        # Display the resulting frame
        cv2.imshow(webcam_win_name, frame)
        cv2.imshow(effect_win_name, frameOut)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # VideoRecorder checks to make sure it doesn't start recording while already recording, and it ignores
        # invoke_write_frame if it isn't writing
        if key == ord('r'):
            rec.invoke_start_recording()
        if key == ord('s'):
            rec.invoke_stop_recording()
        rec.invoke_write_frame(frameOut)

        # Advance stuff eh
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    rec.invoke_stop_recording()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
