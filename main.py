import numpy as np
import av
from av.video.frame import VideoFrame
import pylab
from mpl_toolkits.mplot3d import Axes3D
from numpy import sqrt


class VideoHandler:
    __K_R = 0.299
    __K_G = 0.507
    __K_B = 0.114
    last_height = None
    last_width = None
    last_numb_frames = None

    def __init__(self):
        super().__init__()

    # Возвращает видео в виде списка, содержащего кадры в виду numpy array
    @staticmethod
    def input_video(path, reformat_name=None):
        array_list = []
        input_container = av.open(path)
        
        frame_width = input_container.streams.video[0].codec_context.width
        frame_height = input_container.streams.video[0].codec_context.height
        
        VideoHandler.last_height = frame_height
        VideoHandler.last_width = frame_width
        VideoHandler.last_numb_frames = input_container.streams.video[0].frames

        for frame in input_container.decode(video=0):
            frame = frame.reformat(frame_width, frame_height, reformat_name)
            array = frame.to_ndarray()
            array_list += [array]

        par = [array_list,
               input_container.format.name,  # format_container_name
               input_container.streams.video[0].codec_context.format.name,  # format_frame_name
               input_container.streams.video[0].codec_context.name,  # codec_name
               input_container.streams.video[0].average_rate]  # rate

        if reformat_name is not None:
            par[2] = reformat_name

        input_container.close()

        return par

    @staticmethod
    def viniet(path_in, path_out, n):
        input_container = av.open(path_in)
        output_container = av.open(path_out, 'w')

        input_stream = input_container.streams.video[0]
        frames = []
        out_stream = output_container.add_stream("rawvideo", rate=round(input_stream.average_rate))
          
        out_stream.width = input_stream.width
        out_stream.height = input_stream.height
        out_stream.bit_rate = input_stream.bit_rate
        for packet in input_container.demux(input_stream):
            for frame in packet.decode():
                frames.append(np.array(frame.to_image()))
        frames = np.array(frames)
        h = input_stream.height - 1
        w = input_stream.width - 1
        out_stream.rate = input_stream.average_rate / n
        
        frames = frames[::n, :, :, :]
        
        for frame in frames:
            image = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in out_stream.encode(image):
                output_container.mux(packet)
        output_container.close()

    @staticmethod
    def output_video(path, par):
        output_container = av.open(path, mode='w', format=par[1])
        output_stream = output_container.add_stream(par[3], rate=par[4])
        for array in par[0]:
            frame = VideoFrame.from_ndarray(array, format=par[2])
            frame = output_stream.encode(frame)
            output_container.mux(frame)

        output_container.close()

    @staticmethod
    def output_reverse_video(path, par, count_frame_quantity):
        array_list_out = []
        count = 0
        for array in par[0]:
            frame = VideoFrame.from_ndarray(array, format=par[2])
            frame.pts = count_frame_quantity - count
            array_list_out += [frame.to_ndarray()]
            count += 1
        array_list_out.reverse()
        par[0] = array_list_out
        VideoHandler.output_video(path, par)

    @staticmethod
    def frame_to_image(path):
        container = av.open(path)
        array_list = []
        for frame in container.decode(video=0):
            array_list += [frame.to_image()]
        return array_list

    @staticmethod
    def math_expectation(array):
        return array.sum() / array.size

    @staticmethod
    def sigma(array, math_exp):
        res = 0
        for el in array:
            res += pow(el - math_exp, 2)
        return sqrt(res / array.size)

@staticmethod
def rgb2ycbcr(image):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = image.dot(xform.T) 
    ycbcr[:, :, [1, 2]] += 0.5
    np.clip(ycbcr, 0, 1, out=ycbcr) 
    return np.float32(ycbcr)

def new_y(comp, n): 
    for i in range(3):
        comp[i] = (comp[i] - n) if (comp[i] - n) >= 0 else 0   
    return comp


@staticmethod
def correlation_for_vid(video):
    frames = []

    for packet in input.demux(in_stream):
        for frame in packet.decode():
            frames.append(np.array(frame.to_image()))

    frames = np.array(frames)
    ycbcr_frames = np.array([rgb2ycbcr(x) for x in frames[:,:,:,:]/255.0])

    CorrCoefs = []

    for signal1 in ycbcr_frames:
        tmp = []
        for signal2 in ycbcr_frames:
            tmp.append(np.mean((signal1[:, :, 0] - np.mean(signal1[:, :, 0])) * (signal2[:, :, 0] - np.mean(signal2[:, :, 0]))) / ( np.std(signal1[:, :, 0]) * np.std(signal2[:, :, 0])))
        CorrCoefs.append(tmp)
    CorrCoefs = np.array(CorrCoefs)

    x = np.arange(0, len(CorrCoefs), 1)
    y = np.arange(0, len(CorrCoefs), 1)
    xgrid, ygrid = np.meshgrid(x, y)
    fig = pylab.figure()
    axes = Axes3D(fig, auto_add_to_figure = False)
    fig.add_axes(axes)
    axes.plot_surface(xgrid, ygrid, CorrCoefs, cmap = pylab.colormaps()[88])
    # pylab.show()


if __name__ == "__main__":

    args = VideoHandler.input_video('Videos/lr1_1.avi', 'rgb24')
    
    VideoHandler.output_reverse_video('Videos/out_lr1_1_inv.avi', args, VideoHandler.last_numb_frames)

    args[0] = VideoHandler.input_video('Videos/lr1_1.avi', 'rgb24')[0] + VideoHandler.input_video('Videos/lr1_2.avi', 'rgb24')[0]

    VideoHandler.output_video('Videos/out_lr1_1_2.avi', args)

    print("Correlation is coming!")

    input = av.open("Videos/lr1_3.avi")
    in_stream = input.streams.video[0]
    correlation_for_vid(in_stream)
    print("Correlation complete!")
    input = av.open("Videos/lr1_2.avi")
    in_stream1 = input.streams.video[0]
    correlation_for_vid(in_stream1)
    print("Correlation complete!")
    input = av.open("Videos/lr1_1.avi")
    in_stream2 = input.streams.video[0]
    correlation_for_vid(in_stream2)
    print("Correlation complete!")
    pylab.show()
   # VideoHandler.viniet('Videos/lr1_1.avi', 'Videos/out_lr1_3.avi', 6)