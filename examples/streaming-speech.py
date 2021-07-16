import sys
from pathlib import Path

path = str(Path(__file__).parents[1].resolve())
sys.path.append(path)

import argparse
import numpy as np
import parselmouth
import pyaudio
from sonorus.audio import VADAudioInputStreamer
from sonorus.audio.utils import audio_float2int
from sonorus.audio.praat import reduce_noise, change_gender, change_pitch
import rospy
from std_msgs.msg import UInt8MultiArray

parser = argparse.ArgumentParser()

parser.add_argument(
    "-g",
    "--to_gender",
    choices=["male", "female"],
    help="Specify target gender if different than the operator",
)

parser.add_argument(
    "-v",
    "--voice_conv_fn",
    choices=["change_gender", "change_pitch"],
    help="Specify function to be used for voice conversion",
    default="change_pitch",
)

parser.add_argument(
    "-t",
    "--topic",
    help="ROS topic to which the speech should be published",
    default="/hr/control/audio/stream",
)

parser.add_argument(
    "-m",
    "--multiplier",
    type=float,
    help="Factor/ratio multiplier to be used for formant change ratio in case "
    "of change_gender function or the factor in case of change_pitch "
    " fucntion. Multiplier > 1 for female target voice while < 1 for male"
    " target voice.",
    default=1.5,  # for female voice target using change_pitch function
)

args = parser.parse_args()

audio_streamer = VADAudioInputStreamer(
    pa_format=pyaudio.paInt16, yield_accumulated=(True if args.to_gender else False),
)

pub = rospy.Publisher(args.topic, UInt8MultiArray, queue_size=10)
rospy.init_node("audio_output")
# r = rospy.Rate(10) # 10hz

audio_msg = UInt8MultiArray()

while not rospy.is_shutdown():
    with audio_streamer as streamer:
        for stream in streamer.stream():
            if stream is not None:

                msg_stream = (stream,)
                if args.to_gender:

                    dtype = streamer.FMT2TYPE[streamer.pa_format]
                    sound = parselmouth.Sound(
                        values=np.frombuffer(stream, dtype=dtype),
                        sampling_frequency=streamer.processing_rate,
                    )

                    if args.voice_conv_fn == "change_pitch":
                        changed_sound = change_pitch(sound, factor=args.multiplier)
                    else:
                        changed_sound = change_gender(
                            sound, formant_shift_ratio=args.multiplier,
                        )

                    stream = audio_float2int(changed_sound.values).tobytes()
                    msg_stream = [
                        stream[i : i + streamer.chunk]
                        for i in range(0, len(stream), streamer.chunk)
                    ]

                for msg in msg_stream:
                    audio_msg.data = msg
                    pub.publish(audio_msg)
                    # r.sleep()
