import streamlit as st
from pathlib import Path
from types import SimpleNamespace
import argparse
import base64
import logging
import json
import datetime
import numpy as np
import pandas as pd
import cv2

import fastmot
import fastmot.models
from fastmot.utils import ConfigDecoder, Profiler


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    optional = parser._action_groups.pop()
    optional.add_argument('-c', '--config', metavar="FILE",
                          default=Path(__file__).parent / 'cfg' / 'mot.json',
                          help='path to JSON configuration file')
    optional.add_argument('-l', '--labels', metavar="FILE",
                          help='path to label names (e.g. coco.names)')
    parser._action_groups.append(optional)
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(fastmot.__name__)
    logger.setLevel(logging.INFO)

    # load config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

    # load labels if given
    if args.labels is not None:
        with open(args.labels) as label_file:
            label_map = label_file.read().splitlines()
            fastmot.models.set_label_map(label_map)

    stream = fastmot.VideoIO(config.resize_to, "0", None, **vars(config.stream_cfg))

    mot = fastmot.MOT(config.resize_to, **vars(config.mot_cfg), draw=True)
    mot.reset(stream.cap_dt)


    max_length = 100000
    image_loc = st.empty()
    table = st.empty()
    download_link = st.empty()
    n_video = 8
    chart = st.line_chart(np.zeros((1, n_video)))
    columns = ["time"] + [f"count_video_{i}" for i in range(n_video)]
    df = pd.DataFrame(columns=columns)
    logger.info('Starting video capture...')
    stream.start_capture()
    try:
        with Profiler('webapp') as prof:
            while True:
                frame = stream.read()
                if frame is None:
                    break
                mot.step(frame)
                tlwhs = []
                ids = []
                for track in mot.visible_tracks():
                    tl = track.tlbr[:2] / config.resize_to * stream.resolution
                    br = track.tlbr[2:] / config.resize_to * stream.resolution
                    w, h = br - tl + 1
                    ids.append(track.trk_id)
                    tlwhs.append([tl[0], tl[1], w, h])
                image_loc.image(frame[:, :, ::-1], output_format="PNG")
                counts = np.zeros((3, 3))
                w_u = stream.resolution[0] // 3
                h_u = stream.resolution[1] // 3
                for tlx, tly, w, h in tlwhs:
                    center = (tlx + w / 2, tly + h / 2)
                    counts[int(center[1] / h_u), int(center[0] / w_u)] += 1
                counts = counts.reshape(1, 9)[:, :8]
                chart.add_rows(counts)
                if len(tlwhs) > 0:
                    df = df.append(pd.Series([datetime.datetime.now()] + counts[0].astype(int).tolist(), index=columns),
                                   ignore_index=True)
                    table.dataframe(df)
                    download_link.markdown(get_table_download_link(df), unsafe_allow_html=True)
                if len(df) > max_length:
                    df = df.iloc[(max_length // 2):, :]
    finally:
        stream.release()


if __name__ == '__main__':
    main()
