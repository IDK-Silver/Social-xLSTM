import argparse
import json
import matplotlib.pyplot as plt
from social_xlstm.utils.convert_coords import mercator_projection
from social_xlstm.dataset.utils.json_utils import VDLiveList, VDInfo
from social_xlstm.utils.graph import plot_vd_coordinates


def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--VDListJson", required=True, help="Json檔絕對位置")

    return parser.parse_args()

def main():
    args = parse_arguments()
    plot_vd_coordinates(args.VDListJson)

if __name__ == "__main__":
    main()


