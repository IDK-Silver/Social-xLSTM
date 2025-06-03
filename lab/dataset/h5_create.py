from social_xlstm.dataset.h5_utils import ensure_traffic_hdf5, create_traffic_hdf5
def main():
    reader = create_traffic_hdf5(
        source_dir="blob/dataset/pre-processed/unzip_to_json",
        output_path="blob/dataset/pre-processed/h5/traffic_features.h5",
        selected_vdids=None
    )

if __name__ == "__main__":
    main()