from utils.analysis import data_analysis

if __name__ == "__main__":
    print("Starting data analysis for RGB snowpole images...")
    data_analysis(
        data_path="/datasets/tdt4265/ad/open/Poles/rgb",
        save_dir="./data"
    )
    print("✅ Data analysis completed. Check './outputs/' for results.")