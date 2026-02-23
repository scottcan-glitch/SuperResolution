from DataProcessor import DataProcessor


def main():
    data_processor = DataProcessor(raw_dir='../Data/RawData', processed_dir='../Data/ProcessedData')
    data_processor.calculate_color_std()