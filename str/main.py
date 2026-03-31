from pathlib import Path
from str.core.backend_dataset import check_folder_and_json, load_files_from_folder



DATA_FOLDER = Path("data")

PROCESSED_FILES_JSON = Path("data/processed_files.json")
PROCESSED_FILES_JSON_TEST = Path("data/processed_files.json")

def __main__():
    # 这里可以添加代码来测试上述函数的功能
    #all_available是一个布尔值，表示是否所有pdf文件都已经记录在json文件里面了，missing_files是一个列表，包含没有记录在json文件里面的pdf文件名称
    all_available, missing_files = check_folder_and_json(DATA_FOLDER, PROCESSED_FILES_JSON_TEST)
    print(f"All files in folder are recorded in json: {all_available}")
    if not all_available:
        print("Missing files in json:")
        for file in missing_files:
            print(f"- {file}")
        
        load_files_from_folder(DATA_FOLDER, PROCESSED_FILES_JSON_TEST, missing_files)

if __name__ == "__main__":
    __main__()