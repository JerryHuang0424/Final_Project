#这个文档是用来处理程序的后端数据
#有两个添加文件内的函数，一个是从指定的文件夹中加载数据到vectorstore中
#另一个是从前端用户输入数据，保存的vectorstore中
#使用一个json文件来存储已经处理过的文件的名称，所有数据的增删查改都要经过这个json文件

import json
from pathlib import Path
from datetime import datetime
from str.core.file_processor import get_pdf_text, get_text_into_chunks
from str.core.vector_store import update_vector_store, delete_vector_store
import str.constants as constants

# JSON文件路径

def check_json_exists(json_path):
    #这个方法需要一个参数，就是json文件的地址
    #这个方法的功能是检查json文件是否存在，如果不存在就创建一个空的json文件
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if not json_path.exists():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        print(f"Created new json file at {json_path}")
    else:
        print(f"Json file already exists at {json_path}")

def check_folder_and_json(folder_path, json_path):
    #这个方法需要两个参数，一个是目标folder的地址，还有一个是json文件的地址
    #这个方法的功能是检查目标folder里面的pdf文件是否已经在json文件里面记录了，如果没有就把没有记录的pdf文件名称返回出来，并且返回一个布尔值表示是否所有pdf文件都已经记录在json文件里面了
    #返回值是一个元组，第一个元素是一个布尔值，表示是否所有pdf文件都已经记录在json文件里面了，第二个元素是一个列表，包含没有记录在json文件里面的pdf文件名称
    all_in_json = True
    no_in_json_file = []

    # 确保JSON文件存在
    check_json_exists(json_path)

    # 把json文件里面数据加载到data_in_json数组中
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data_in_json = json.load(f)
        except:
            data_in_json = {}

    for pdf_file in folder_path.glob("*.pdf"):
        #检查json文件里面是否已经存在，如果没有就写入json文件中
        if pdf_file.name in data_in_json:
            print(f"{pdf_file.name} ✅ already recorded")
        elif not pdf_file.name in data_in_json:
            all_in_json = False
            no_in_json_file.append(pdf_file.name)
            print(f"{pdf_file.name} ❌ not recorded")

    return all_in_json, no_in_json_file


def load_files_from_folder(folder_path, json_path, no_in_json_file=None):
    #从指定的文件夹里面加载数据到vectorstore中，这个方法需要三个参数，第一个是目标folder的地址，第二个是json文件的地址，第三个是没有记录在json文件里面的pdf文件名称列表
    """从指定的文件夹中加载数据到vectorstore中"""
    if no_in_json_file is None:
        no_in_json_file = []

    # 确保JSON文件存在
    check_json_exists(json_path)

    #pdf文件要先经历text-->chunks-->embedding-->vectorstore的过程
    with open(json_path, "r+", encoding="utf-8") as f:
        #首先要把新的文件更新到json文件中
        data_in_json = json.load(f)
        for file_name in no_in_json_file:
            data_in_json[file_name] = {
            "status": "updated",
            "vector_store_updated": False,
            "last_processed": datetime.now().isoformat(),
            "size": Path(folder_path, file_name).stat().st_size
            }

    # 1. 把指定的pdf文件加载到text中
            print(f"Processing {file_name}...")
            text = get_pdf_text(Path(folder_path, file_name))

    # 2. 把text切分成chunks
            print(f"Splitting text from {file_name} into chunks...")
            chunks = get_text_into_chunks(text,constants.CHUNK_SIZE,constants.CHUNK_OVERLAP)

            print(f"Embedding chunks from {file_name} and updating vector store...")
            update_vector_store(chunks, constants.EMBEDDING_MODEL)

            # 更新JSON文件中的状态
            data_in_json[file_name]["vector_store_updated"] = True
            data_in_json[file_name]["last_processed"] = datetime.now().isoformat()

        f.seek(0) #把文件指针移动到文件开头
        f.truncate()
        json.dump(data_in_json, f, indent=4)


def load_files_from_user_input(user_input):
    """从前端用户输入数据，保存的vectorstore中"""
    # 这里可以添加代码来处理用户输入的数据并将其保存到vectorstore中
    pass


def delete_processed_file(file_name, json_path):
    """从json文件中删除已经处理过的文件名称,并且从vectorstore中删除对应的数据"""
    # 确保JSON文件存在
    check_json_exists(json_path)

    with open(json_path, "r+", encoding="utf-8") as f:
        data_in_json = json.load(f)
        if file_name in data_in_json:
            del data_in_json[file_name]
            f.seek(0)
            f.truncate()
            json.dump(data_in_json, f, indent=4)
            print(f"Deleted {file_name} from JSON file")
        else:
            print(f"{file_name} not found in JSON file")

    # 注意：这里需要更复杂的逻辑来从vectorstore中删除特定文件的嵌入
    # 由于FAISS不支持部分删除，通常需要重新构建整个索引
    # 暂时只删除JSON记录，vectorstore保持完整


def initialize_vector_store():
    """初始化向量存储，如果磁盘上有保存的向量存储则加载"""
    from  models.embeddings import get_embedding_model
    from  core.vector_store import load_vector_store

    embedding_model = get_embedding_model()
    if embedding_model:
        vector_store = load_vector_store(embedding_model)
        return vector_store
    return None

