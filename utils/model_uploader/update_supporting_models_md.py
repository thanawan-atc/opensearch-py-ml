import os
from mdutils.mdutils import MdUtils

MODEL_JSON_FILENAME = "utils/model_uploader/supported_models.json"
KEYS = ['Model ID', 'Model Version', 'Tracing Format', 'Embedding Dimension', 'Pooling Mode', 'Uploaded by', 'Date']

def modify_model_json_file(
    model_id: str,
    model_version: str,
    tracing_format: str,
    embedding_dimension: Optional[int] = None,
    pooling_mode: Optional[str] = None,
    model_uploader: Optional[str] = None,
    uploader_time: Optional[str] = None,
) -> list[dict]:
    models = []
    if os.path.exists(MODEL_JSON_FILENAME):
        with open(MODEL_JSON_FILENAME, 'r') as f:
            models = json.load(f)
            
    new_model = {
        'Model ID': model_id,
        'Model Version': model_version,
        'Tracing Format': tracing_format if not 'BOTH' else 'TORCH_SCRIPT & ONNX',
        'Embedding Dimension': 'N/A' if embedding_dimension is None else embedding_dimension,
        'Pooling Mode': 'N/A' if pooling_mode is None else pooling_mode,
        'Model Upload': '@'+ model_uploader,
        'Upload Time': uploader_time
    }
    
    models.append(model)
    with open(MODEL_JSON_FILENAME, 'w') as f:
        json.dump(models, f, indent=4)

def create_table():
    models = []
    if os.path.exists(MODEL_JSON_FILENAME):
        with open(MODEL_JSON_FILENAME, 'r') as f:
            models = json.load(f)
            
    mdFile = MdUtils(file_name='SUPPORTED_MODELS',title='Supported models')
    mdFile.new_header(level=1, title='Sentence Transformers')
    table_data = KEYS[:]
    for m in models:
        for k in key_lst:
            table_data.append(m[k])
    mdFile.new_line()
    mdFile.new_paragraph("The following table provides a list of sentence transformer models available on model hub.")
    mdFile.new_table(columns=len(key_lst), rows=len(models)+1, text=text_lst, text_align='center')
    mdFile.create_md_file()

 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID for auto-tracing and uploading (e.g. sentence-transformers/msmarco-distilbert-base-tas-b)",
    )
    parser.add_argument(
        "model_version", type=str, help="Model version number (e.g. 1.0.1)"
    )
    parser.add_argument(
        "tracing_format",
        choices=["BOTH", "TORCH_SCRIPT", "ONNX"],
        help="Model format for auto-tracing",
    )
    parser.add_argument(
        "-ed",
        "--embedding_dimension",
        type=int,
        nargs="?",
        default=None,
        const=None,
        help="Embedding dimension of the model to use if it does not exist in original config.json",
    )
    
    parser.add_argument(
        "-pm",
        "--pooling_mode",
        type=str,
        nargs="?",
        default=None,
        const=None,
        choices=["CLS", "MEAN", "MAX", "MEAN_SQRT_LEN"],
        help="Pooling mode if it does not exist in original config.json",
    )
    
    parser.add_argument(
        "-u",
        "--model_uploader",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Model Uploader",
    )
    
    parser.add_argument(
        "-d",
        "--upload_time",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Upload Time",
    )
    args = parser.parse_args()
    
    modify_model_json_file(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.embedding_dimension,
        args.pooling_mode,
        args.model_uploader,
        args.upload_time
    )
    
    create_table()