import argparse
import json
import os
from mdutils.mdutils import MdUtils
from mdutils.fileutils import MarkDownFile
from mdutils.tools.Table import Table
from typing import Optional

MD_FILENAME = "MODEL_UPLOAD_HISTORY"
DIRNAME = "utils/model_uploader"
MODEL_JSON_FILENAME = DIRNAME + "/supported_models.json"
KEYS = ['Upload Time', 'Model Uploader', 'Model ID', 'Model Version', 'Tracing Format', 'Embedding Dimension', 'Pooling Mode']
HEADER = header = '# Pretrained Model Upload History\n\nThe model-serving framework supports a variety of open-source pretrained models that can assist with a range of machine learning (ML) search and analytics use cases. \n\n\n## Uploaded Pretrained Models\n\n\n### Sentence transformers\n\nSentence transformer models map sentences and paragraphs across a dimensional dense vector space. The number of vectors depends on the model. Use these models for use cases such as clustering and semantic search. \n\nThe following table shows sentence transformer model upload history.\n\n[//]: # (This may be the most platform independent comment)\n'

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
        'Model Uploader': '@'+ model_uploader if model_uploader is not None else 'N/A',
        'Upload Time': uploader_time if uploader_time is not None else 'N/A',
        'Model ID': model_id,
        'Model Version': model_version,
        'Tracing Format': tracing_format,
        'Embedding Dimension': embedding_dimension if embedding_dimension is not None else 'N/A',
        'Pooling Mode': pooling_mode if pooling_mode is not None else 'N/A', 
    }
    
    models.append(new_model)
    models = [dict(t) for t in {tuple(m.items()) for m in models}]
    models = sorted(models, key=lambda d: d['Upload Time'])
    with open(MODEL_JSON_FILENAME, 'w') as f:
        json.dump(models, f, indent=4)

def create_md_file():
    models = []
    if os.path.exists(MODEL_JSON_FILENAME):
        with open(MODEL_JSON_FILENAME, 'r') as f:
            models = json.load(f)
    models = sorted(models, key=lambda d: d['Upload Time'])
    table_data = KEYS[:]
    for m in models:
        for k in KEYS:
            if k == 'Model ID':
                 table_data.append(f"`{m[k]}`")
            else:
                table_data.append(m[k])
            
    table = Table().create_table(columns=len(KEYS), rows=len(models)+1, text=table_data, text_align='center')
    
    mdFile = MarkDownFile(MD_FILENAME, dirname=DIRNAME)
    mdFile.rewrite_all_file(data=header+table)
    print(f'Finished updating {MD_FILENAME}')
    
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
        "-t",
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
    
    create_md_file()
