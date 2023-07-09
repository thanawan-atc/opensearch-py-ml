# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import argparse
import os
import shutil
import sys
import warnings
from zipfile import ZipFile

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.abspath(os.path.join("../opensearch-py-ml"))
LICENSE_PATH = os.path.join(ROOT_DIR, "LICENSE")
sys.path.append(ROOT_DIR)

from opensearch_py_ml.ml_commons import MLCommonClient
from opensearch_py_ml.ml_models.sentencetransformermodel import SentenceTransformerModel
from tests import OPENSEARCH_TEST_CLIENT

TORCH_SCRIPT_FORMAT = "TORCH_SCRIPT"
ONNX_FORMAT = "ONNX"
BOTH_FORMAT = "BOTH"
ORIGINAL_FOLDER_PATH = "sentence-transformers-original/"
TORCHSCRIPT_FOLDER_PATH = "sentence-transformers-torchscript/"
ONNX_FOLDER_PATH = "sentence-transformers-onnx/"
MODEL_CONFIG_FILE_NAME = "ml-commons_model_config.json"
TEST_SENTENCES = ["First test sentence", "Second test sentence"]
RTOL_TEST = 1e-03
ATOL_TEST = 1e-05
ML_BASE_URI = "/_plugins/_ml"


def trace_sentence_transformer_model(
    model_id, model_version, embedding_dimension, pooling_mode, model_format
):
    folder_path = (
        TORCHSCRIPT_FOLDER_PATH
        if model_format == TORCH_SCRIPT_FORMAT
        else ONNX_FOLDER_PATH
    )

    pre_trained_model = None
    try:
        pre_trained_model = SentenceTransformerModel(
            model_id=model_id, folder_path=folder_path, overwrite=True
        )
    except Exception as e:
        assert (
            False
        ), f"Raised Exception in tracing {model_format} model\
                             during initiating a sentence transformer model class object: {e}"

    model_path = None
    try:
        if model_format == TORCH_SCRIPT_FORMAT:
            model_path = pre_trained_model.save_as_pt(
                model_id=model_id, sentences=TEST_SENTENCES
            )
        else:
            model_path = pre_trained_model.save_as_onnx(model_id=model_id)
    except Exception as e:
        assert False, f"Raised Exception during saving model as {model_format}: {e}"

    try:
        pre_trained_model.make_model_config_json(
            version_number=model_version,
            model_format=model_format,
            embedding_dimension=embedding_dimension,
            pooling_mode=pooling_mode,
        )
    except Exception as e:
        assert (
            False
        ), f"Raised Exception during making model config file for {model_format} model: {e}"

    model_config_path = folder_path + MODEL_CONFIG_FILE_NAME

    return model_path, model_config_path


def upload_sentence_transformer_model(
    ml_client, model_path, model_config_path, model_format
):
    embedding_data = None

    model_id = ""
    task_id = ""
    try:
        model_id = ml_client.register_model(
            model_path=model_path,
            model_config_path=model_config_path,
            deploy_model=False,
            isVerbose=True,
        )
        print()
        print(f"{model_format}_model_id:", model_id)
        assert model_id != "" or model_id is not None
    except Exception as e:
        assert False, f"Raised Exception in {model_format} model registration: {e}"

    try:
        ml_client.deploy_model(model_id)
        api_url = f"{ML_BASE_URI}/models/{model_id}/_deploy"
        task_id = ml_client._client.transport.perform_request(
            method="POST", url=api_url
        )["task_id"]
        assert task_id != "" or task_id is not None
        ml_model_status = ml_client.get_model_info(model_id)
        assert ml_model_status.get("model_state") != "DEPLOY_FAILED"
        print(f"{model_format}_task_id:", task_id)
    except Exception as e:
        assert False, f"Raised Exception in {model_format} model deployment: {e}"

    try:
        ml_model_status = ml_client.get_model_info(model_id)
        print()
        print("Model Status:")
        print(ml_model_status)
        assert ml_model_status.get("model_format") == model_format
        assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"
    except Exception as e:
        assert False, f"Raised Exception in getting {model_format} model info: {e}"

    ml_task_status = None
    try:
        ml_task_status = ml_client.get_task_info(task_id, wait_until_task_done=True)
        print()
        print("Task Status:")
        print(ml_task_status)
        assert ml_task_status.get("task_type") == "DEPLOY_MODEL"
        assert ml_task_status.get("state") != "FAILED"
    except Exception as e:
        print("Model Task Status:", ml_task_status)
        assert (
            False
        ), f"Raised Exception in pulling task info for {model_format} model: {e}"

    try:
        embedding_output = ml_client.generate_embedding(model_id, TEST_SENTENCES)
        assert len(embedding_output.get("inference_results")) == 2
        embedding_data = embedding_output["inference_results"][0]["output"][0]["data"]
    except Exception as e:
        assert (
            False
        ), f"Raised Exception in generating sentence embedding with {model_format} model: {e}"

    try:
        delete_task_obj = ml_client.delete_task(task_id)
        assert delete_task_obj.get("result") == "deleted"
    except Exception as e:
        assert False, f"Raised Exception in deleting task for {model_format} model: {e}"

    try:
        ml_client.undeploy_model(model_id)
        ml_model_status = ml_client.get_model_info(model_id)
        assert ml_model_status.get("model_state") != "UNDEPLOY_FAILED"
    except Exception as e:
        assert False, f"Raised Exception in {model_format} model undeployment: {e}"

    try:
        delete_model_obj = ml_client.delete_model(model_id)
        assert delete_model_obj.get("result") == "deleted"
    except Exception as e:
        assert False, f"Raised Exception in deleting {model_format} model: {e}"

    return embedding_data


def verify_embedding_data(
    original_embedding_data, tracing_embedding_data, tracing_format
):
    try:
        np.testing.assert_allclose(
            original_embedding_data,
            tracing_embedding_data,
            rtol=RTOL_TEST,
            atol=ATOL_TEST,
        )
    except Exception as e:
        assert False, f"Raised Exception in embedding verification: {e}"

    print()
    print(f"Original embeddings matches {tracing_format} embeddings")


def prepare_files_for_uploading(
    model_id, model_version, model_format, src_model_path, src_model_config_path
):
    model_name = str(model_id.split("/")[-1])
    model_format = model_format.lower()
    folder_to_delete = (
        TORCHSCRIPT_FOLDER_PATH if model_format == "torch_script" else ONNX_FOLDER_PATH
    )

    try:
        dst_model_dir = f"upload/{model_name}/{model_version}/{model_format}"
        os.makedirs(dst_model_dir, exist_ok=True)
        dst_model_filename = (
            f"sentence-transformers_{model_name}-{model_version}-{model_format}.zip"
        )
        dst_model_path = dst_model_dir + "/" + dst_model_filename
        with ZipFile(src_model_path, "a") as zipObj:
            zipObj.write(filename=LICENSE_PATH, arcname="LICENSE")
        shutil.copy(src_model_path, dst_model_path)
        print()
        print(f"Copied {src_model_path} to {dst_model_path}")

        dst_model_config_dir = f"upload/{model_name}/{model_version}/{model_format}"
        os.makedirs(dst_model_config_dir, exist_ok=True)
        dst_model_config_filename = "config.json"
        dst_model_config_path = dst_model_config_dir + "/" + dst_model_config_filename
        shutil.copy(src_model_config_path, dst_model_config_path)
        print(f"Copied {src_model_config_path} to {dst_model_config_path}")
    except Exception as e:
        assert (
            False
        ), f"Raised Exception during preparing {model_format} files for uploading: {e}"

    try:
        shutil.rmtree(folder_to_delete)
    except Exception as e:
        assert False, f"Raised Exception while deleting {folder_to_delete}: {e}"


def main(args):
    print()
    print("=== Begin running model_autotracing.py ===")
    print("Model ID: ", args.model_id)
    print("Model Version: ", args.model_version)
    print("Tracing Format: ", args.tracing_format)
    print("Embedding Dimension: ", args.embedding_dimension)
    print("Pooling Mode: ", args.pooling_mode)
    print("==========================================")

    ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)

    pre_trained_model = SentenceTransformer(args.model_id)
    original_embedding_data = pre_trained_model.encode(
        TEST_SENTENCES, convert_to_numpy=True
    )[0]

    if args.tracing_format in [TORCH_SCRIPT_FORMAT, BOTH_FORMAT]:
        print("--- Begin tracing a model in TORCH_SCRIPT ---")
        (
            torchscript_model_path,
            torchscript_model_config_path,
        ) = trace_sentence_transformer_model(
            args.model_id,
            args.model_version,
            args.embedding_dimension,
            args.pooling_mode,
            TORCH_SCRIPT_FORMAT,
        )
        torch_embedding_data = upload_sentence_transformer_model(
            ml_client,
            torchscript_model_path,
            torchscript_model_config_path,
            TORCH_SCRIPT_FORMAT,
        )
        verify_embedding_data(
            original_embedding_data, torch_embedding_data, TORCH_SCRIPT_FORMAT
        )
        prepare_files_for_uploading(
            args.model_id,
            args.model_version,
            TORCH_SCRIPT_FORMAT,
            torchscript_model_path,
            torchscript_model_config_path,
        )
        print("--- Finished tracing a model in TORCH_SCRIPT ---")

    if args.tracing_format in [ONNX_FORMAT, BOTH_FORMAT]:
        print("--- Begin tracing a model in ONNX ---")
        onnx_model_path, onnx_model_config_path = trace_sentence_transformer_model(
            args.model_id,
            args.model_version,
            args.embedding_dimension,
            args.pooling_mode,
            ONNX_FORMAT,
        )
        onnx_embedding_data = upload_sentence_transformer_model(
            ml_client, onnx_model_path, onnx_model_config_path, ONNX_FORMAT
        )

        verify_embedding_data(original_embedding_data, onnx_embedding_data, ONNX_FORMAT)
        prepare_files_for_uploading(
            args.model_id,
            args.model_version,
            ONNX_FORMAT,
            onnx_model_path,
            onnx_model_config_path,
        )
        print("--- Finished tracing a model in ONNX ---")

    print()
    print("=== Finished running model_autotracing.py ===")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    warnings.filterwarnings("ignore", message="TracerWarning: torch.tensor")
    warnings.filterwarnings(
        "ignore", message="using SSL with verify_certs=False is insecure."
    )

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
    args = parser.parse_args()

    main(args)

    # TODO: Check if model exists in database
