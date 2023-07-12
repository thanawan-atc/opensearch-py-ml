# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import argparse
import os


def get_model_file_path(model_id, model_version, model_format):
    model_name = str(model_id.split("/")[-1])
    model_format = model_format.lower()
    model_dirname = f"{model_name}/{model_version}/{model_format}"
    model_filename = (
        f"sentence-transformers_{model_name}-{model_version}-{model_format}.zip"
    )
    model_file_path = model_dirname + "/" + model_filename
    return model_file_path


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
        "model_format",
        choices=["TORCH_SCRIPT", "ONNX"],
        help="Model format for auto-tracing",
    )

    args = parser.parse_args()

    model_file_path = get_model_file_path(
        args.model_id, args.model_version, args.model_format
    )

    env_file = os.getenv('GITHUB_ENV')
    var_name = "TORCH_FILE_PATH" if model_format == "TORCH_SCRIPT" else "ONNX_FILE_PATH"
    with open(env_file, "a") as f:
        f.write(f"{var_name}={model_file_path}")
