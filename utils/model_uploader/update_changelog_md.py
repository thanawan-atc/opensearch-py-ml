# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

# This program is run by "Model Auto-tracing & Uploading" workflow
# (See model_uploader.yml) to update MODEL_UPLOAD_HISTORY.md & supported_models.json
# after uploading the model to our model hub.

import argparse

from mdutils.fileutils import MarkDownFile
from mdutils.tools.Table import Table

CHANGELOG_FILENAME = "CHANGELOG.MD"


def update_changelog_file(
    model_id: str,
    model_version: str,
    tracing_format: str,
    pr_number: int,
    pr_url: str,
) -> None:
    """
    Update supported_models.json

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param tracing_format: Tracing format ("TORCH_SCRIPT", "ONNX", or "BOTH")
    :type tracing_format: string
    :param pr_number: Pull request number
    :type pr_number: int
    :param pr_url: Pull request URL
    :type pr_url: string
    """
    mdFile = MarkDownFile(CHANGELOG_FILENAME)
    mdFile_str = mdFile.read_file()
    mdFile.rewrite_all_file(data=mdFile_str)


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
        "pr_number",
        type=int,
        help="Pull request number",
    )

    parser.add_argument(
        "pr_url",
        type=str,
        help="Pull request URL",
    )

    args = parser.parse_args()

    update_changelog_file(
        args.model_id,
        args.model_version,
        args.tracing_format,
        args.pr_number,
        args.pr_url,
    )
