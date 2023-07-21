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

CHANGELOG_DIRNAME = "."
CHANGELOG_FILENAME = "CHANGELOG.md"
SECTION_NAME = "Added"

def update_changelog_file(
    model_id: str,
    model_version: str,
    tracing_format: str,
    workflow_initiator: str,
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
    :param workflow_initiator: Workflow initiator
    :type workflow_initiator: string
    :param pr_number: Pull request number
    :type pr_number: int
    :param pr_url: Pull request URL
    :type pr_url: string
    """
    changelog_data = MarkDownFile.read_file(f"{CHANGELOG_DIRNAME}/{CHANGELOG_FILENAME}")
    
    this_version_ptr = changelog_data.find('## [')
    assert this_version_ptr != -1, "Cannot find a version section in the CHANGELOG.md"
    next_version_ptr = changelog_data.find('## [', this_version_ptr + 1)
    this_version_section = changelog_data[this_version_ptr:next_version_ptr]
    
    this_subsection_ptr = newest_version.find(f'### {SECTION_NAME}')
    
    new_line = f"Update model upload history - {model_id} (v.{model_version})({tracing_format}) by @{workflow_initiator} ([#{pr_number}]({pr_url}))"
    
    if this_subsection_ptr != -1:
        next_subsection_ptr = newest_version.find('### ', this_subsection_ptr + 1)
        this_subsection = this_version_section[this_subsection_ptr:next_subsection_ptr].strip()
        this_subsection += '\n- ' + new_line + '\n\n'
        new_version_section = this_version_section[:this_subsection_ptr] + this_subsection + this_version_section[next_subsection_ptr:]
    else:
        this_subsection = this_version_section.strip()
        this_subsection += '\n\n' + f'### {SECTION_NAME}\n- ' + new_line + '\n\n'
        new_version_section = this_subsection
        
   
    new_changelog_data = file_data[:this_version_ptr] + new_version_section + file_data[next_version_ptr:]
    
    mdFile = MarkDownFile(CHANGELOG_FILENAME, dirname=CHANGELOG_DIRNAME)
    mdFile.rewrite_all_file(data=new_changelog_data)


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
        "workflow_initiator",
        type=str,
        help="Workflow Initiator",
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
        arg.workflow_initiator,
        args.pr_number,
        args.pr_url,
    )
