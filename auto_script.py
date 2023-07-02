import os
import opensearchpy
from opensearchpy import OpenSearch

import json
import warnings
from opensearchpy import OpenSearch

ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))
OPENSEARCH_PY_ML_PARENT_DIR = os.path.join(ROOT_DIR, "..")

from OPENSEARCH_PY_ML_PARENT_DIR import opensearch_py_ml
from opensearch_py_ml.common import os_version 
from opensearch_py_ml.ml_models import SentenceTransformerModel
from opensearch_py_ml.ml_commons import MLCommonClient

OPENSEARCH_HOST = "https://instance:9200"
TORCH_SCRIPT_FORMAT = 'TORCH_SCRIPT'
ONNX_FORMAT = 'ONNX'
TORCHSCRIPT_FOLDER_PATH = "sentence-transformers-torchscript/"
ONXX_FOLDER_PATH = "sentence-transformers-onxx/"
MODEL_CONFIG_FILE_NAME = "ml-commons_model_config.json"
TEST_SENTENCES = ["First test sentence", "Second test sentence"]
RTOL_TEST = 1e-03
ATOL_TEST = 1e-05
    
    
def get_embedding_dimension(embedding_dimension_input):
    if embedding_dimension_input.isnumeric():
        return int(embedding_dimension_input)
    raise AssertionError(f"Invalid embedding dimension input: {embedding_dimension_input}")

    
def get_os_client(cluster_url = OPENSEARCH_HOST,
                  username='admin',
                  password='admin'):
    '''
    Get OpenSearch client
    :param cluster_url: cluster URL like https://ml-te-netwo-1s12ba42br23v-ff1736fa7db98ff2.elb.us-west-2.amazonaws.com:443
    :return: OpenSearch client
    '''
    os_client = OpenSearch(
        hosts=[cluster_url],
        http_auth=(username, password),
        verify_certs=False
    )
    try:
        _os_version = os_version(OPENSEARCH_CLIENT)
    except opensearchpy.exceptions.ConnectionError:
        raise AssertionError("Failed to connect to OpenSearch cluster")
    return client 


def trace_sentence_transformer_model(model_id, embedding_dimension, model_format):
    folder_path = TORCHSCRIPT_FOLDER_PATH if model_format == TORCH_SCRIPT_FORMAT else ONXX_FOLDER_PATH
    save_in_format_function = save_as_pt() if model_format == TORCH_SCRIPT_FORMAT else save_as_onnx()
    
    pre_trained_model = None
    try:
        pre_trained_model =  SentenceTransformerModel(folder_path=folder_path, overwrite=True)
    except:
        raise AssertionError(f"Raised Exception in tracing {model_format} model\
                             during initiating a sentence transformer model class object")
    
    # TODO: Check if model exists in database
    
    model_path = None
    raised = False
    try:
        model_path = pre_trained_model.save_in_format_function(model_id=model_id)
    except:  # noqa: E722
        raised = True
    assert raised == False, "Raised Exception during saving model as torchscript"
        
    raised = False
    try:
        pre_trained_model.make_model_config_json(
            folder_path=folder_path,
            model_format=model_format, 
            embedding_dimension=embedding_dimension
        )
    except:
        raised = True
    assert raised == False, "Raised Exception during making model config file for torchscript model"
    model_config_path = folder_path + MODEL_CONFIG_FILE_NAME
    
    return model_path, model_config_path


def upload_sentence_transformer_model(model_path, model_config_path, model_format):
    embedding_data = None
    
    raised = False
    try:
        ml_client.register_model(
        model_path=model_path,
        model_config_path=torchscript_model_config_path,
        deploy_model=False,
        isVerbose=True,
    )
    except:  # noqa: E722
        raised = True
    assert raised == False, f"Raised Exception in {model_format} model registration"
    
    model_id = ""
    task_id = ""
    raised = False
    try:
        model_id = ml_client.register_model(
            model_path=model_path,
            model_config_path=torchscript_model_config_path,
            deploy_model=False,
            isVerbose=True,
        )
        print("torchscript_model_id:", model_id)
    except:  # noqa: E722
        raised = True
    assert raised == False, f"Raised Exception in {model_format} model registration"
    
    if model_id:
        raised = False
        try:
            ml_load_status = ml_client.deploy_model(model_id)
            task_id = ml_load_status.get("task_id")
            assert task_id != "" or task_id is not None

            ml_model_status = ml_client.get_model_info(model_id)
            assert ml_model_status.get("model_state") != "DEPLOY_FAILED"
        except:  # noqa: E722
            raised = True
        assert raised == False, f"Raised Exception in {model_format} model deployment"

        raised = False
        try:
            ml_model_status = ml_client.get_model_info(model_id)
            assert ml_model_status.get("model_format") == model_format
            assert ml_model_status.get("algorithm") == "TEXT_EMBEDDING"
        except:  # noqa: E722
            raised = True
        assert raised == False, f"Raised Exception in getting {model_format} model info"
    
        if task_id:
            raised = False
            ml_task_status = None
            try:
                ml_task_status = ml_client.get_task_info(task_id, wait_until_task_done=True)
                assert ml_task_status.get("task_type") == "DEPLOY_MODEL"
                print("State:", ml_task_status.get("state"))
                assert ml_task_status.get("state") != "FAILED"
            except:  # noqa: E722
                print("Model Task Status:", ml_task_status)
                raised = True
            assert raised == False, f"Raised Exception in pulling task info for {model_format} model"
            
            # This is test is being flaky. Sometimes the test is passing and sometimes showing 500 error
            # due to memory circuit breaker.
            # Todo: We need to revisit this test.
            try:
                embedding_output = ml_client.generate_embedding(model_id, TEST_SENTENCES)
                assert len(embedding_output.get("inference_results")) == 2
                embedding_data = embedding_output["inference_results"][0]["output"][0]["data"]
            except:  # noqa: E722
                raised = True
            assert raised == False, f"Raised Exception in generating sentence embedding with {model_format} model"
    
            try:
                delete_task_obj = ml_client.delete_task(task_id)
                assert delete_task_obj.get("result") == "deleted"
            except:  # noqa: E722
                raised = True
            assert raised == False, f"Raised Exception in deleting task for {model_format} model"

            try:
                ml_client.undeploy_model(model_id)
                ml_model_status = ml_client.get_model_info(model_id)
                assert ml_model_status.get("model_state") != "UNDEPLOY_FAILED"
            except:  # noqa: E722
                raised = True
            assert raised == False, f"Raised Exception in {model_format} model undeployment"

            try:
                delete_model_obj = ml_client.delete_model(model_id)
                assert delete_model_obj.get("result") == "deleted"
            except:  # noqa: E722
                raised = True
            assert raised == False, f"Raised Exception in deleting {model_format} model"
            
    return embedding_data


def verify_embedding_data(torch_embedding_data, onnx_embedding_data):
    raised = False
    try:
        np.testing.assert_allclose(
            torch_embedding_data, 
            onnx_embedding_data, 
            rtol=RTOL_TEST, 
            atol=ATOL_TEST
        )
    except:
        raised = True
    assert raised == False, "Raised Exception in embedding verification"


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    warnings.filterwarnings("ignore", message="TracerWarning: torch.tensor")
    warnings.filterwarnings("ignore", message="using SSL with verify_certs=False is insecure.")
    
    args = sys.argv[1:]
    if len(args):
        model_id = args[0]
        embedding_dimension = 0
        if len(args) == 2:
            embedding_dimension = get_embedding_dimension(args[1])
        else:
            raise AssertionError("Too many arguments")
    else:
        raise AssertionError("Require model id")
    
    client = get_os_client()
    ml_client = MLCommonClient(client)
    
    torchscript_model_path, torchscript_model_config_path = trace_sentence_transformer_model(
        model_id, embedding_dimension, TORCH_SCRIPT_FORMAT
    )
    torch_embedding_data = upload_sentence_transformer_model(
        torchscript_model_path, torchscript_model_config_path, TORCH_SCRIPT_FORMAT
    )
    
    onnx_model_path, onnx_model_config_path = trace_sentence_transformer_model(
        model_id, embedding_dimension, ONNX_FORMAT
    )
    onnx_embedding_data = upload_sentence_transformer_model(
        torchscript_model_path, torchscript_model_config_path, ONNX_FORMAT
    )
    
    verify_sentence_embedding_data(torch_embedding_data, onnx_embedding_data)
    
    # TODO: upload_model_to_amazon_s3_prod
    # TODO: Clean data
    # TODO: Do not need to push things
    
   
    
    
    
    
    



