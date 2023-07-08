import argparse

def main(args):
    print('ARGUMENTS')
    print(args.model_id, args.model_version, args.tracing_format, args.embedding_dimension, args.pooling_mode
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_id', type=str,
                        help="Model ID for auto-tracing and uploading (e.g. sentence-transformers/msmarco-distilbert-base-tas-b)")
    parser.add_argument('model_version', type=str,
                        help="Model version number (e.g. 1.0.1)")
    parser.add_argument('tracing_format', choices=['BOTH', 'TORCH_SCRIPT', 'ONNX'],
                        help="Model format for auto-tracing")
    parser.add_argument('-ed', '--embedding_dimension',
                        type=int, nargs='?', default=None, const=None,
                        help="Embedding dimension of the model to use if it does not exist in original config.json")
    parser.add_argument('-pm', '--pooling_mode',
                        type=str, nargs='?', default=None, const=None,
                        choices=['CLS', 'MEAN', 'MAX', 'MEAN_SQRT_LEN'],
                        help="Pooling mode if it does not exist in original config.json")
    args = parser.parse_args()
    
    main(args)




