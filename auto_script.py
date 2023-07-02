import json
from opensearchpy import OpenSearch
import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", message="TracerWarning: torch.tensor")
warnings.filterwarnings("ignore", message="using SSL with verify_certs=False is insecure.")

import sys
import datetime

def dump():
    d = { str(datetime.datetime.now()) : sys.path}
    with open('experiment.json', 'w') as f:
        json.dump(d, f, indent=4)
    
if __name__ == '__main__':
    dump()