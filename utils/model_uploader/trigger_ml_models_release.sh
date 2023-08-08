JENKINS_URL="https://build.ci.opensearch.org"
JENKINS_TRIGGER_TOKEN=$1
BASE_DOWNLOAD_PATH=$2
VERSION=$3
FORMAT=$4

JENKINS_URL=$5 # TODO: Remove this

TIMEPASS=0
TIMEOUT=7200
RESULT="null"

JENKINS_PARAMS="{\"BASE_DOWNLOAD_PATH\":\"$BASE_DOWNLOAD_PATH\", \"VERSION\":\"$VERSION\", \"FORMAT\":\"$FORMAT\"}"

echo "Trigger ml-models Jenkins workflows"
JENKINS_REQ=$(curl -s -XPOST \
             -H "Authorization: Bearer $JENKINS_TRIGGER_TOKEN" \
             -H "Content-Type: application/json" \
             "$JENKINS_URL/generic-webhook-trigger/invoke" \
             --data "$JENKINS_PARAMS")

echo $JENKINS_PARAMS
echo $JENKINS_REQ

QUEUE_URL=$(echo $JENKINS_REQ | jq --raw-output '.jobs."ml-models-release".url')
echo "QUEUE_URL: $QUEUE_URL"
echo "Wait for jenkins to start workflow" && sleep 15

echo "Check if queue exist in Jenkins after triggering"
if [ -z "$QUEUE_URL" ] || [ "$QUEUE_URL" != "null" ]; then
    WORKFLOW_URL=$(curl -s -XGET ${JENKINS_URL}/${QUEUE_URL}api/json | jq --raw-output .executable.url)
    echo WORKFLOW_URL $WORKFLOW_URL

    echo "Use queue information to find build number in Jenkins if available"
    if [ -z "$WORKFLOW_URL" ] || [ "$WORKFLOW_URL" != "null" ]; then

        RUNNING="true"

        echo "Waiting for Jenkins to complete the run"
        while [ "$RUNNING" = "true" ] && [ "$TIMEPASS" -le "$TIMEOUT" ]; do
            echo "Still running, wait for another 5 seconds before checking again, max timeout $TIMEOUT"
            echo "Jenkins Workflow Url: $WORKFLOW_URL"
            TIMEPASS=$(( TIMEPASS + 5 )) && echo time pass: $TIMEPASS
            sleep 5
            RUNNING=$(curl -s -XGET ${WORKFLOW_URL}api/json | jq --raw-output .building)
        done

        if [ "$RUNNING" = "true" ]; then
            echo "Timed out"
            RESULT="TIMEOUT"
        else
            echo "Complete the run, checking results now......"
            RESULT=$(curl -s -XGET ${WORKFLOW_URL}api/json | jq --raw-output .result)
        fi

    fi
fi

echo "Please check jenkins url for logs: $WORKFLOW_URL"
echo "Result: $RESULT"
if [ "$RESULT" != "SUCCESS" ]; then
    exit 1
fi
