export GCSFUSE_REPO=`lsb_release -c -s`

echo "deb https://packages.cloud.google.com/apt gcsfuse-$GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

pip install -qq --no-cache-dir -U crcmod

SERVICE_ACCT=`gcloud secrets versions access latest --secret='service_account'`

gcloud config set auth/impersonate_service_account $SERVICE_ACCT

DATA_DIR=`gcloud secrets versions access latest --secret='data_dir'`
MODELS_DIR=`gcloud secrets versions access latest --secret='models_dir'`

mkdir -p $DATA_DIR
mkdir -p $MODELS_DIR

gcsfuse --implicit-dirs czc-${DATA_DIR} ${DATA_DIR}
gcsfuse --implicit-dirs czc-${MODELS_DIR} ${MODELS_DIR}
