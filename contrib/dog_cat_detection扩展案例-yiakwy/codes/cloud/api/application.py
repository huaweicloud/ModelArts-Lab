from modelarts.session import Session

class Bucket:

    def __init__(self, obs_name="", bucket_name=""):
        session = Session()
        self._session = session
        self._obs_name = obs_name
        # self._obs_client = session.get_obs_client()
        self._bucket_name = bucket_name
        self._bucket_clients = []

    def get_bucket_client(self, bucket_name=""):
        return self._obs_client.bucketClient(bucket_name)

    # https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0127.html
    def download(self, src, dest='./'):
        self._session.download_data(bucket_path=src, path=dest)

    # https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0126.html
    def upload(self, src, dest):
        print("copy %s to %s" % (src, dest))
        self._session.upload_data(bucket_path=dest, path=src)

if __name__ == "__main__":
    bucket = Bucket()

