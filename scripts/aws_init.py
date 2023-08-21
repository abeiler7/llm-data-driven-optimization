import sagemaker
def init_sagemaker(sagemaker_session_bucket):
    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    print(f"sagemaker bucket: {sess.default_bucket()}")
    print(f"sagemaker session region: {sess.boto_region_name}")

    return sess