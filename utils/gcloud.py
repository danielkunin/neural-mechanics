import os
from google.cloud import storage


def post_file_to_bucket(filename):
    # Will delete successfully posted files afterwards
    assert filename[0:5] == "gs://"
    gcs = storage.Client()
    bucket = gcs.get_bucket(filename.split("gs://")[1].split("/")[0])
    remote_filename = "/".join(filename.split("gs://")[1].split("/")[1:])
    blob = bucket.blob(remote_filename)
    blob.upload_from_filename(filename=filename)
    print_fn(f"File {filename} posted to gcs")
    # TODO: check for success?
    os.remove(filename)
