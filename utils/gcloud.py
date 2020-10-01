import os
import glob
from google.cloud import storage


def post_file_to_bucket(filename):
    # Will delete successfully posted files afterwards
    assert filename[0:5] == "gs://"
    gcs = storage.Client()
    bucket = gcs.get_bucket(filename.split("gs://")[1].split("/")[0])
    glob_path = f"{filename.split('step')[0]}step*.{filename.split('.')[1]}"
    paths = glob.glob(glob_path)
    for file in paths:
        remote_filename = "/".join(file.split("gs://")[1].split("/")[1:])
        blob = bucket.blob(remote_filename)
        blob.upload_from_filename(filename=file)
        print_fn(f"File {file} posted to gcs")
        # TODO: check for success?
        os.remove(file)

