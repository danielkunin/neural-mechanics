import os
import glob
from google.cloud import storage


def lookup_tpu_ip_by_name(tpu_name, tpu_zone="us-central1-b"):
    # TODO: This is not secure and might let attacker execute arbitrary code
    gcloud_cmd = f"gcloud compute tpus list --zone={tpu_zone} | grep {tpu_name}"
    out = os.system.exec(gcloud_cmd)
    ip = out.split()[3].split(":")[0]
    return ip


def configure_env_for_tpu(tpu_ip):
    os.environ["XRT_TPU_CONFIG"] = f"tpu_worker;0;{tpu_ip}:8470"
    print(f"XRT_TPU_CONFIG env variable set to: {os.environ['XRT_TPU_CONFIG']}")


def post_file_to_bucket(filename):
    # Will delete successfully posted files afterwards
    assert filename[0:5] == "gs://"
    gcs = storage.Client()
    bucket = gcs.get_bucket(filename.split("gs://")[1].split("/")[0])
    if "step" in filename:
        glob_path = f"{filename.split('step')[0]}step*.{filename.split('.')[1]}"
        paths = glob.glob(glob_path)
    else:
        paths = [filename]
    for file in paths:
        remote_filename = "/".join(file.split("gs://")[1].split("/")[1:])
        blob = bucket.blob(remote_filename)
        blob.upload_from_filename(filename=file)
        print(f"File {file} posted to gcs")
        # TODO: check for success?
        os.remove(file)
