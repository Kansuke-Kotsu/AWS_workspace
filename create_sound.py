import boto3
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import os

# S3のバケット名と保存先のパスを指定
s3_bucket_name = "kansuke-huggingface-storage-01"  # あなたのS3バケット名に置き換えてください
s3_save_path = "models/musicgen-large/"  # S3内の保存先パス

# ローカルに保存するディレクトリ
save_dir = "/tmp/musicgen-large"  # EC2インスタンス上の一時ディレクトリ

# 保存先ディレクトリを作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Hugging Faceからモデルとトークナイザーをダウンロード
tokenizer = AutoTokenizer.from_pretrained("facebook/musicgen-large")
model = AutoModelForTextToWaveform.from_pretrained("facebook/musicgen-large")

# ローカルディスクにモデルとトークナイザーを保存
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

# S3クライアントを作成
s3 = boto3.client('s3')

# ローカルファイルをS3にアップロードする関数
def upload_directory_to_s3(local_directory, bucket_name, s3_directory):
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_directory, relative_path).replace("\\", "/")
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}")
            s3.upload_file(local_path, bucket_name, s3_path)

# ローカルディレクトリをS3にアップロード
upload_directory_to_s3(save_dir, s3_bucket_name, s3_save_path)

print("Model and tokenizer uploaded to S3 successfully.")
