import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client


def get_client() -> Client:
	load_dotenv()
	url = os.getenv("SUPABASE_URL")
	key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
	if not url or not key:
		raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/ANON_KEY in env")
	return create_client(url, key)


def sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			h.update(chunk)
	return h.hexdigest()


def upload_artifact(bucket: str, src: Path, dest_key: str) -> str:
	client = get_client()
	with src.open("rb") as f:
		client.storage.from_(bucket).upload(dest_key, f, file_options={"content-type": "application/octet-stream", "upsert": True})
	public_url = client.storage.from_(bucket).get_public_url(dest_key)
	return public_url


def create_bucket_if_missing(bucket: str, public: bool = True) -> None:
	client = get_client()
	try:
		# Try to list to see if it exists
		client.storage.from_(bucket).list("")
	except Exception:
		# storage3 v2 expects options dict
		client.storage.create_bucket(bucket, {"public": public})


def log_training_run(dataset: str, model_name: str, params: dict, metrics: dict, status: str = "completed", notes: Optional[str] = None) -> str:
	client = get_client()
	row = {
		"dataset": dataset,
		"model_name": model_name,
		"params": params,
		"metrics": metrics,
		"status": status,
		"notes": notes,
	}
	res = client.table("training_runs").insert(row).execute()
	return res.data[0]["id"]


def log_model_artifact(training_run_id: Optional[str], model_name: str, classes: list, storage_path: str, size_bytes: Optional[int], sha256: Optional[str]):
	client = get_client()
	row = {
		"training_run_id": training_run_id,
		"model_name": model_name,
		"classes": classes,
		"storage_path": storage_path,
		"size_bytes": size_bytes,
		"sha256": sha256,
	}
	client.table("model_artifacts").insert(row).execute()


def main():
	parser = argparse.ArgumentParser(description="Supabase utils: upload artifacts and log runs")
	sub = parser.add_subparsers(dest="cmd", required=True)

	up = sub.add_parser("upload")
	up.add_argument("--bucket", required=True)
	up.add_argument("--src", required=True)
	up.add_argument("--dest", required=True)
	up.add_argument("--dataset", required=False)
	up.add_argument("--model_name", required=False)
	up.add_argument("--classes_json", required=False, help="JSON list of class names")
	up.add_argument("--params_json", required=False, help="JSON of params")
	up.add_argument("--metrics_json", required=False, help="JSON of metrics")

	prep = sub.add_parser("prepare-bucket")
	prep.add_argument("--bucket", required=True)
	prep.add_argument("--public", action="store_true")

	args = parser.parse_args()

	if args.cmd == "prepare-bucket":
		create_bucket_if_missing(args.bucket, public=args.public)
		print(f"Bucket ensured: {args.bucket}")
	elif args.cmd == "upload":
		src = Path(args.src)
		public_url = upload_artifact(args.bucket, src, args.dest)
		file_sha = sha256_file(src)
		print(f"Uploaded to {public_url}\nsha256={file_sha}")
		if args.dataset and args.model_name:
			params = json.loads(args.params_json) if args.params_json else {}
			metrics = json.loads(args.metrics_json) if args.metrics_json else {}
			run_id = log_training_run(args.dataset, args.model_name, params, metrics)
			classes = json.loads(args.classes_json) if args.classes_json else []
			log_model_artifact(run_id, args.model_name, classes, public_url, src.stat().st_size, file_sha)
			print(f"Logged training_run={run_id}")


if __name__ == "__main__":
	main()

