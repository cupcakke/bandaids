import modal

VOLUME_NAME = "jaide-bench-data"
LOCAL_PATH = "tokenizer.vocab"
REMOTE_PATH = "tokenizer/tokenizer.vocab"

app = modal.App("jaide-upload-vocab")

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

with open(LOCAL_PATH, "rb") as f:
    data = f.read()

print(f"Read {len(data)} bytes from {LOCAL_PATH}")

vol.write_file(REMOTE_PATH, data)
vol.commit()

print(f"Uploaded {LOCAL_PATH} -> {VOLUME_NAME}/{REMOTE_PATH} ({len(data)} bytes)")

# verify
entries = vol.listdir("tokenizer/")
print(f"Contents of tokenizer/: {entries}")
