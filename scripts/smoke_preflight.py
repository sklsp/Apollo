"""Live preflight smoke test against a running server (temp project)."""
import io
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8765"

from PIL import Image


def main() -> None:
    # create project
    req = urllib.request.Request(
        f"{BASE}/loras/projects",
        data=json.dumps({"name": "Preflight Demo", "trigger_word": "demo"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    project = json.loads(urllib.request.urlopen(req).read())
    pid = project["id"]
    print("project:", pid)

    try:
        # upload one image
        buf = io.BytesIO()
        Image.new("RGB", (512, 512), (180, 60, 60)).save(buf, format="PNG")
        png = buf.getvalue()
        boundary = "----b"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"a.png\"\r\n"
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + png + f"\r\n--{boundary}--\r\n".encode()
        urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/loras/projects/{pid}/images", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}))

        # caption it
        cap = json.dumps({"caption": "demo, red square, flat background"}).encode()
        urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/loras/projects/{pid}/captions/a", data=cap,
            headers={"Content-Type": "application/json"}, method="PUT"))

        for label, config in (
            ("SAFE CONFIG", {"steps": 2000, "resolution": [512], "batch_size": 1}),
            ("HEAVY CONFIG", {"steps": 2000, "resolution": [1536], "batch_size": 4, "lora_rank": 64}),
        ):
            req = urllib.request.Request(
                f"{BASE}/loras/projects/{pid}/preflight",
                data=json.dumps(config).encode(),
                headers={"Content-Type": "application/json"},
            )
            report = json.loads(urllib.request.urlopen(req).read())
            print(f"\n=== {label} ===")
            print("verdict:", report["verdict"])
            print("summary:", report.get("summary", ""))
            print("est VRAM MB:", report.get("estimated_vram_mb"))
            for check in report["checks"]:
                print(f"  [{check['severity']:7}] {check['name']}: {check['detail']}")
            for rec in report["recommendations"]:
                print("  REC:", rec)
    finally:
        urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/loras/projects/{pid}", method="DELETE"))
        print("\n(cleaned up)")


if __name__ == "__main__":
    main()
