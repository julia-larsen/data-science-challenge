from __future__ import annotations

import json
import textwrap


def main() -> None:
    path = "data/interviews.json"
    with open(path) as f:
        data = json.load(f)

    print(f"records: {len(data)}")
    print(f"keys: {list(data[0].keys())}")

    sample = data[0]
    print("sample patient_id:", sample["patient_id"])
    print("sample transcript:")
    print(textwrap.fill(sample["interview_transcript"], width=100))


if __name__ == "__main__":
    main()
