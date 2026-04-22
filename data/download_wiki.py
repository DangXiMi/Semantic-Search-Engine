import jsonlines
from datasets import load_dataset

def download_and_process_data():
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True
    )

    with jsonlines.open("data/raw/simplewiki_10k.jsonl", mode="w") as writer:
        for i, item in enumerate(dataset):
            if i >= 10000:
                break

            writer.write({
                "title": item.get("title"),
                "text": item.get("text")
            })

if __name__ == "__main__":
    download_and_process_data()