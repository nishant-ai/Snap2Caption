import os, glob, json, random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def match_and_sample(img_dir, cap_dir, count=None, fraction=None):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    caps = sorted(glob.glob(os.path.join(cap_dir, "*.txt")))

    img_ids = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
    cap_ids = {os.path.splitext(os.path.basename(p))[0] for p in caps}
    common_ids = img_ids & cap_ids

    imgs = [p for p in imgs if os.path.splitext(os.path.basename(p))[0] in common_ids]
    caps = [p for p in caps if os.path.splitext(os.path.basename(p))[0] in common_ids]

    paired = list(zip(imgs, caps))
    random.shuffle(paired)

    if fraction:
        k = int(len(paired) * fraction)
    elif count:
        k = min(count, len(paired))
    else:
        k = len(paired)

    return paired[:k]

def load_split(base_dir, split_type, config):
    all_pairs = []
    cities = config.dataset.cities
    split_mode = config.dataset.split_mode

    for city in cities:
        img_dir = os.path.join(base_dir, city, "img_resized_1M")
        cap_dir = os.path.join(base_dir, city, "captions_resized_1M")

        if not os.path.exists(img_dir) or not os.path.exists(cap_dir):
            print(f"⚠️ Skipping {city}: missing image or caption directory.")
            continue

        if split_mode == "percent":
            fraction = config.dataset[f"{split_type}_split"]
            pairs = match_and_sample(img_dir, cap_dir, fraction=fraction)
        else:
            total = config.dataset[f"{split_type}_count"]
            per_city = total // len(cities)
            pairs = match_and_sample(img_dir, cap_dir, count=per_city)

        all_pairs.extend(pairs)

    return all_pairs

def process_pair(pair, prompt):
    img_path, cap_path = pair
    try:
        with open(cap_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip().replace("\n", " ")
            if not caption:
                return None
        return {
            "image_path": img_path,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": caption}]}
            ]
        }
    except Exception:
        return None

def prepare_jsonl(split_name, pairs, prompt):
    fn = partial(process_pair, prompt=prompt)
    with Pool(cpu_count()) as pool:
        processed = list(tqdm(pool.imap(fn, pairs), total=len(pairs)))

    data = [d for d in processed if d]
    print(f"{split_name}: {len(data)} samples ✅")
    return data
