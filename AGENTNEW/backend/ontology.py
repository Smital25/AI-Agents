CCS = {
    "machine learning": ["supervised learning", "unsupervised learning", "reinforcement learning"],
    "natural language processing": ["information extraction", "question answering", "summarization"],
    "computer vision": ["object detection", "segmentation", "image classification"],
}


def map_topics(keywords):
    tags = set()
    for k in keywords:
        k = k.lower()
        for root, subs in CCS.items():
            if k == root or k in subs:
                tags.add(root)
    return sorted(tags)
