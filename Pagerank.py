# PageRank Algorithm in Python

pages = {
    'A': ['B', 'C'],
    'B': ['C'],
    'C': ['A'],
    'D': ['C']
}

damping = 0.85
num_pages = len(pages)
ranks = {page: 1/num_pages for page in pages}

for _ in range(10):  # iterate 10 times
    new_ranks = {}
    for page in pages:
        incoming = [p for p in pages if page in pages[p]]
        new_ranks[page] = (1 - damping) / num_pages + damping * sum(
            ranks[p] / len(pages[p]) for p in incoming
        )
    ranks = new_ranks

print("Final Page Ranks:")
for page, rank in ranks.items():
    print(page, ":", round(rank, 4))
