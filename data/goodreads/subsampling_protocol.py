import csv
import random
from collections import defaultdict

random.seed(42)

input_file = 'goodreads/data/goodreads_interactions.csv'
output_file = 'data/goodreads/goodreads_sample.csv'

MAX_USERS = 3000
MAX_ITEMS = 4000

print("Pass 1: Collecting user and item counts...")
user_ratings = defaultdict(list)
item_counts = defaultdict(int)

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i % 10_000_000 == 0:
            print(f"  Read {i:,} rows...")
        user_id = row['user_id']
        book_id = row['book_id']
        rating = row['rating']
        
        # Only keep rated items (rating > 0)
        if int(rating) > 0:
            user_ratings[user_id].append(row)
            item_counts[book_id] += 1
        
        # Early stop after collecting enough data
        if len(user_ratings) > 50000:
            print(f"  Collected enough users at row {i:,}, stopping scan...")
            break

print(f"Found {len(user_ratings)} users with ratings")
print(f"Found {len(item_counts)} unique items")

# Filter to users with at least 20 ratings (for meaningful recommendations)
qualified_users = [u for u, ratings in user_ratings.items() if len(ratings) >= 20]
print(f"Users with >= 20 ratings: {len(qualified_users)}")

# Sample users
sampled_users = set(random.sample(qualified_users, min(MAX_USERS, len(qualified_users))))
print(f"Sampled {len(sampled_users)} users")

# Get all items rated by sampled users
items_from_sampled = set()
sampled_ratings = []
for user in sampled_users:
    for row in user_ratings[user]:
        items_from_sampled.add(row['book_id'])
        sampled_ratings.append(row)

print(f"Items rated by sampled users: {len(items_from_sampled)}")

# Sample items if too many
if len(items_from_sampled) > MAX_ITEMS:
    # Prioritize popular items
    item_list = list(items_from_sampled)
    item_popularity = [(item, item_counts[item]) for item in item_list]
    item_popularity.sort(key=lambda x: -x[1])
    sampled_items = set(item for item, _ in item_popularity[:MAX_ITEMS])
    print(f"Sampled {len(sampled_items)} items (by popularity)")
else:
    sampled_items = items_from_sampled

# Filter ratings to only include sampled items
final_ratings = [r for r in sampled_ratings if r['book_id'] in sampled_items]
print(f"Final ratings after item filter: {len(final_ratings)}")

# Count final stats
final_users = set(r['user_id'] for r in final_ratings)
final_items = set(r['book_id'] for r in final_ratings)
print(f"Final: {len(final_users)} users, {len(final_items)} items, {len(final_ratings)} ratings")

# Write output
print(f"Writing to {output_file}...")
with open(output_file, 'w', newline='') as f:
    fieldnames = ['user_id', 'book_id', 'is_read', 'rating', 'is_reviewed']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in final_ratings:
        writer.writerow(row)

print("Done!")