import random
from collections import defaultdict

def diversify_recommendations(candidates, top_n):
    """
    Diversify recommendations using tags/categories, rating, and popularity.
    - Avoid duplicates.
    - Prefer high-rated and popular products.
    - Ensure coverage of different tags/categories.
    """
    # Group by tag/category
    tag_groups = defaultdict(list)
    for item in candidates:
        tag = item.get("Tag") or item.get("Category") or "Other"
        tag_groups[tag].append(item)

    # Sort each group by rating and popularity
    for tag in tag_groups:
        tag_groups[tag].sort(key=lambda x: (x["Rating"], x["Popularity"]), reverse=True)

    # Round-robin pick from each tag for diversity
    recommendations = []
    tag_list = sorted(tag_groups.keys(), key=lambda t: len(tag_groups[t]), reverse=True)
    tag_indices = {tag: 0 for tag in tag_list}
    while len(recommendations) < top_n:
        added = False
        for tag in tag_list:
            idx = tag_indices[tag]
            group = tag_groups[tag]
            if idx < len(group):
                candidate = group[idx]
                # Avoid duplicates
                if candidate["ProdID"] not in [r["ProdID"] for r in recommendations]:
                    recommendations.append(candidate)
                    tag_indices[tag] += 1
                    added = True
                    if len(recommendations) >= top_n:
                        break
        if not added:
            break

    # If not enough, fill with highest rated/popular left
    if len(recommendations) < top_n:
        all_sorted = sorted(candidates, key=lambda x: (x["Rating"], x["Popularity"]), reverse=True)
        for item in all_sorted:
            if item["ProdID"] not in [r["ProdID"] for r in recommendations]:
                recommendations.append(item)
            if len(recommendations) >= top_n:
                break

    return recommendations
