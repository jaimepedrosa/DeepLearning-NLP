def generate_alert(entities, sentiment, sentence):
    sentiment = sentiment.lower()
    sentence = sentence.lower()


    keywords = {
        "crisis": ["earthquake", "explosion", "flood", "tsunami", "war", "conflict", "violence"],
        "economic": ["inflation", "market", "gdp", "stocks", "economic", "unemployment", "recession"],
        "health": ["virus", "covid", "pandemic", "ebola", "cancer", "hospital"],
        "politics": ["election", "vote", "president", "resign", "minister", "campaign", "government"],
    }

    per_entities = [e["text"] for e in entities if e["type"] == "PER"]
    org_entities = [e["text"] for e in entities if e["type"] == "ORG"]
    loc_entities = [e["text"] for e in entities if e["type"] in ["LOC", "GPE", "GEO"]]

    
    if loc_entities:
        if any(word in sentence for word in keywords["crisis"]):
            return f"Crisis alert in {loc_entities[0].title()}: potential geopolitical or humanitarian emergency."


    if org_entities:
        if any(word in sentence for word in keywords["economic"]):
            return f"Economic signal: {org_entities[0]} involved in economic context ({sentiment})."


    if per_entities or loc_entities:
        if any(word in sentence for word in keywords["health"]):
            entity = per_entities[0] if per_entities else loc_entities[0]
            return f"Health alert: {entity} mentioned in relation to a health event."


    if per_entities or org_entities:
        if any(word in sentence for word in keywords["politics"]):
            entity = per_entities[0] if per_entities else org_entities[0]
            return f"Political development: {entity} referenced in political context."


    for ent_list, label in [(per_entities, "person"), (org_entities, "organization"), (loc_entities, "location")]:
        if ent_list:
            entity = ent_list[0]
            if sentiment == "negative":
                return f"Reputation risk: {entity} ({label}) mentioned negatively."
            elif sentiment == "neutral":
                return f"Neutral mention: {entity} ({label}) referenced in neutral tone."
            elif sentiment == "positive":
                return f"Positive spotlight: {entity} ({label}) received positive sentiment."
    
    return None
