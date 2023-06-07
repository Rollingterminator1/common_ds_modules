def get_column(data, query):
    for c in data.columns:
        if query.lower() in c.lower():
            print(c)