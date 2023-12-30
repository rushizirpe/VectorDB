from main import database

# Initialize query processor
query_processor = QueryProcessor(database)

# Get user input (prompt)
user_input = input("Enter a prompt: ")

# Process the query and get similar meanings
similar_meanings = query_processor.execute_query(user_input, num_neighbors=5)

# Display the results
print("Similar Meanings:")
for idx, meaning in enumerate(similar_meanings):
    print(idx, meaning)