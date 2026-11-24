








import sqlite3

conn = sqlite3.connect("crowd_data.db")

columns = conn.execute(
    "PRAGMA table_info(crowd_data)"
).fetchall()

print("COLUMNS:")
for column in columns:
    print(column)

rows = conn.execute(
    "SELECT * FROM crowd_data LIMIT 5"
).fetchall()

print("\nSAMPLE DATA:")
for row in rows:
    print(row)

conn.close()